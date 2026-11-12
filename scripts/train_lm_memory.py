# SPDX-License-Identifier: MIT
"""Language-model check of RTRL on the memory layers.

The memory layers (the last ``--layers`` GDN layers below the last
full-attention layer, which carry their state across steps and have LoRA) read
the ticks prepared by scripts/prepare_lm_data.py in order, and the frozen layers
above them (the full-attention layer, which attends to the instruction and this
tick's tokens, the final norm and the LM head) give the next-token loss on every
token of the step text. Only the LoRA of the memory layers learns, with Adam.
Their state is carried across episodes and games; every other layer is emptied
every tick (the GDN layers below the memory layers start each tick from the
state the instruction leaves, as in the agent), so anything the text needs from
earlier ticks comes through the memory layers. The layers below do not depend on
anything that learns, so their output is computed once when the data is loaded.

``--method rtrl`` updates after every tick with the RTRL gradient (the trace is
carried across episodes, like the state). With more than one memory layer each
layer carries the trace of its own LoRA only and takes its input as given
(layer-local RTRL), which leaves out how a lower layer's
LoRA reaches an upper layer's state across ticks. ``--method bptt`` reads
``--window`` ticks at once with the chunk kernel and updates once per window,
cutting the gradient (not the state) at the window boundary.

The last game is held out. Every ``--eval-every`` trained ticks (and before
training, which is the frozen layer since LoRA's B starts at 0) the held-out
game is read from an empty state with the weights fixed, both carrying the
state and emptying it every tick, and the mean loss per item goes to
``eval.tsv``.
"""

from vla_streaming_rl.script_setup import setup_runtime

setup_runtime()

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from vla_streaming_rl.networks.modules.gdn_memory import LORA_TARGETS, GDNMemoryLayer
from vla_streaming_rl.networks.modules.vlm_backbone import load_model
from vla_streaming_rl.script_setup import seed_everything

MODEL_ID = "Qwen/Qwen3.5-0.8B"
LORA_RANK = 8
LORA_ALPHA = 16.0


class Readout(nn.Module):
    """LoRA 付きの GDN の層の出力から、ステップの文の次のトークンの損失を出す凍結した上の層（float32）。"""

    def __init__(
        self, language_model: nn.Module, attention_layer: int, prompt: dict[str, torch.Tensor]
    ) -> None:
        super().__init__()
        self.layer = copy.deepcopy(language_model.layers[attention_layer]).float()
        assert self.layer.block_type == "full_attention", self.layer.block_type
        self.norm = copy.deepcopy(language_model.norm).float()
        self.rotary_emb = language_model.rotary_emb
        self.register_buffer("lm_head", language_model.embed_tokens.weight.detach().float())
        self.register_buffer("prompt_keys", prompt["keys"].float())
        self.register_buffer("prompt_values", prompt["values"].float())
        self.requires_grad_(False)

    def forward(
        self,
        y: torch.Tensor,
        positions: torch.Tensor,
        loss_index: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """1ティックの文の部分のLoRA 付きの GDN の層の出力 y (L, D) と M-RoPE の位置 (3, L) から、位置
        ``loss_index`` (n,) で ``target_ids`` (n,) を当てる損失 (n,)。"""
        attention = self.layer.self_attn
        L = y.shape[0]
        residual = y[None]
        h = self.layer.input_layernorm(residual)
        hidden_shape = (1, L, -1, attention.head_dim)
        query, gate = torch.chunk(
            attention.q_proj(h).view(1, L, -1, attention.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(1, L, -1)
        query = attention.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(h).view(hidden_shape)).transpose(1, 2)
        value = attention.v_proj(h).view(hidden_shape).transpose(1, 2)
        cos, sin = self.rotary_emb(h, positions[:, None, :])
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = torch.cat([self.prompt_keys, key], dim=2)
        value = torch.cat([self.prompt_values, value], dim=2)
        groups = attention.num_key_value_groups
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        # 指示の文はすべて見え、このティックのトークンは自分までが見える
        prompt_len = self.prompt_keys.shape[2]
        mask = torch.ones(L, prompt_len + L, dtype=torch.bool, device=y.device)
        mask[:, prompt_len:] = torch.tril(mask[:, prompt_len:])
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, scale=attention.scaling
        )
        out = out.transpose(1, 2).reshape(1, L, -1) * torch.sigmoid(gate)
        hidden = residual + attention.o_proj(out)
        hidden = hidden + self.layer.mlp(self.layer.post_attention_layernorm(hidden))
        logits = self.norm(hidden[0, loss_index]) @ self.lm_head.T
        return F.cross_entropy(logits, target_ids, reduction="none")


class Game:
    """前処理した1ゲーム分のティック（GPU に置く）。``x`` はLoRA 付きの GDN の層の最も下の層の入力で、
    保存した入力を、その下の凍結した層 ``lower`` に、指示の文のあとの状態 ``lower_states`` から
    毎ティック読ませて作る。"""

    def __init__(
        self,
        path: Path,
        lower: list[GDNMemoryLayer],
        lower_states: list[tuple[torch.Tensor, torch.Tensor]],
        device: str,
    ) -> None:
        data = torch.load(path)
        self.offsets = data["offsets"].tolist()
        self.num_ticks = len(self.offsets) - 1
        x = data["x"].to(device)
        if lower:
            pieces = []
            with torch.no_grad():
                for index in range(self.num_ticks):
                    y = x[self.offsets[index] : self.offsets[index + 1]].float()[None]
                    for layer, (S, h) in zip(lower, lower_states, strict=True):
                        y = layer(
                            y, torch.ones(y.shape[:2], dtype=torch.bool, device=device), S, h, False
                        )[0]
                    pieces.append(y[0].to(torch.bfloat16))
            x = torch.cat(pieces)
        self.x = x
        self.input_ids = data["input_ids"].to(device)
        self.positions = data["positions"].to(device)
        self.items = data["items"].to(device)

    def tick(self, index: int) -> dict[str, torch.Tensor]:
        start, end = self.offsets[index], self.offsets[index + 1]
        items = self.items[start:end]
        loss_index = torch.nonzero(items >= 0)[:, 0]
        # 上の層は因果的なので、損失のある最後の位置の次までで足りる（画像は文の後ろ）
        text_len = int(loss_index.max()) + 1
        return {
            "x": self.x[start:end].float(),
            "positions": self.positions[:, start : start + text_len],
            "loss_index": loss_index,
            "target_ids": self.input_ids[start + 1 : end][loss_index],
            "items": items[loss_index],
            "text_len": text_len,
        }


def tick_loss(readout: Readout, y: torch.Tensor, tick: dict) -> torch.Tensor:
    """1ティックのLoRA 付きの GDN の層の出力 y (L, D) の、文のトークンごとの損失 (n,)。上の層の活性は
    逆伝播のときに計算し直す（窓が長いとロジットが大きい）。"""
    return checkpoint(
        readout,
        y[: tick["text_len"]],
        tick["positions"],
        tick["loss_index"],
        tick["target_ids"],
        use_reentrant=False,
    )


def empty_state(layer: GDNMemoryLayer, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    S = torch.zeros(1, layer.num_heads, layer.key_dim, layer.value_dim, device=device)
    h = torch.zeros(1, layer.conv_size - 1, layer.hidden_size, device=device)
    return S, h


def empty_states(layers: list[GDNMemoryLayer], device: str) -> list:
    return [empty_state(layer, device) for layer in layers]


def read_chunked(
    layers: list[GDNMemoryLayer], x: torch.Tensor, states: list
) -> tuple[torch.Tensor, list]:
    """層を下から順に、詰め物のない入力 x (1, L, D) をまとめて読ませる。出力と次の状態。"""
    next_states = []
    for layer, (S, h) in zip(layers, states, strict=True):
        x, S, h = layer(x, torch.ones(x.shape[:2], dtype=torch.bool, device=x.device), S, h, False)
        next_states.append((S, h))
    return x, next_states


@torch.no_grad()
def prompt_states(
    lower: list[GDNMemoryLayer], prompt_x: torch.Tensor, device: str
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """凍結した下の層が、空の状態から指示の文を読んだあとの状態。"""
    return read_chunked(lower, prompt_x.float()[None], empty_states(lower, device))[1]


@torch.no_grad()
def evaluate(
    layers: list[GDNMemoryLayer],
    readout: Readout,
    game: Game,
    num_items: int,
    carry: bool,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """評価のゲームを空の状態から読み、項目ごとの損失の和と数 (num_items,)。``carry`` でなければ
    毎ティック状態を空にする。"""
    sums = torch.zeros(num_items, device=device)
    counts = torch.zeros(num_items, device=device)
    states = empty_states(layers, device)
    for index in range(game.num_ticks):
        tick = game.tick(index)
        if not carry:
            states = empty_states(layers, device)
        y, states = read_chunked(layers, tick["x"][None], states)
        losses = readout(
            y[0, : tick["text_len"]], tick["positions"], tick["loss_index"], tick["target_ids"]
        )
        sums.index_add_(0, tick["items"], losses)
        counts.index_add_(0, tick["items"], torch.ones_like(losses))
    return sums, counts


def main(args: argparse.Namespace) -> None:
    device = "cuda"
    seed_everything(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "args.txt").write_text(
        "\n".join(f"{key}\t{value}" for key, value in vars(args).items()) + "\n"
    )

    meta = torch.load(args.data / "meta.pt")
    item_names = meta["items"]
    attention_layer = meta["attention_layer"]
    first_memory_layer = attention_layer - args.layers
    assert first_memory_layer >= meta["first_layer"], (args.layers, meta["first_layer"])
    prompt = torch.load(args.data / "prompt.pt")

    vlm, _ = load_model(model_id=MODEL_ID, load_in_4bit=False, device=torch.device(device))
    language_model = vlm.model.language_model
    lower = [
        GDNMemoryLayer(language_model.layers[index], (), LORA_RANK, LORA_ALPHA)
        for index in range(meta["first_layer"], first_memory_layer)
    ]
    layers = [
        GDNMemoryLayer(language_model.layers[index], LORA_TARGETS, LORA_RANK, LORA_ALPHA)
        for index in range(first_memory_layer, attention_layer)
    ]
    readout = Readout(language_model, attention_layer, prompt).to(device)
    del vlm, language_model
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(
        [p for layer in layers for p in layer.parameters() if p.requires_grad], lr=args.lr
    )

    lower_states = prompt_states(lower, prompt["x"].to(device), device)
    game_paths = sorted(args.data.glob("game_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    assert len(game_paths) >= 2, game_paths
    train_games = [Game(path, lower, lower_states, device) for path in game_paths[:-1]]
    eval_game = Game(game_paths[-1], lower, lower_states, device)
    train_ticks = [(game, index) for game in train_games for index in range(game.num_ticks)]
    print(f"train ticks {len(train_ticks)}  eval ticks {eval_game.num_ticks}", flush=True)

    eval_file = open(out_dir / "eval.tsv", "w")
    eval_file.write("trained_ticks\tstate\titem\tloss\ttokens\n")
    train_file = open(out_dir / "train.tsv", "w")
    train_file.write("trained_ticks\tloss\telapsed_sec\n")

    def run_eval(trained_ticks: int) -> None:
        for carry in (True, False):
            sums, counts = evaluate(layers, readout, eval_game, len(item_names), carry, device)
            state = "carry" if carry else "reset"
            for name, total, count in zip(item_names, sums.tolist(), counts.tolist(), strict=True):
                eval_file.write(f"{trained_ticks}\t{state}\t{name}\t{total / count}\t{count}\n")
            all_loss = sums.sum().item() / counts.sum().item()
            eval_file.write(f"{trained_ticks}\t{state}\tall\t{all_loss}\t{counts.sum().item()}\n")
            print(f"[eval] ticks {trained_ticks}  {state}  loss {all_loss:.4f}", flush=True)
        eval_file.flush()

    start_time = time.time()
    trained_ticks = 0
    next_eval = 0
    loss_sum, loss_count = 0.0, 0
    for _ in range(args.passes):
        # 1周の終わりと次の周の始めはつながっていないので、状態とトレースを空にする
        states = empty_states(layers, device)
        for layer in layers:
            layer.reset_trace()
        position = 0
        while position < len(train_ticks):
            if trained_ticks >= next_eval:
                run_eval(trained_ticks)
                next_eval += args.eval_every
            if args.method == "rtrl":
                game, index = train_ticks[position]
                tick = game.tick(index)
                valid = torch.ones(1, tick["x"].shape[0], dtype=torch.bool, device=device)
                y = tick["x"][None]
                next_states = []
                for layer, (S, h) in zip(layers, states, strict=True):
                    y, S, h = layer(y, valid, S, h, True)
                    next_states.append((S, h))
                states = next_states
                losses = tick_loss(readout, y[0], tick)
                step_ticks = 1
            else:
                window = train_ticks[position : position + args.window]
                ticks = [game.tick(index) for game, index in window]
                y, states = read_chunked(
                    layers, torch.cat([tick["x"] for tick in ticks])[None], states
                )
                pieces = torch.split(y[0], [tick["x"].shape[0] for tick in ticks])
                losses = torch.cat(
                    [tick_loss(readout, piece, tick) for piece, tick in zip(pieces, ticks)]
                )
                step_ticks = len(window)
            loss = losses.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += losses.sum().item()
            loss_count += losses.shape[0]
            position += step_ticks
            trained_ticks += step_ticks
            if trained_ticks // args.log_every != (trained_ticks - step_ticks) // args.log_every:
                elapsed = time.time() - start_time
                train_file.write(f"{trained_ticks}\t{loss_sum / loss_count}\t{elapsed:.1f}\n")
                train_file.flush()
                print(
                    f"ticks {trained_ticks}  train loss {loss_sum / loss_count:.4f}"
                    f"  {elapsed / 60:.1f} min",
                    flush=True,
                )
                loss_sum, loss_count = 0.0, 0
    run_eval(trained_ticks)
    torch.save(
        {
            f"layer{first_memory_layer + offset}.{name}": p.detach().cpu()
            for offset, layer in enumerate(layers)
            for name, p in layer.named_parameters()
        },
        out_dir / "lora.pt",
    )
    eval_file.close()
    train_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="output of scripts/prepare_lm_data.py")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--method", choices=["rtrl", "bptt"], required=True)
    parser.add_argument("--layers", type=int, required=True, help="memory layers, 1 to 3")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--window", type=int, required=True, help="ticks per BPTT window")
    parser.add_argument("--passes", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True, help="trained ticks")
    parser.add_argument("--log-every", type=int, required=True, help="trained ticks")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    assert args.method == "bptt" or args.window == 1, "RTRL updates every tick"
    return args


if __name__ == "__main__":
    main(parse_args())
