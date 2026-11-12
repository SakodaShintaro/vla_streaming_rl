# SPDX-License-Identifier: MIT
"""Turn episodes from scripts/collect_lm_data.py into the input of the GDN layers
for the language-model check of RTRL (scripts/train_lm_memory.py).

Every tick is read the way the agent reads it with the memory layers (the stream is
emptied every step, so the VLM reads the instruction and this tick's text and
image). The GDN layers just below the last full-attention layer are the ones the
check may train (the last one, two or three), so the input of the lowest of
them, ``FIRST_LAYER``, is kept for every token of the tick, in bf16 as the replay
buffer keeps it. ``game_<g>.pt`` holds one game's ticks in order, with their
token ids, M-RoPE positions and, per token, which item of the step text the next
token belongs to (-1 where the next token is not a text token of the step, so
there is no loss). ``prompt.pt`` holds the instruction's input of ``FIRST_LAYER``
(the layers that are not trained start every tick from the state the
instruction leaves) and its keys and values at the last full-attention layer,
which every tick's text attends to.
"""

from vla_streaming_rl.script_setup import setup_runtime

setup_runtime()

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from vla_streaming_rl.networks.modules.vlm_encoder import (
    IM_START,
    IMAGE_TOKEN,
    PAD_TOKEN,
    VLMEncoder,
    step_text_items,
)
from vla_streaming_rl.scalar_obs import SCALAR_OBS_DIM

# 学習しうる最も下の GDN の層。この層の入力を保存する。上には GDN の層が 21, 22 と続き、
# 23 が最後の full attention の層
FIRST_LAYER = 20
ATTENTION_LAYER = 23

# 次のトークンがどの項目にも入らない（区切りの ", " と改行）ときの名前
SEPARATOR = "separator"


def item_names() -> list[str]:
    names = [name for name, _ in step_text_items(np.zeros(SCALAR_OBS_DIM), np.zeros(2), 0.0)]
    return names + [SEPARATOR]


def target_items(
    encoder: VLMEncoder, scalar_obs: np.ndarray, prev_action: np.ndarray, reward: float
) -> tuple[torch.Tensor, int]:
    """ステップの文のトークンごとの項目の番号 (n_text,) と、文のトークンの数。トークンの最後の
    文字がある項目にする（区切りの直後の " reward" は reward に入る）。"""
    names = item_names()
    items = step_text_items(scalar_obs, prev_action, reward)
    text = ", ".join(item for _, item in items) + "\n"
    char_item = []
    for index, (name, item) in enumerate(items):
        if index > 0:
            char_item += [names.index(SEPARATOR)] * 2
        char_item += [names.index(name)] * len(item)
    char_item.append(names.index(SEPARATOR))
    assert len(char_item) == len(text), (len(char_item), len(text))
    encoded = encoder.processor.tokenizer(text, return_offsets_mapping=True)
    labels = [char_item[end - 1] for _, end in encoded["offset_mapping"]]
    return torch.tensor(encoded["input_ids"]), torch.tensor(labels)


@torch.no_grad()
def save_prompt(encoder: VLMEncoder, instruction: str, path: Path) -> None:
    """指示の文の FIRST_LAYER の入力と、ATTENTION_LAYER の KV。KV はティックを読んだ直後の
    キャッシュの指示の文の分（どのティックでも同じ）。"""
    cache = encoder.stream_cache.layers[ATTENTION_LAYER]
    prompt_len = encoder.stream_prompt_len
    prompt_ids = encoder.processor.tokenizer(
        f"{IM_START}user\n{instruction}\n", return_tensors="pt"
    )["input_ids"].to(encoder.device)
    assert prompt_ids.shape[1] == prompt_len, (prompt_ids.shape, prompt_len)
    outputs = encoder.vlm(
        input_ids=prompt_ids,
        position_ids=torch.arange(prompt_len, device=encoder.device)
        .view(1, 1, -1)
        .expand(3, 1, -1),
        output_hidden_states=True,
        logits_to_keep=1,
    )
    torch.save(
        {
            "x": outputs.hidden_states[FIRST_LAYER][0].to(torch.bfloat16).cpu(),
            "keys": cache.keys[..., :prompt_len, :].cpu(),
            "values": cache.values[..., :prompt_len, :].cpu(),
        },
        path,
    )


def main(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    default_vla = OmegaConf.load(
        Path(__file__).parents[1] / "configs" / "agent" / "vla_off_policy.yaml"
    ).vla
    # 記憶の層を FIRST_LAYER からにすると、エンコーダはその入力を保存する表現として返す
    vla_config = OmegaConf.merge(
        default_vla,
        {
            "gdn_memory_layers": ATTENTION_LAYER - FIRST_LAYER,
            "stream_window_steps": 0,
            "image_layer_start": ATTENTION_LAYER,
            "image_layer_stop": ATTENTION_LAYER + 1,
        },
    )
    instruction = (data_dir / "instruction.txt").read_text()
    episodes = [line.split("\t") for line in (data_dir / "episodes.tsv").read_text().splitlines()]
    header, rows = episodes[0], episodes[1:]
    assert header[:2] == ["episode", "game"], header

    first = np.load(data_dir / f"ep_{int(rows[0][0]):06d}.npz")
    encoder = VLMEncoder(
        image_shapes=[tuple(first["image"].shape[1:])],
        action_dim=first["prev_action"].shape[1],
        vla_config=vla_config,
        use_memory=True,
        device="cuda",
    )
    assert encoder.memory_layer_start == FIRST_LAYER, encoder.memory_layer_start
    layer_types = encoder.vlm.config.text_config.layer_types
    assert layer_types[FIRST_LAYER:ATTENTION_LAYER] == ["linear_attention"] * (
        ATTENTION_LAYER - FIRST_LAYER
    ), layer_types
    assert layer_types[ATTENTION_LAYER] == "full_attention", layer_types
    vision_start_id = encoder.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    torch.save(
        {
            "items": item_names(),
            "instruction": instruction,
            "first_layer": FIRST_LAYER,
            "attention_layer": ATTENTION_LAYER,
        },
        out_dir / "meta.pt",
    )

    games = sorted({int(row[1]) for row in rows})
    prompt_saved = False
    for game in games:
        xs, ids, positions, items, offsets, episode_of_tick = [], [], [], [], [0], []
        for row in rows:
            if int(row[1]) != game:
                continue
            episode = int(row[0])
            data = np.load(data_dir / f"ep_{episode:06d}.npz")
            for tick in range(data["reward"].shape[0]):
                image = torch.from_numpy(data["image"][tick]).to("cuda").float() / 255.0
                scalar_obs = data["scalar_obs"][tick]
                prev_action = data["prev_action"][tick]
                reward = float(data["reward"][tick])
                representation = encoder.represent(
                    [image], scalar_obs, prev_action, reward, instruction, True
                )
                inputs, step_positions = encoder.step_inputs(
                    [image], scalar_obs, prev_action, reward, encoder.stream_prompt_len
                )
                step_ids = inputs["input_ids"][0]
                kinds = representation[:, -1]
                length = int((kinds != PAD_TOKEN).sum())
                assert length == step_ids.shape[0], (length, step_ids.shape)
                assert int((kinds == IMAGE_TOKEN).sum()) == encoder.num_positions
                text_ids, text_items = target_items(encoder, scalar_obs, prev_action, reward)
                n_text = text_ids.shape[0]
                assert torch.equal(step_ids[:n_text].cpu(), text_ids), (step_ids, text_ids)
                assert int(step_ids[n_text]) == vision_start_id
                # 位置 i の損失は次のトークン i + 1 を当てる。最初の文のトークンは指示の文の
                # 最後の位置が当てるので、ここには入れない
                tick_items = torch.full((length,), -1, dtype=torch.long)
                tick_items[: n_text - 1] = text_items[1:]
                xs.append(representation[:length, :-1].to(torch.bfloat16).cpu())
                ids.append(step_ids.cpu())
                positions.append(step_positions[:, 0].cpu())
                items.append(tick_items)
                offsets.append(offsets[-1] + length)
                episode_of_tick.append(episode)
                if not prompt_saved:
                    save_prompt(encoder, instruction, out_dir / "prompt.pt")
                    prompt_saved = True
        torch.save(
            {
                "x": torch.cat(xs),
                "input_ids": torch.cat(ids),
                "positions": torch.cat(positions, dim=1),
                "items": torch.cat(items),
                "offsets": torch.tensor(offsets),
                "episode": torch.tensor(episode_of_tick),
            },
            out_dir / f"game_{game}.pt",
        )
        print(f"game {game}: {len(offsets) - 1} ticks, {offsets[-1]} tokens", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="output of scripts/collect_lm_data.py")
    parser.add_argument("out_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args.data_dir.resolve(), cli_args.out_dir.resolve())
