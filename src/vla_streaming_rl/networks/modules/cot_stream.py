# SPDX-License-Identifier: MIT
"""A frozen VLM kept mid-thought across environment steps.

The chain of thought is the slow loop. It is prefilled from one frame and then
advanced by a fixed budget of tokens per environment step, so a single line of
reasoning spans many control ticks. When it ends -- EOS, or ``max_len`` tokens --
the next advance prefills again from whatever frame is current, starting a fresh
chain on a fresh image. That prefill carries the chain that just ended as the
model's own turn, so the new one picks up where the last left off instead of
opening on the scene from scratch.

What leaves this module is not text but the activation feeding the VLM's lm_head
at each generated position: the state the model was in when it chose that token,
which carries far more than the token id does. Every hidden state behind it is
kept -- the embedding and each layer's output -- leaving which depth to read to
a weighting the network trains. Nothing here trains and nothing here is
differentiable, so downstream the chain is an ordinary observation stream,
stored in the replay buffer next to the image.

Not an ``nn.Module`` on purpose: registering it would put a frozen 0.8B model
into the network's ``parameters()`` and its ``state_dict()``.
"""

import time

import torch
from omegaconf import DictConfig

from vla_streaming_rl.agents.prompt import PromptBuilder, assistant_turn

from .vlm_backbone import load_model


class CoTStream:
    def __init__(
        self,
        high_level_config: DictConfig,
        prompt_builder: PromptBuilder,
        device: torch.device,
    ) -> None:
        tokens_per_step = high_level_config.cot_tokens_num
        max_len = high_level_config.max_new_tokens
        assert tokens_per_step >= 1, f"tokens_per_step must be positive; got {tokens_per_step}"
        assert max_len >= tokens_per_step, (
            f"max_len {max_len} below the per-step budget {tokens_per_step}: "
            "every step would restart the chain"
        )
        self.model, self.processor = load_model(
            model_id=high_level_config.model_id,
            use_lora=False,
            load_in_4bit=high_level_config.load_in_4bit,
            device=device,
        )
        self.model.eval().requires_grad_(False)
        self.tokens_per_step = tokens_per_step
        self.max_len = max_len
        self.temperature = high_level_config.temperature
        # The conversation is the agent's; a chain reads it on the steps it
        # restarts and writes its own turn back when it ends.
        self.prompt_builder = prompt_builder
        self.device = device
        text_config = self.model.config.text_config
        self.hidden_size = text_config.hidden_size
        # The embedding plus every layer's output.
        self.layers_num = text_config.num_hidden_layers + 1
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        # ChainGenerator と同じ仕組み: キャッシュはモデルがプレフィルで作る
        # DynamicCache で、チェーンの仕切り直しごとに作り直す。デコード1
        # ステップの入力は使い回しの小さなテンソルに書き込む。
        self._token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self._position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        self.reset()

    def reset(self) -> None:
        """Drop the chain. The next advance prefills from the frame it is given."""
        self._needs_prefill = True
        self._hidden = None
        self._next_token = None
        self._tokens = []
        self._last_conversation = []
        # The prompt the chain was prefilled on, and what this step's tokens
        # cost. Reported to the render panel, not used by the chain itself.
        self._input_tokens = 0
        self._msec = 0.0
        self._position = 0
        self._cache = None

    @torch.inference_mode()
    def age(self) -> int:
        """How many environment steps ago what is being read was generated.

        Always 0 here: this mode issues its tokens on the step that reads them,
        which is the whole difference from `CoTBatch`. Kept so both modes answer
        the same question.
        """
        return 0

    def advance(self) -> torch.Tensor:
        """The ``tokens_per_step`` activations this environment step issues.

        The builder's conversation is read only where a chain restarts, and only
        its tail -- the standing task, the chain that ended last, and the turn
        being opened on -- to keep the prefill cost per restart flat however
        long the conversation has grown. That the frame is read there and
        nowhere else is what makes the chain the slow loop.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16.
        """
        start = time.perf_counter()
        activations = []
        while len(activations) < self.tokens_per_step:
            if self._needs_prefill:
                self._prefill(self.prompt_builder.conversation())
            activations.append(self._hidden)
            position = self._position
            self._write_step_inputs(position)
            outputs = self._forward_step()
            self._position = position + 1
            self._consume(outputs.hidden_states, outputs.logits[0, -1])
        self._msec = (time.perf_counter() - start) * 1000.0
        return torch.stack(activations)

    def _prefill(self, conversation: list[dict]) -> None:
        # 送るのは、恒常のタスク・直前に終わったチェーン（モデル自身の発言と
        # して）・いま開くターンだけ。それより前は捨てる。会話がどれだけ
        # 伸びても仕切り直しのプレフィルは一定の長さで、載せるフレームも
        # 現在の1枚だけになる。
        turn = conversation[-1]
        image = turn["content"][1]["image"]
        replies = [reply for reply in conversation if reply["role"] == "assistant"]
        current = {
            "role": "user",
            "content": [turn["content"][0], {"type": "image"}, turn["content"][2]],
        }
        # Thinking off: with the <think> block left open the model spends the
        # chain reasoning about the request rather than about the scene.
        text = self.processor.apply_chat_template(
            [conversation[0]] + replies[-1:] + [current],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text],
            images=[image.detach().float().clamp(0.0, 1.0)],
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        self._tokens = []
        self._last_conversation = conversation
        self._input_tokens = int(prompt_len)
        self._needs_prefill = False
        outputs = self.model(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
        )
        self._cache = outputs.past_key_values
        self._position = prompt_len
        self._consume(outputs.hidden_states, outputs.logits[0, -1])

    def _write_step_inputs(self, position: int) -> None:
        """デコード1ステップの入力バッファを書く。位置はモデル任せにせず
        こちらで数え、画像トークン分のずれはプレフィルが残した rope_deltas
        を足して合わせる。"""
        self._token.copy_(self._next_token)
        self._cache_position.fill_(position)
        rope_deltas = self.model.model.rope_deltas
        self._position_ids.copy_(
            (self._cache_position.view(1, 1, -1) + rope_deltas.unsqueeze(0)).expand(3, -1, -1)
        )

    def _forward_step(self):
        return self.model(
            input_ids=self._token,
            past_key_values=self._cache,
            use_cache=True,
            output_hidden_states=True,
            cache_position=self._cache_position,
            position_ids=self._position_ids,
        )

    def _consume(self, hidden_states, logits: torch.Tensor) -> None:
        """Keep the position's activation at every depth behind it --
        (layers_num, hidden_size) -- sample the token it implies, and decide
        whether the chain lives on.

        """
        self._hidden = torch.stack([state[0, -1] for state in hidden_states]).to(torch.bfloat16)
        probs = torch.softmax(logits.float() / self.temperature, dim=-1)
        self._next_token = torch.multinomial(probs, 1).view(1, 1)
        self._tokens.append(self._next_token.item())
        ended = self._tokens[-1] == self.eos_token_id or len(self._tokens) >= self.max_len
        if ended:
            self.prompt_builder.add_reply(self.text())
            self._needs_prefill = True

    def stats(self) -> dict:
        """What the chain costs: the prompt it was prefilled on, the tokens it
        has written since, and the wall time this step's tokens took."""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": len(self._tokens),
            "msec": self._msec,
        }

    def text(self) -> str:
        """The chain as written so far, for logging."""
        return self.processor.tokenizer.decode(self._tokens, skip_special_tokens=True).strip()

    def exchange(self) -> list[dict]:
        """The chain in progress as it stands: the conversation it was prefilled
        on and what it has written since, for the render panel."""
        return self._last_conversation + [assistant_turn(self.text())]
