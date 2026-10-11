# SPDX-License-Identifier: MIT
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from ..agents.prompt import assistant_turn, read_action_reply, user_turn
from ..replay_buffer import ReplayBufferData
from .interface import (
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from .modules.head_output import HeadOutput
from .modules.value_head import DistributionalValueHead
from .modules.vlm_backbone import load_model
from .modules.vlm_inputs import build_vlm_inputs, render_conversation


@dataclass
class LossTerms:
    """リプレイバッチ1つ分の損失と、それを計算する途中で得た推論結果。"""

    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    info: dict
    state: torch.Tensor
    action_chunk: torch.Tensor
    next_state: torch.Tensor
    next_action: torch.Tensor
    next_critic_out: HeadOutput


def _rows(data: ReplayBufferData) -> tuple[torch.Tensor, ...]:
    """What a replay batch holds of each tick's prompt, in the order
    ``_act`` reads them."""
    return (
        data.observations,
        data.system_token_ids,
        data.turn_token_ids,
        data.reply_token_ids,
        data.episode_step,
    )


class VLMActorCriticWithActionValue(NetworkInterface):
    """VLM backbone + DiffusionPolicy + Action Value critic.

    Every tick the VLM first writes the reply the zero-shot controller would --
    its action -- to the conversation of the episode so far.
    The state is then read off the same VLM run over that conversation with the
    reply in it, so the heads see what the VLM saw, thought and proposed. The
    VLM is frozen, so what it gives a tick is computed once, when the tick is
    acted on, and kept in the replay buffer: learning reads it back and never
    runs the VLM. Two
    candidate actions come out of a tick, the one the reply names and the policy
    head's, and the critic picks: the head's runs only where it is valued above
    the reply's by more than ``select_margin``. ``.inf`` there leaves the
    zero-shot controller alone at the wheel and ``-.inf`` the head; while the
    buffer fills it is always the reply's.

    Architecture:
    - VLM (Qwen3.5/Qwen-VL, frozen): processes images + text.
    - Stored representation: the mean across every VLM hidden state (embedding
      + each transformer layer output), AdaptiveAvgPool1d over the tokens to
      ``num_state_queries`` of them.
    - State extractor: a per-token Linear projection of that representation.
    - DiffusionPolicy: denoises actions conditioned on extracted state.
    - Critic: Q(state, action) with dueling architecture.
    """

    def __init__(
        self,
        *,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        value_head_factory: Callable[[int, int], DistributionalValueHead],
        horizon: int,
        critic_loss_weight: float,
        policy_head_factory,
        vla_config: DictConfig,
        pad_token_id: int,
        parse_action_text,
        action_low: np.ndarray,
        action_high: np.ndarray,
        decision_fps: float,
    ) -> None:
        super().__init__()
        self.seq_len = vla_config.seq_len
        # 状態の元になる表現は、行動を決めたステップでだけ計算して保存する。チャンクの途中の
        # ステップには保存されないので、毎ステップ行動を決める設定でだけ使える
        assert horizon == 1, f"the VLM network stores a state on every tick; got horizon {horizon}"
        self.horizon = horizon
        self.action_dim = action_space_shape[0]
        self.observation_space_shape = observation_space_shape
        self.critic_loss_weight = critic_loss_weight
        # The prompt of a tick is a conversation of the last ``seq_len`` buffer
        # rows ending on the tick, each a frame under its own text answered by
        # the reply written on it. A reply is written on every tick.
        # 返答の生成。思考の量は "none" なら思考を閉じた形、それ以外はテンプレートの
        # reasoning_effort に渡す
        assert vla_config.max_new_tokens >= 1, vla_config.max_new_tokens
        assert vla_config.temperature >= 0.0, vla_config.temperature
        assert vla_config.reasoning_effort in ("none", "low", "medium", "xhigh"), (
            vla_config.reasoning_effort
        )
        self.max_new_tokens = vla_config.max_new_tokens
        self.temperature = vla_config.temperature
        # 返答の行動と頭の行動を行動価値で選ぶときの余裕。頭の行動の価値が返答の行動を
        # これより大きく上回るときだけ、頭の行動を実行する
        self.select_margin = vla_config.select_margin
        self.parse_action_text = parse_action_text
        self.action_low = action_low
        self.action_high = action_high
        self.decision_fps = decision_fps

        # Load VLM
        device = "cuda"
        self.use_lora = bool(vla_config.use_lora)
        self.vlm_model, self.processor = load_model(
            vla_config.model_id,
            use_lora=self.use_lora,
            load_in_4bit=vla_config.load_in_4bit,
            device=device,
        )
        self.device = device
        # 返答を書かせるときにテンプレートへ渡す引数。思考の量の段階（reasoning_effort）を
        # 持たないテンプレートには、思考の有無だけを渡す
        self.generation_template_kwargs = {"enable_thinking": vla_config.reasoning_effort != "none"}
        if vla_config.reasoning_effort != "none" and (
            "reasoning_effort" in self.processor.chat_template
        ):
            self.generation_template_kwargs["reasoning_effort"] = vla_config.reasoning_effort

        # VLM config
        vlm_cfg = self.vlm_model.config.text_config
        self.vlm_hidden_size = vlm_cfg.hidden_size
        self.pad_token_id = pad_token_id

        self.num_state_queries = vla_config.num_state_queries

        self.state_out_proj = nn.Linear(self.vlm_hidden_size, vla_config.state_out_dim).to(device)
        # AdaptiveAvgPool1d fixes the token count to num_state_queries, so
        # state_dim is determined purely by config.
        state_dim = vla_config.num_state_queries * vla_config.state_out_dim

        self.policy_head = policy_head_factory(state_dim=state_dim, action_dim=self.action_dim)

        # Critic: Q(state, action)
        self.value_head = value_head_factory(state_dim, self.action_dim)

        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def stored_image_shape(self) -> tuple[int, ...]:
        """The image itself: the VLM reads the frames in its prompt."""
        return tuple(self.observation_space_shape)

    def to_stored_image(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def stored_state_shape(self) -> tuple[int, ...]:
        """VLM の表現を ``num_state_queries`` トークンに均したもの。射影は学習するので、
        その手前を保存する。"""
        return (self.num_state_queries, self.vlm_hidden_size)

    def observe_scalar_obs(
        self,
        velocity_x: float,
        velocity_y: float,
        velocity_z: float,
        episode_return: float,
        pass_mark: float,
        remaining_return: float,
        global_step: float,
        episode_step: float,
        health: float,
    ) -> None:
        del velocity_x, velocity_y, velocity_z, episode_return, pass_mark
        del remaining_return, global_step, episode_step, health

    def tokenize(self, text: str) -> list[int]:
        return self.processor.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, token_ids: torch.Tensor) -> list[str]:
        """Strings back from their stored token IDs, (N, max_prompt_tokens)."""
        results = []
        for ids in token_ids:
            valid_ids = ids[ids != self.pad_token_id].tolist()
            results.append(self.processor.tokenizer.decode(valid_ids, skip_special_tokens=True))
        return results

    def _conversation(
        self,
        observations: torch.Tensor,
        system_token_ids: torch.Tensor,
        turn_token_ids: torch.Tensor,
        reply_token_ids: torch.Tensor,
        episode_step: torch.Tensor,
    ) -> list[dict]:
        """The conversation of the window's last tick, up to the tick's own turn.

        Its turns are the window's rows, each a frame under its timestamp and
        text, the earlier ones answered by the reply their tick wrote. Only the
        last tick's own episode is kept, as the zero-shot controller starts
        every episode on an empty conversation.
        """
        rows = list(range(observations.shape[0]))[-self.seq_len :]
        # エピソードの中ではステップ数が増えていく。増えていない所より前は別のエピソード
        steps = [float(episode_step[row, 0]) for row in rows]
        first = len(rows) - 1
        while first > 0 and steps[first - 1] < steps[first]:
            first -= 1
        rows, steps = rows[first:], steps[first:]
        turn_texts = self._decode(turn_token_ids[rows])
        reply_texts = self._decode(reply_token_ids[rows])
        system_text = self._decode(system_token_ids[rows[-1:]])[0]
        conversation = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
        for i, row in enumerate(rows):
            conversation.append(
                user_turn(steps[i] / self.decision_fps, observations[row], turn_texts[i])
            )
            if i < len(rows) - 1 and reply_texts[i]:
                conversation.append(assistant_turn(reply_texts[i]))
        return conversation

    def _generate(self, conversation: list[dict]) -> tuple[str, int, int, float]:
        """``conversation`` への返答を書く。ゼロショットの制御器が書くものと同じ。返答と、
        読んだトークン数、書いたトークン数、かかった時間（ミリ秒）を返す。"""
        start = time.perf_counter()
        text, frames = render_conversation(
            self.processor, conversation, True, self.generation_template_kwargs
        )
        inputs = self.processor(
            text=[text], images=[frames], do_rescale=False, return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        sampling = (
            {"do_sample": True, "temperature": self.temperature}
            if self.temperature > 0.0
            else {"do_sample": False}
        )
        output = self.vlm_model.generate(**inputs, max_new_tokens=self.max_new_tokens, **sampling)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        written = output[0, prompt_tokens:]
        reply = self.processor.tokenizer.decode(written, skip_special_tokens=True).strip()
        return reply, prompt_tokens, len(written), (time.perf_counter() - start) * 1000.0

    def _text_actions(self, replies: list[str], like: torch.Tensor) -> tuple[torch.Tensor, list]:
        """各返答の <action> が指す行動を、方策の正規化した行動空間のチャンク
        (B, horizon, action_dim) にしたものと、読めたかどうか。"""
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        actions, parsed = [], []
        for reply in replies:
            _, action, parse_ok = read_action_reply(
                reply, self.parse_action_text, self.action_low, self.action_high
            )
            actions.append((action - bias) / scale)
            parsed.append(parse_ok)
        chunk = torch.from_numpy(np.stack(actions)).to(like.device, like.dtype)
        return chunk.unsqueeze(1).expand(-1, self.horizon, -1), parsed

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        return self._act(
            data.s_seq,
            data.system_token_ids_seq,
            data.turn_token_ids_seq,
            data.reply_token_ids_seq,
            data.episode_step_seq,
            data.warmup,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        # 次の状態は最後のステップに保存された表現と返答から作る
        replies = self._decode(data.reply_token_ids[:, -1])
        terms = self._loss_terms(data, data.states[:, -1], replies)
        return LossResult(loss=terms.total_loss, info=terms.info)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        # 最後のステップはまだ行動を決めていないので、ここで返答を書いて表現を作る
        with torch.inference_mode():
            infer_result = self._act(*_rows(data), False)
        terms = self._loss_terms(
            data, infer_result.stored_state.unsqueeze(0).clone(), [infer_result.text]
        )
        # -Q(s,a) for eligibility trace backward (detached from encoder)
        neg_value_detached = -self.value_head.scalar_value(
            terms.state.detach(), terms.action_chunk.detach()
        ).mean()
        return InferLossResult(
            infer_result=infer_result,
            loss_result=LossResult(loss=terms.total_loss, info=terms.info),
            et_info=EligibilityTraceInfo(
                # Actor-only loss (no critic component)
                actor_entropy_loss=terms.actor_loss,
                neg_value=neg_value_detached,
                delta=terms.info["losses/delta"],
            ),
        )

    ####################
    # Internal methods #
    ####################

    def _act(
        self,
        observations: torch.Tensor,
        system_token_ids: torch.Tensor,
        turn_token_ids: torch.Tensor,
        reply_token_ids: torch.Tensor,
        episode_step: torch.Tensor,
        warmup: bool,
    ) -> InferResult:
        """最後のステップの行動を決める。会話に返答を書き、その返答まで含めた会話の表現から
        状態を作り、返答の行動と頭の行動を行動価値で選ぶ。"""
        assert observations.shape[0] == 1, "Batch size must be 1 for inference"
        conversation = self._conversation(
            observations[0],
            system_token_ids[0],
            turn_token_ids[0],
            reply_token_ids[0],
            episode_step[0],
        )
        reply, prompt_tokens, completion_tokens, msec = self._generate(conversation)
        exchange = conversation + [assistant_turn(reply)]
        representation = self._representation(exchange)
        state = self._state(representation.unsqueeze(0))
        head_action, _ = self.policy_head.get_action(state)
        critic_out = self.value_head(state, head_action)
        text_action, parsed = self._text_actions([reply], head_action)
        q_head = self.value_head.scalar_value(state, head_action).item()
        q_text = self.value_head.scalar_value(state, text_action).item()
        # バッファを貯めている間は返答の行動だけで動く
        text_chosen = warmup or q_head - q_text <= self.select_margin

        return InferResult(
            action=text_action if text_chosen else head_action,
            value_report={
                **self.value_head.value_report(critic_out.output),
                "select/q_head": q_head,
                "select/q_text": q_text,
                "select/text_chosen": float(text_chosen),
                "vlm/parse_failed": float(not parsed[0]),
                "vlm/prompt_tokens": float(prompt_tokens),
                "vlm/completion_tokens": float(completion_tokens),
                "vlm/msec": msec,
            },
            rnn_state=self._dummy_state.clone(),
            features=state,
            stored_state=representation,
            text=reply,
            text_status=(
                f"in {prompt_tokens} tok   out {completion_tokens} tok   {msec:.0f} ms   "
                f"parse {'ok' if parsed[0] else 'failed'}   "
                f"{'reply' if text_chosen else 'head'} action runs"
            ),
            exchange=exchange,
        )

    def _loss_terms(
        self, data: ReplayBufferData, next_representation: torch.Tensor, next_replies: list[str]
    ) -> LossTerms:
        """リプレイバッチの損失。状態は、行動を決めたステップで保存した VLM の表現から作る。
        現在の状態はチャンクの手前のステップのもので、次の状態は ``next_representation``
        （最後のステップのもの）から作る。"""
        # 次の状態で実行される行動は、推論のときと同じく、そのステップの返答の行動と頭の
        # 行動から行動価値で選んだもの
        with torch.inference_mode():
            next_state = self._state(next_representation)
            next_action, _ = self.policy_head.get_action(next_state)
            text_action, _ = self._text_actions(next_replies, next_action)
            q_head = self.value_head.scalar_value(next_state, next_action)
            q_text = self.value_head.scalar_value(next_state, text_action)
            text_chosen = q_head - q_text <= self.select_margin
            next_action = torch.where(text_chosen.view(-1, 1, 1), text_action, next_action)
            next_critic_out = self.value_head(next_state, next_action)
        target_value = self.value_head.compute_target_value(
            next_critic_out.output,
            data.rewards[:, -self.horizon :],
            data.dones[:, -self.horizon :],
        )

        state = self._state(data.states[:, -self.horizon - 1])
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)
        critic_loss, critic_info = self.value_head.compute_critic_loss(
            state, action_chunk, target_value
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state, action_chunk, value_head=self.value_head
        )
        info = {f"losses/{key}": value for key, value in {**critic_info, **actor_info}.items()}
        return LossTerms(
            total_loss=self.critic_loss_weight * critic_loss + actor_loss,
            actor_loss=actor_loss,
            info=info,
            state=state,
            action_chunk=action_chunk,
            next_state=next_state,
            next_action=next_action,
            next_critic_out=next_critic_out,
        )

    def _get_vlm_model_inner(self) -> nn.Module:
        """Get the inner Qwen3_5Model (handles PEFT wrapping)."""
        if self.use_lora:
            return self.vlm_model.model.model
        return self.vlm_model.model

    def _state(self, representation: torch.Tensor) -> torch.Tensor:
        """保存した表現 (B, num_state_queries, hidden_size) を射影して、頭が読む状態にする。"""
        projected = self.state_out_proj(representation.to(torch.float32))
        return projected.flatten(start_dim=1)

    @torch.no_grad()
    def _representation(self, exchange: list[dict]) -> torch.Tensor:
        """返答まで含めた会話を VLM に通し、全層（埋め込みと各層の出力）の隠れ状態の平均を
        ``num_state_queries`` トークンに均したもの、(num_state_queries, hidden_size)。"""
        text, frames = render_conversation(self.processor, exchange, False, {})
        inputs = build_vlm_inputs(self.processor, [text], [frames], self.device)
        vlm_inner = self._get_vlm_model_inner()

        # The vision tower's tokens are scattered into the placeholder positions
        # of the text embeddings, so the language model is fed embeddings only.
        inputs_embeds = vlm_inner.get_input_embeddings()(inputs["input_ids"])
        visual = vlm_inner.visual
        pixel_values = inputs["vision_pixel_values"].type(visual.dtype)
        vision_output = visual(pixel_values, grid_thw=inputs["vision_grid_thw"])
        vision_embeds = vision_output.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
        vision_mask = (
            (inputs["input_ids"] == inputs["vision_token_id"])
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        inputs_embeds = inputs_embeds.masked_scatter(vision_mask, vision_embeds)

        # 3D position_ids are what m-rope reads the image token positions off.
        position_ids = vlm_inner.compute_3d_position_ids(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            video_grid_thw=inputs["video_grid_thw"],
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            past_key_values=None,
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        # The outer model, not the language model, so lm_head and the cache
        # wrapping are handled for us.
        outputs = self.vlm_model.forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        # 全層を一度に積まず、足し込みながら平均する
        mean = None
        for hidden_state in outputs.hidden_states:
            term = hidden_state[0].to(torch.float32)
            mean = term if mean is None else mean + term
        mean = mean / len(outputs.hidden_states)  # (T, hidden_size)
        pooled = F.adaptive_avg_pool1d(mean.T.unsqueeze(0), self.num_state_queries)
        return pooled[0].T.to(torch.bfloat16)
