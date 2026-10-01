# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

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


def _user_turn(image: torch.Tensor, text: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
    }


def _rows(data: ReplayBufferData) -> tuple[torch.Tensor, ...]:
    """What a replay batch holds of each tick's prompt, in the order
    ``_prompts_at`` reads them."""
    return (
        data.observations,
        data.system_token_ids,
        data.turn_token_ids,
        data.reply_token_ids,
    )


class VLMActorCriticWithActionValue(NetworkInterface):
    """VLM backbone + DiffusionPolicy + Action Value critic.

    Architecture:
    - VLM (Qwen3.5/Qwen-VL, frozen unless LoRA): processes images + text.
    - State extractor: softmax-weighted sum across every VLM hidden state
      (embedding + each transformer layer output) -> per-token Linear projection
      -> AdaptiveAvgPool1d to ``num_state_queries`` tokens.
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
        steps_per_reply: int,
    ) -> None:
        super().__init__()
        self.seq_len = vla_config.seq_len
        self.horizon = horizon
        self.action_dim = action_space_shape[0]
        self.observation_space_shape = observation_space_shape
        self.critic_loss_weight = critic_loss_weight
        # The prompt of a tick is the conversation the zero-shot controller
        # would read there: its turns are the buffer rows ``steps_per_reply``
        # apart ending on the tick, as many as ``seq_len`` ticks hold, each
        # a frame under its own text answered by the reply the high-level
        # policy wrote on its tick.
        assert steps_per_reply >= 1, steps_per_reply
        self.steps_per_reply = steps_per_reply

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

        # VLM config
        vlm_cfg = self.vlm_model.config.text_config
        vlm_hidden_size = vlm_cfg.hidden_size
        num_layers = vlm_cfg.num_hidden_layers
        # Input-independent learnable logits over all (embedding + per-layer) hidden
        # states; softmax-weighted sum forms the representation used downstream.
        self.layer_logits = nn.Parameter(torch.zeros(num_layers + 1, device=device))
        self.pad_token_id = pad_token_id

        self.num_state_queries = vla_config.num_state_queries

        self.state_out_proj = nn.Linear(vlm_hidden_size, vla_config.state_out_dim).to(device)
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

    def to_stored_subtask(self, activations: torch.Tensor) -> torch.Tensor:
        """活性は読まない。高レベル方策の返答は会話のテキストとして届く。"""
        del activations
        return torch.zeros(self.subtask_shape)

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

    def _prompts_at(
        self,
        observations: torch.Tensor,
        system_token_ids: torch.Tensor,
        turn_token_ids: torch.Tensor,
        reply_token_ids: torch.Tensor,
        slot: int,
    ) -> tuple[list[str], list[list[torch.Tensor]]]:
        """The conversation of each batch element's ``slot`` tick, rendered.

        Its turns are the rows ``steps_per_reply`` apart ending on the slot:
        each a frame under its own text, the earlier ones answered by the reply
        their tick wrote, as many as ``seq_len`` ticks hold; an episode boundary
        does not cut them, so what the episodes before did and what came of them
        stays in view.
        """
        cursor = slot % observations.shape[1]
        stride = self.steps_per_reply
        system_texts = self._decode(system_token_ids[:, cursor])
        texts, images = [], []
        for b in range(observations.shape[0]):
            # Capped by the state window rather than the rows behind the slot,
            # so the next state's prompt holds no turn the current's could not.
            turns_num = min((self.seq_len - 1) // stride, cursor // stride)
            rows = [cursor - stride * k for k in range(turns_num, -1, -1)]
            turn_texts = self._decode(turn_token_ids[b, rows])
            reply_texts = self._decode(reply_token_ids[b, rows[:-1]])
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_texts[b]}]}
            ]
            for row, turn_text, reply_text in zip(rows, turn_texts, reply_texts, strict=False):
                conversation.append(_user_turn(observations[b, row], turn_text))
                if reply_text:
                    conversation.append(
                        {"role": "assistant", "content": [{"type": "text", "text": reply_text}]}
                    )
            conversation.append(_user_turn(observations[b, rows[-1]], turn_texts[-1]))
            # ゼロショットの高レベル方策と同じく、モデル自身の思考は閉じる
            text, frames = render_conversation(self.processor, conversation, False)
            texts.append(text)
            images.append(frames)
        return texts, images

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        texts, images = self._prompts_at(
            data.s_seq,
            data.system_token_ids_seq,
            data.turn_token_ids_seq,
            data.reply_token_ids_seq,
            -1,
        )
        state, action, critic_out = self._infer(texts, images)

        return InferResult(
            action=action,
            value_report=self.value_head.value_report(critic_out.output),
            rnn_state=data.rnn_state,
            features=state,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        terms = self._loss_terms(data)
        return LossResult(loss=terms.total_loss, info=terms.info)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        terms = self._loss_terms(data)
        # -Q(s,a) for eligibility trace backward (detached from encoder)
        neg_value_detached = -self.value_head.scalar_value(
            terms.state.detach(), terms.action_chunk.detach()
        ).mean()
        return InferLossResult(
            infer_result=InferResult(
                action=terms.next_action,
                value_report=self.value_head.value_report(terms.next_critic_out.output),
                rnn_state=self._dummy_state.clone(),
                features=terms.next_state,
            ),
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

    def _loss_terms(self, data: ReplayBufferData) -> LossTerms:
        """リプレイバッチの損失。各状態のプロンプトはその状態のステップに保存されたもので、
        次の状態は最後のステップ、現在の状態はチャンクの手前のステップのもの。"""
        next_state, next_action, next_critic_out = self._infer(*self._prompts_at(*_rows(data), -1))
        target_value = self.value_head.compute_target_value(
            next_critic_out.output,
            data.rewards[:, -self.horizon :],
            data.dones[:, -self.horizon :],
        )

        state = self._forward_prompt(*self._prompts_at(*_rows(data), -self.horizon - 1))
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

    def _state_from_hidden_states(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Softmax-weighted sum across embedding + per-layer hidden states -> state."""
        # Projecting each layer before the weighted sum rather than after is the
        # same number by linearity, but it never holds all (num_layers + 1)
        # hidden states at once: the running sum is state_out_dim wide, not
        # vlm_hidden_size, which for a long prompt is the difference between
        # gigabytes and megabytes. The bias is added once, since the softmax
        # weights sum to one.
        weights = F.softmax(self.layer_logits, dim=0)
        projected = None
        for weight, hidden_state in zip(weights, hidden_states, strict=True):
            term = weight * F.linear(
                hidden_state.detach().to(torch.float32), self.state_out_proj.weight, None
            )
            projected = term if projected is None else projected + term

        state = projected + self.state_out_proj.bias  # (B, T, state_out_dim)
        # AdaptiveAvgPool1d folds the variable T into a fixed num_state_queries
        # so the downstream policy/critic sees a constant state dim.
        state = state.transpose(1, 2)  # (B, state_out_dim, T)
        state = F.adaptive_avg_pool1d(state, self.num_state_queries)
        state = state.transpose(1, 2)  # (B, num_state_queries, state_out_dim)
        return state.flatten(start_dim=1)

    def _forward_prompt(self, texts: list[str], images: list[list[torch.Tensor]]) -> torch.Tensor:
        """Run the VLM over a batch of rendered conversations and read the state
        off its hidden states."""
        inputs = build_vlm_inputs(self.processor, texts, images, self.device)
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

        # Nothing downstream differentiates this pass: the state is detached off
        # the hidden states. Keeping its activations would be a graph the size
        # of the whole prompt that no backward ever reaches.
        with torch.no_grad():
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
        return self._state_from_hidden_states(outputs.hidden_states)

    @torch.inference_mode()
    def _infer(
        self, texts: list[str], images: list[list[torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, HeadOutput]:
        state = self._forward_prompt(texts, images)
        action, _ = self.policy_head.get_action(state)

        critic_out = self.value_head(state, action)
        return state, action, critic_out
