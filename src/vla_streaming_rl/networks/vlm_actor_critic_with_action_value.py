# SPDX-License-Identifier: MIT
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..replay_buffer import ReplayBufferData
from .interface import (
    ActivationFeatures,
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from .modules.head_output import HeadOutput
from .modules.image_processor import ImageProcessor
from .modules.policy_head import build_policy_head
from .modules.prediction_head import StatePredictionHead
from .modules.reward_processor import RewardProcessor
from .modules.value_head import DistributionalValueHead
from .modules.vlm_backbone import load_model
from .modules.vlm_inputs import build_vlm_inputs


@dataclass
class PromptForward:
    """What one VLM pass over the observation window leaves behind.

    ``state`` is what the policy and critic read; the other three are what a
    reasoning chain needs on top of it -- the cache to sample from and the prompt
    tokens/embeddings to teacher-force the chain against.
    """

    state: torch.Tensor
    past_key_values: object
    inputs: dict
    inputs_embeds: torch.Tensor


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
        seq_len: int,
        horizon: int,
        critic_loss_weight: float,
        denoising_steps: int,
        denoising_time: float,
        dacer_loss_weight: float,
        som_alpha: float,
        som_w: float,
        use_reasoning: bool,
        reasoning_loss_weight: float,
        reasoning_max_tokens: int,
        reasoning_temperature: float,
        predictor_step_num: int,
        disable_state_predictor: bool,
        detach_actor: bool,
        detach_critic: bool,
        detach_predictor: bool,
        use_lora: bool,
        vlm_model_id: str,
        max_prompt_tokens: int,
        pad_token_id: int,
        num_state_queries: int,
        state_out_dim: int,
        actor_hidden_dim: int,
        actor_block_num: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        sparsity: float,
        decision_fps: float,
        predictor_type: str,
        policy_type: str,
        image_encoder_type: str,
        image_encoder_output_dim: int,
        image_encode_mode: str,
        image_encoder_trainable: bool,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.action_dim = action_space_shape[0]
        self.observation_space_shape = observation_space_shape
        self.critic_loss_weight = critic_loss_weight
        self.decision_fps = decision_fps
        self.use_reasoning = bool(use_reasoning)
        self.reasoning_loss_weight = reasoning_loss_weight
        self.reasoning_max_tokens = reasoning_max_tokens
        self.reasoning_temperature = reasoning_temperature

        self.predictor_step_num = predictor_step_num
        self.disable_state_predictor = disable_state_predictor
        self.detach_actor = detach_actor
        self.detach_critic = detach_critic
        self.detach_predictor = detach_predictor

        # this network's spatial-temporal attention is built around the patch
        # grid; a single pooled token would leave it nothing to attend over, so
        # "single_token" is for the animal backbone (see ``networks/animal_ppo.py``)
        assert image_encode_mode == "grid"
        self.image_processor = ImageProcessor(
            observation_space_shape,
            image_encoder_type,
            image_encoder_output_dim,
            image_encode_mode,
            image_encoder_trainable,
        )
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        # Load VLM
        device = "cuda"
        self.use_lora = bool(use_lora)
        assert not (self.use_reasoning and not self.use_lora), (
            "use_reasoning trains the VLM through the reasoning tokens, so use_lora must be on"
        )
        self.vlm_model, self.processor = load_model(
            vlm_model_id,
            use_lora=self.use_lora,
            device=device,
        )
        self.device = device

        # VLM config
        vlm_cfg = self.vlm_model.config.text_config
        vlm_hidden_size = vlm_cfg.hidden_size
        num_layers = vlm_cfg.num_hidden_layers
        self.num_layers = num_layers
        self.vlm_num_kv_heads = vlm_cfg.num_key_value_heads
        self.vlm_head_dim = vlm_cfg.head_dim
        # Input-independent learnable logits over all (embedding + per-layer) hidden
        # states; softmax-weighted sum forms the representation used downstream.
        self.layer_logits = nn.Parameter(torch.zeros(num_layers + 1, device=device))
        self.max_prompt_tokens = max_prompt_tokens
        self.pad_token_id = pad_token_id

        self.num_state_queries = num_state_queries

        self.state_out_proj = nn.Linear(vlm_hidden_size, state_out_dim).to(device)
        # AdaptiveAvgPool1d fixes the token count to num_state_queries, so
        # state_dim is determined purely by config.
        state_dim = num_state_queries * state_out_dim

        self.policy_type = policy_type
        self.policy_head = build_policy_head(
            policy_type=policy_type,
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=actor_hidden_dim,
            block_num=actor_block_num,
            horizon=horizon,
            sparsity=sparsity,
            denoising_time=denoising_time,
            denoising_steps=denoising_steps,
            dacer_loss_weight=dacer_loss_weight,
            som_alpha=som_alpha,
            som_w=som_w,
        )

        # Critic: Q(state, action)
        self.value_head = value_head_factory(state_dim, self.action_dim)

        self.prediction_head = StatePredictionHead(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            action_dim=self.action_dim,
            predictor_hidden_dim=predictor_hidden_dim,
            predictor_block_num=predictor_block_num,
            predictor_type=predictor_type,
        )
        # Project state output to match FluxDiT context_in_dim
        self.state_to_predictor_proj = nn.Linear(state_out_dim, hidden_image_dim)

        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

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

    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
        """Tokenize a task prompt string into token IDs."""
        return self.processor.tokenizer.encode(task_prompt, add_special_tokens=False)

    def _decode_task_prompt_ids(self, token_ids: torch.Tensor) -> list[str]:
        """Decode task prompt token IDs back to strings.

        Args:
            token_ids: (B, max_prompt_tokens) tensor of token IDs
        Returns:
            List of decoded strings, one per batch element
        """
        results = []
        for i in range(token_ids.shape[0]):
            ids = token_ids[i]
            # Remove padding tokens
            mask = ids != self.pad_token_id
            valid_ids = ids[mask].tolist()
            text = self.processor.tokenizer.decode(valid_ids, skip_special_tokens=True)
            results.append(text)
        return results

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        state, action, actor_activation, critic_out = self._infer(data.s_seq, data.task_prompts)

        next_image_latent, next_reward_latent, predictor_activation = (
            self.prediction_head.predict_next_state(
                self._state_for_predictor(state),
                action[:, 0],
                self.predictor_step_num,
                self.disable_state_predictor,
            )
        )

        activations = ActivationFeatures(
            state=state,
            actor=actor_activation,
            critic=critic_out.activation,
            state_predictor=predictor_activation,
        )

        return InferResult(
            action=action,
            value_report=self.value_head.value_report(critic_out.output),
            rnn_state=data.rnn_state,
            next_image_latent=next_image_latent,
            next_reward_latent=next_reward_latent,
            activations=activations,
            features=state,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        # Decode task prompts from buffer: use last timestep's prompt for next-state
        next_prompts = self._decode_task_prompt_ids(data.task_prompt_token_ids[:, -1])
        # Use prompt at the boundary between seq and horizon for current state
        curr_prompts = self._decode_task_prompt_ids(
            data.task_prompt_token_ids[:, -self.horizon - 1]
        )

        _, _, _, next_critic_out = self._infer(data.observations[:, self.horizon :], next_prompts)
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self.value_head.compute_target_value(
            next_critic_out.output, chunk_rewards, chunk_dones
        )

        curr_obs = data.observations[:, : -self.horizon]
        prompt = self._forward_prompt(curr_obs, curr_prompts)
        state = prompt.state
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)

        # Critic loss
        critic_loss, critic_info = self.value_head.compute_critic_loss(
            state, action_chunk, target_value, self.detach_critic
        )

        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state,
            action_chunk,
            value_head=self.value_head,
            detach_actor=self.detach_actor,
        )

        # Sequence (state prediction) loss
        seq_loss, seq_info = self.prediction_head.compute_loss(
            self._state_for_predictor(state),
            data.actions[:, -1],
            data.observations[:, -1],
            data.rewards[:, -1],
            self.detach_predictor,
            self.disable_state_predictor,
        )

        reasoning_loss, reasoning_info = self._reasoning_loss_or_zero(prompt)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss + reasoning_loss

        info_dict = {
            f"losses/{key}": value
            for key, value in {
                **critic_info,
                **actor_info,
                **seq_info,
                **reasoning_info,
            }.items()
        }

        return LossResult(loss=total_loss, info=info_dict)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        next_prompts = self._decode_task_prompt_ids(data.task_prompt_token_ids[:, -1])
        curr_prompts = self._decode_task_prompt_ids(
            data.task_prompt_token_ids[:, -self.horizon - 1]
        )

        next_state, next_action, actor_activation, critic_out = self._infer(
            data.observations[:, self.horizon :], next_prompts
        )
        critic_activation = critic_out.activation
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self.value_head.compute_target_value(
            critic_out.output, chunk_rewards, chunk_dones
        )

        curr_obs = data.observations[:, : -self.horizon]
        prompt = self._forward_prompt(curr_obs, curr_prompts)
        state = prompt.state
        action_chunk = data.actions[:, -self.horizon :]

        # Critic loss
        critic_loss, critic_info = self.value_head.compute_critic_loss(
            state, action_chunk, target_value, self.detach_critic
        )

        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state,
            action_chunk,
            value_head=self.value_head,
            detach_actor=self.detach_actor,
        )

        # Sequence (state prediction) loss
        seq_loss, seq_info = self.prediction_head.compute_loss(
            self._state_for_predictor(state),
            data.actions[:, -1],
            data.observations[:, -1],
            data.rewards[:, -1],
            self.detach_predictor,
            self.disable_state_predictor,
        )

        reasoning_loss, reasoning_info = self._reasoning_loss_or_zero(prompt)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss + reasoning_loss

        # Actor-only loss (no critic component)
        actor_entropy_loss = actor_loss + seq_loss + reasoning_loss

        # -Q(s,a) for eligibility trace backward (detached from encoder)
        et_critic_out = self.value_head(state.detach(), action_chunk.detach())
        neg_value_detached = -self.value_head.to_value(et_critic_out.output).mean()

        next_image_latent, next_reward_latent, predictor_activation = (
            self.prediction_head.predict_next_state(
                self._state_for_predictor(state),
                next_action[:, 0],
                self.predictor_step_num,
                self.disable_state_predictor,
            )
        )

        activations = ActivationFeatures(
            state=next_state,
            actor=actor_activation,
            critic=critic_activation,
            state_predictor=predictor_activation,
        )

        infer_result = InferResult(
            action=next_action,
            value_report=self.value_head.value_report(critic_out.output),
            rnn_state=self._dummy_state.clone(),
            next_image_latent=next_image_latent,
            next_reward_latent=next_reward_latent,
            activations=activations,
            features=next_state,
        )
        info_dict = {
            f"losses/{key}": value
            for key, value in {
                **critic_info,
                **actor_info,
                **seq_info,
                **reasoning_info,
            }.items()
        }

        et_info = EligibilityTraceInfo(
            actor_entropy_loss=actor_entropy_loss,
            neg_value=neg_value_detached,
            delta=critic_info["delta"],
        )

        return InferLossResult(
            infer_result=infer_result,
            loss_result=LossResult(loss=total_loss, info=info_dict),
            et_info=et_info,
        )

    ####################
    # Internal methods #
    ####################

    def _get_vlm_model_inner(self) -> nn.Module:
        """Get the inner Qwen3_5Model (handles PEFT wrapping)."""
        if self.use_lora:
            return self.vlm_model.model.model
        return self.vlm_model.model

    def _state_from_hidden_states(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Softmax-weighted sum across embedding + per-layer hidden states -> state."""
        stacked = torch.stack([h.to(torch.float32).detach() for h in hidden_states], dim=0)
        weights = F.softmax(self.layer_logits, dim=0)
        hidden = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)

        state = self.state_out_proj(hidden)  # (B, T, state_out_dim)
        # AdaptiveAvgPool1d folds the variable T into a fixed num_state_queries
        # so the downstream policy/critic sees a constant state dim.
        state = state.transpose(1, 2)  # (B, state_out_dim, T)
        state = F.adaptive_avg_pool1d(state, self.num_state_queries)
        state = state.transpose(1, 2)  # (B, num_state_queries, state_out_dim)
        return state.flatten(start_dim=1)

    def _forward_prompt(self, obs: torch.Tensor, task_prompts: list[str]) -> PromptForward:
        """Run the VLM over the observation window."""
        inputs = build_vlm_inputs(
            processor=self.processor,
            images=obs,
            task_prompts=task_prompts,
            decision_fps=self.decision_fps,
        )
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

        # When the VLM weights themselves are frozen we can save a lot of memory
        # by skipping autograd through them.
        with nullcontext() if self.use_lora else torch.no_grad():
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
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )

        # Store last input_id for reasoning generation seeding
        self._last_input_ids = inputs["input_ids"]

        return PromptForward(
            state=self._state_from_hidden_states(outputs.hidden_states),
            past_key_values=outputs.past_key_values,
            inputs=inputs,
            inputs_embeds=inputs_embeds,
        )

    def _reason(self, prompt: PromptForward) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a reasoning chain on top of the prompt and score it.

        Sampling extends the prompt's KV cache in place -- Qwen3.5's hybrid
        linear-attention cache is not copyable -- and the chain is then
        teacher-forced as one full sequence, because incremental decoding cannot
        be differentiated through: the fused recurrent kernel of the
        linear-attention layers has no backward. That second pass also yields the
        reasoning-conditioned hidden states.

        Returns (mean token log-prob, reasoning-conditioned state, valid_mask);
        ``valid_mask`` is False for the padding that follows the EOS token of an
        already finished row.
        """
        vlm_inner = self._get_vlm_model_inner()
        eos_token_id = self.processor.tokenizer.eos_token_id
        inputs = prompt.inputs
        inputs_embeds = prompt.inputs_embeds

        kv = prompt.past_key_values
        next_ids = self._last_input_ids[:, -1:].to(self.device)
        batch_size = next_ids.shape[0]
        cur_pos = kv.get_seq_length() - 1

        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        alive = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        was_training = self.vlm_model.training
        self.vlm_model.eval()
        with torch.no_grad():
            for _ in range(self.reasoning_max_tokens):
                seq_len = next_ids.shape[1]
                cache_position = torch.arange(cur_pos, cur_pos + seq_len, device=self.device)
                text_pos = cache_position.view(1, 1, -1).expand(1, batch_size, -1)
                rope_deltas = vlm_inner.rope_deltas
                if rope_deltas is not None:
                    text_pos = text_pos + rope_deltas.unsqueeze(0)

                decode_out = self.vlm_model(
                    input_ids=next_ids,
                    attention_mask=torch.ones(batch_size, cur_pos + seq_len, device=self.device),
                    past_key_values=kv,
                    cache_position=cache_position,
                    position_ids=text_pos.expand(3, -1, -1),
                )
                kv = decode_out.past_key_values
                cur_pos = cur_pos + seq_len

                logits = decode_out.logits[:, -1, :].to(torch.float32) / self.reasoning_temperature
                sampled = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)  # (B, 1)

                tokens.append(sampled)
                masks.append(alive.clone())
                alive = alive & (sampled[:, 0] != eos_token_id)
                if not bool(alive.any()):
                    break
                next_ids = sampled
        if was_training:
            self.vlm_model.train()

        token_ids = torch.cat(tokens, dim=1)  # (B, L)
        valid_mask = torch.stack(masks, dim=1)  # (B, L)
        length = token_ids.shape[1]

        reasoning_embeds = vlm_inner.get_input_embeddings()(token_ids).to(inputs_embeds.dtype)
        full_embeds = torch.cat([inputs_embeds, reasoning_embeds], dim=1)
        full_ids = torch.cat([inputs["input_ids"], token_ids], dim=1)
        full_mask = torch.cat([inputs["attention_mask"], torch.ones_like(token_ids)], dim=1)
        full_mm_type = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(token_ids)], dim=1)

        position_ids = vlm_inner.compute_3d_position_ids(
            input_ids=full_ids,
            image_grid_thw=inputs["image_grid_thw"],
            video_grid_thw=inputs["video_grid_thw"],
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            past_key_values=None,
            mm_token_type_ids=full_mm_type,
        )
        scored = self.vlm_model.forward(
            input_ids=None,
            inputs_embeds=full_embeds,
            position_ids=position_ids,
            attention_mask=full_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            logits_to_keep=length + 1,
        )

        # logits_to_keep keeps the last length+1 positions; dropping the final one
        # leaves exactly the positions that predict token_ids.
        logits = scored.logits[:, :-1].to(torch.float32) / self.reasoning_temperature
        token_log_probs = F.log_softmax(logits, dim=-1).gather(2, token_ids.unsqueeze(-1))
        token_log_probs = token_log_probs.squeeze(-1)  # (B, L)

        mask = valid_mask.to(token_log_probs.dtype)
        token_num = mask.sum(dim=1).clamp(min=1.0)
        sequence_log_prob = (token_log_probs * mask).sum(dim=1) / token_num

        state = self._state_from_hidden_states(scored.hidden_states)
        return sequence_log_prob, state, valid_mask

    def _reasoning_loss_or_zero(self, prompt: PromptForward) -> tuple[torch.Tensor, dict]:
        """REINFORCE on the reasoning chain with Q(with reasoning) - Q(without) as return."""
        if not self.use_reasoning:
            return torch.zeros((), device=prompt.state.device), {"reasoning_loss": 0.0}

        sequence_log_prob, state_with_reasoning, valid_mask = self._reason(prompt)
        state_without_reasoning = prompt.state

        with torch.no_grad():
            action_with, _ = self.policy_head.get_action(state_with_reasoning)
            action_without, _ = self.policy_head.get_action(state_without_reasoning)
            q_with = self._compute_q(state_with_reasoning, action_with)
            q_without = self._compute_q(state_without_reasoning, action_without)
        advantage = q_with - q_without

        reasoning_loss = -(advantage * sequence_log_prob).mean() * self.reasoning_loss_weight

        info_dict = {
            "reasoning_loss": reasoning_loss.item(),
            "reasoning_advantage": advantage.mean().item(),
            "reasoning_q_with": q_with.mean().item(),
            "reasoning_q_without": q_without.mean().item(),
            "reasoning_log_prob": sequence_log_prob.mean().item(),
            "reasoning_token_num": valid_mask.sum(dim=1).to(torch.float32).mean().item(),
        }
        return reasoning_loss, info_dict

    def _compute_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute scalar Q-value for a (state, action) pair."""
        q_out = self.value_head(state, action)
        return self.value_head.to_value(q_out.output).view(-1)

    @torch.inference_mode()
    def _infer(
        self, obs: torch.Tensor, task_prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, HeadOutput]:
        prompt = self._forward_prompt(obs, task_prompts)
        state = self._reason(prompt)[1] if self.use_reasoning else prompt.state

        action, actor_activation = self.policy_head.get_action(state)

        critic_out = self.value_head(state, action)
        return state, action, actor_activation, critic_out

    def _state_for_predictor(self, state: torch.Tensor) -> torch.Tensor:
        """Reshape and project state for StatePredictionHead context."""
        B = state.shape[0]
        x = state.view(B, self.num_state_queries, -1)
        return self.state_to_predictor_proj(x)
