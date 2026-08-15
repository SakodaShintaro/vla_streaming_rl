# SPDX-License-Identifier: MIT
from collections.abc import Callable
from contextlib import nullcontext

import numpy as np
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
from .modules.policy_head import CFGDiffusionPolicy, DiffusionPolicy, MeanFlowPolicy
from .modules.prediction_head import StatePredictionHead
from .modules.reward_processor import RewardProcessor
from .modules.value_head import DistributionalValueHead
from .modules.video_encoder import VideoEncoder
from .modules.vlm_backbone import load_model
from .modules.vlm_input_cache import VLMInputCache


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
        parse_action_text: Callable[[str], tuple[np.ndarray, bool]] | None,
        value_head_factory: Callable[[int, int], DistributionalValueHead],
        seq_len: int,
        horizon: int,
        critic_loss_weight: float,
        denoising_steps: int,
        denoising_time: float,
        dacer_loss_weight: float,
        som_alpha: float,
        som_w: float,
        text_q_margin: float,
        text_action_mode: str,
        predictor_step_num: int,
        disable_state_predictor: bool,
        detach_actor: bool,
        detach_critic: bool,
        detach_predictor: bool,
        use_lora: bool,
        vlm_model_id: str,
        max_new_tokens: int,
        max_prompt_tokens: int,
        pad_token_id: int,
        num_state_queries: int,
        state_out_dim: int,
        actor_hidden_dim: int,
        actor_block_num: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        sparsity: float,
        image_mode: str,
        predictor_type: str,
        policy_type: str,
        image_encoder_type: str,
        image_encoder_output_dim: int,
    ) -> None:
        super().__init__()
        assert image_mode in ("mem", "sequence"), f"Unknown image_mode: {image_mode}"
        self.seq_len = seq_len
        self.horizon = horizon
        self.action_dim = action_space_shape[0]
        self.observation_space_shape = observation_space_shape
        self.critic_loss_weight = critic_loss_weight
        self.text_q_margin = text_q_margin
        self.text_action_mode = text_action_mode
        self.image_mode = image_mode

        self.predictor_step_num = predictor_step_num
        self.disable_state_predictor = disable_state_predictor
        self.detach_actor = detach_actor
        self.detach_critic = detach_critic
        self.detach_predictor = detach_predictor

        self.image_processor = ImageProcessor(
            observation_space_shape, image_encoder_type, image_encoder_output_dim
        )
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        # Load VLM
        device = "cuda"
        self.use_lora = bool(use_lora)
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
        self.parse_action_text = parse_action_text
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.pad_token_id = pad_token_id

        # Index attention layers for ``_extract_kv`` (text generation).
        layer_types = vlm_cfg.layer_types
        self.attn_layer_indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]

        self.num_state_queries = num_state_queries
        self.video_encoder = VideoEncoder()

        self.state_out_proj = nn.Linear(vlm_hidden_size, state_out_dim).to(device)
        # AdaptiveAvgPool1d fixes the token count to num_state_queries, so
        # state_dim is determined purely by config.
        state_dim = num_state_queries * state_out_dim

        self.policy_type = policy_type
        if self.policy_type == "diffusion":
            self.policy_head = DiffusionPolicy(
                state_dim=state_dim,
                action_dim=self.action_dim,
                hidden_dim=actor_hidden_dim,
                block_num=actor_block_num,
                denoising_time=denoising_time,
                sparsity=sparsity,
                horizon=horizon,
                denoising_steps=denoising_steps,
                dacer_loss_weight=dacer_loss_weight,
            )
        elif self.policy_type == "cfgrl":
            self.policy_head = CFGDiffusionPolicy(
                state_dim=state_dim,
                action_dim=self.action_dim,
                hidden_dim=actor_hidden_dim,
                block_num=actor_block_num,
                denoising_time=denoising_time,
                sparsity=sparsity,
                cfgrl_beta=1.5,
                horizon=horizon,
                denoising_steps=denoising_steps,
                condition_drop_prob=0.1,
            )
        elif self.policy_type == "som":
            self.policy_head = MeanFlowPolicy(
                state_dim=state_dim,
                action_dim=self.action_dim,
                hidden_dim=actor_hidden_dim,
                block_num=actor_block_num,
                horizon=horizon,
                sparsity=sparsity,
                som_alpha=som_alpha,
                som_w=som_w,
            )
        else:
            raise ValueError(f"Unknown policy_type: {self.policy_type}")

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

        # VLMInputCache caches everything that depends
        # only on image dimensions (image_grid_thw, chat template format) but
        # tokenizes the prompt fresh every step so envs that vary their
        # task prompt (LIBERO, future CoT, ...) keep working.
        self._input_cache = VLMInputCache(
            processor=self.processor,
            observation_shape=observation_space_shape,
            seq_len=seq_len,
            device=torch.device(device),
            image_mode=self.image_mode,
        )

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def observe_scalar_obs(
        self,
        velocity_x: float,
        velocity_y: float,
        velocity_z: float,
        episode_return: float,
        pass_mark: float,
        global_step: float,
        episode_step: float,
        health: float,
    ) -> None:
        del velocity_x, velocity_y, velocity_z, episode_return, pass_mark
        del global_step, episode_step, health

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
        state, _ = self._forward_state(curr_obs, curr_prompts)
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

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        info_dict = {
            f"losses/{key}": value
            for key, value in {**critic_info, **actor_info, **seq_info}.items()
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
        state, _ = self._forward_state(curr_obs, curr_prompts)
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

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        # Actor-only loss (no critic component)
        actor_entropy_loss = actor_loss + seq_loss

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
            for key, value in {**critic_info, **actor_info, **seq_info}.items()
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

    def _get_visual(self) -> nn.Module:
        """Get the visual encoder from the VLM model (handles PEFT wrapping)."""
        if self.use_lora:
            return self.vlm_model.model.model.visual
        return self.vlm_model.model.visual

    def _get_vlm_model_inner(self) -> nn.Module:
        """Get the inner Qwen3_5Model (handles PEFT wrapping)."""
        if self.use_lora:
            return self.vlm_model.model.model
        return self.vlm_model.model

    def _build_inputs_embeds(self, inputs: dict) -> torch.Tensor:
        """Build inputs_embeds and scatter image embeddings into <image_pad> positions.

        Two image paths share the same scatter step but differ in what gets scattered:

        - mem mode: only the last frame appears in the LLM context. ``VideoEncoder``
          processes all frames jointly (with causal temporal attention) and returns
          merged tokens for the last frame only.
        - sequence mode: every frame appears in the LLM context in temporal order.
          The VLM's stock vision encoder processes each frame independently and
          returns merged tokens for all frames, concatenated in the same order as
          the <image_pad> placeholders.
        """
        vlm_inner = self._get_vlm_model_inner()
        inputs_embeds = vlm_inner.get_input_embeddings()(inputs["input_ids"])

        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["seq_len"]

        if self.image_mode == "sequence":
            visual = self._get_visual()
            pixel_values = inputs["all_pixel_values"].type(visual.dtype)
            vision_output = visual(pixel_values, grid_thw=inputs["all_image_grid_thw"])
            image_embeds = vision_output.pooler_output
        else:
            image_embeds = self.video_encoder(
                self._get_visual(),
                inputs["all_pixel_values"],
                inputs["all_image_grid_thw"],
                batch_size,
                seq_len,
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        image_token_id = vlm_inner.config.image_token_id
        image_mask = (inputs["input_ids"] == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return inputs_embeds

    def _vlm_language_forward(self, inputs: dict, inputs_embeds: torch.Tensor):
        """Run the VLM language model with pre-built inputs_embeds (no pixel_values)."""
        vlm_inner = self._get_vlm_model_inner()

        # Compute 3D position_ids (needed for image token positions)
        position_ids = vlm_inner.compute_3d_position_ids(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            past_key_values=None,
        )

        forward_kwargs = dict(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

        # language_model forward via the outer model (handles lm_head, cache wrapping)
        return self.vlm_model.forward(**forward_kwargs)

    def _forward_state(
        self, obs: torch.Tensor, task_prompts: list[str]
    ) -> tuple[torch.Tensor, object]:
        """Run VLM forward and return (state, past_key_values)."""
        inputs = self._input_cache(obs, task_prompts)
        inputs_embeds = self._build_inputs_embeds(inputs)

        # When the VLM weights themselves are frozen we can save a lot of memory
        # by skipping autograd through them.
        with nullcontext() if self.use_lora else torch.no_grad():
            outputs = self._vlm_language_forward(inputs, inputs_embeds)

        # Store last input_id for text generation seeding
        self._last_input_ids = inputs["input_ids"]

        # Compute state representation: softmax-weighted sum across embedding + each layer's hidden states
        stacked = torch.stack([h.to(torch.float32).detach() for h in outputs.hidden_states], dim=0)
        weights = F.softmax(self.layer_logits, dim=0)
        hidden = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)

        state = self.state_out_proj(hidden)  # (B, T, state_out_dim)
        # AdaptiveAvgPool1d folds the variable T into a fixed num_state_queries
        # so the downstream policy/critic sees a constant state dim.
        state = state.transpose(1, 2)  # (B, state_out_dim, T)
        state = F.adaptive_avg_pool1d(state, self.num_state_queries)
        state = state.transpose(1, 2)  # (B, num_state_queries, state_out_dim)
        return state.flatten(start_dim=1), outputs.past_key_values

    def _generate_text_and_extend_kv(self, vlm_past_kv, max_new_tokens: int):
        """Generate text via manual forward loop (supports batched KV cache).

        Uses greedy decoding (argmax) with manual model.forward() calls
        instead of generate() to avoid rope_deltas batch mismatch issues.
        Returns (first_item_text, extended_kv_cache).
        """
        tokenizer = self.processor.tokenizer
        kv_len = vlm_past_kv.key_cache[self.attn_layer_indices[0]].shape[2]
        eos_token_id = tokenizer.eos_token_id

        next_ids = self._last_input_ids[:, -1:].to(self.device)  # (B, 1)
        B = next_ids.shape[0]
        cur_pos = kv_len - 1  # re-feed last cached token

        # rope_deltas from the initial VLM forward (accounts for image token positions)
        rope_deltas = self._get_vlm_model_inner().rope_deltas  # (B, 1)

        self.vlm_model.eval()

        generated_tokens = [[] for _ in range(B)]
        finished = [False] * B

        for step in range(max_new_tokens + 1):  # +1 for initial seed step
            seq_len = next_ids.shape[1]
            cache_position = torch.arange(cur_pos, cur_pos + seq_len, device=self.device)

            # Build 3D position_ids: (3, B, seq_len) for mrope
            text_pos = cache_position.view(1, 1, -1).expand(1, B, -1)  # (1, B, seq_len)
            if rope_deltas is not None:
                text_pos = text_pos + rope_deltas.unsqueeze(0)  # broadcast (1, B, 1)
            position_ids = text_pos.expand(3, -1, -1)  # (3, B, seq_len)

            outputs = self.vlm_model(
                input_ids=next_ids,
                attention_mask=torch.ones(B, cur_pos + seq_len, device=self.device),
                past_key_values=vlm_past_kv,
                cache_position=cache_position,
                position_ids=position_ids,
            )

            vlm_past_kv = outputs.past_key_values
            cur_pos = cur_pos + seq_len

            # Skip collecting tokens on seed step (step 0 when no prompt)
            if step == 0:
                next_ids = outputs.logits[:, -1:, :].argmax(dim=-1)
                continue

            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)  # (B, 1)

            for b in range(B):
                if not finished[b]:
                    tid = next_token[b, 0].item()
                    if tid == eos_token_id:
                        finished[b] = True
                    else:
                        generated_tokens[b].append(tid)

            if all(finished):
                break

            next_ids = next_token

        self.vlm_model.train()

        first_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        return first_text, vlm_past_kv

    def _compute_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute scalar Q-value for a (state, action) pair."""
        q_out = self.value_head(state, action)
        return self.value_head.to_value(q_out.output).view(-1)

    @torch.inference_mode()
    def _infer(
        self, obs: torch.Tensor, task_prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, HeadOutput]:
        state, vlm_past_kv = self._forward_state(obs, task_prompts)
        mode = self.text_action_mode

        if mode == "high_level":
            generated_text, _ = self._generate_text_and_extend_kv(vlm_past_kv, max_new_tokens=30)
            print(f"[HighLevel] {generated_text}")
        elif mode == "text_action":
            generated_text, _ = self._generate_text_and_extend_kv(
                vlm_past_kv, max_new_tokens=self.max_new_tokens
            )
            print(f"[TextAction] {generated_text}")
        elif mode == "pi_fast":
            raise NotImplementedError("pi_fast mode is not yet implemented")
        elif mode != "none":
            raise ValueError(f"Unknown text_action_mode: {mode}")

        diff_action, actor_activation = self.policy_head.get_action(state)
        diff_q = self._compute_q(state, diff_action)

        if mode == "text_action":
            action_array, parse_success = self.parse_action_text(generated_text)
            text_action = torch.from_numpy(action_array).unsqueeze(0).to(obs.device)
            text_q = self._compute_q(state, text_action)
            use_text = text_q > diff_q + self.text_q_margin
            action = torch.where(use_text.unsqueeze(-1).unsqueeze(-1), text_action, diff_action)
            print(
                f"[ActionSelect] diff_q={diff_q.item():.3f}, text_q={text_q.item():.3f}, "
                f"use_text={use_text.item()}, parse_success={parse_success}, "
                f"action_text={generated_text}"
            )
        else:
            action = diff_action

        critic_out = self.value_head(state, action)
        return state, action, actor_activation, critic_out

    def _state_for_predictor(self, state: torch.Tensor) -> torch.Tensor:
        """Reshape and project state for StatePredictionHead context."""
        B = state.shape[0]
        x = state.view(B, self.num_state_queries, -1)
        return self.state_to_predictor_proj(x)
