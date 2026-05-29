# SPDX-License-Identifier: MIT
from collections.abc import Callable
from contextlib import nullcontext

import numpy as np
import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn
from torch.nn import functional as F

from .diffusion_utils import compute_actor_loss_with_dacer
from .image_processor import ImageProcessor
from .policy_head import DiffusionPolicy
from .prediction_head import StatePredictionHead
from .reward_processor import RewardProcessor
from .value_head import ActionValueHead, maybe_update_hl_gauss_range
from .video_encoder import VideoEncoder
from .vlm_backbone import is_qwen35, load_model
from .vlm_input_cache import VLMInputCache


class VLMActorCriticWithActionValue(nn.Module):
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
        gamma: float,
        num_bins: int,
        seq_len: int,
        horizon: int,
        critic_loss_weight: float,
        denoising_steps: int,
        denoising_time: float,
        dacer_loss_weight: float,
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
        critic_hidden_dim: int,
        critic_block_num: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        sparsity: float,
        image_mode: str,
        predictor_type: str,
    ) -> None:
        super().__init__()
        if image_mode not in ("mem", "sequence"):
            raise ValueError(f"Unknown image_mode: {image_mode}")
        self.gamma = gamma
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.horizon = horizon
        self.action_dim = action_space_shape[0]
        self.observation_space_shape = observation_space_shape
        self.critic_loss_weight = critic_loss_weight
        self.dacer_loss_weight = dacer_loss_weight
        self.text_q_margin = text_q_margin
        self.text_action_mode = text_action_mode
        self.image_mode = image_mode

        self.predictor_step_num = predictor_step_num
        self.disable_state_predictor = disable_state_predictor
        self.detach_actor = detach_actor
        self.detach_critic = detach_critic
        self.detach_predictor = detach_predictor

        # Image processor (for replay buffer obs_z encoding)
        self.image_processor = ImageProcessor(observation_space_shape)
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
        self.is_qwen35 = is_qwen35(vlm_model_id)
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
        if self.is_qwen35:
            layer_types = vlm_cfg.layer_types
            self.attn_layer_indices = [
                i for i, lt in enumerate(layer_types) if lt == "full_attention"
            ]
        else:
            self.attn_layer_indices = list(range(num_layers))

        self.num_state_queries = num_state_queries
        self.video_encoder = VideoEncoder()

        self.state_out_proj = nn.Linear(vlm_hidden_size, state_out_dim).to(device)
        # AdaptiveAvgPool1d fixes the token count to num_state_queries, so
        # state_dim is determined purely by config.
        state_dim = num_state_queries * state_out_dim

        # Diffusion policy head: action denoising conditioned on state
        self.policy_head = DiffusionPolicy(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=actor_hidden_dim,
            block_num=actor_block_num,
            denoising_time=denoising_time,
            sparsity=sparsity,
            horizon=horizon,
            denoising_steps=denoising_steps,
        )

        # Critic: Q(state, action)
        self.value_head = ActionValueHead(
            in_channels=state_dim,
            action_dim=self.action_dim,
            horizon=horizon,
            hidden_dim=critic_hidden_dim,
            block_num=critic_block_num,
            num_bins=num_bins,
            sparsity=sparsity,
        )

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

        self.value_range = 1.0
        if self.num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-self.value_range,
                max_value=+self.value_range,
                num_bins=num_bins,
                clamp_to_range=True,
            )

        self._dummy_state = torch.zeros(1, 1, 1)

        # prepare_vlm_inputs allocates thousands of small Python objects per
        # call; in long Bench2Drive routes the heap state churned by scenario
        # runtime amplifies that allocation cost, so the call slows by ~8x
        # over a single run. VLMInputCache caches everything that depends
        # only on image dimensions (image_grid_thw, chat template format) but
        # tokenizes the prompt fresh every step so envs that vary their
        # task prompt (TrackingSquare, future CoT, ...) keep working.
        self._input_cache = VLMInputCache(
            processor=self.processor,
            observation_shape=observation_space_shape,
            seq_len=seq_len,
            device=torch.device(device),
            image_mode=self.image_mode,
        )

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
        """Tokenize a task prompt string into token IDs."""
        return self.processor.tokenizer.encode(task_prompt, add_special_tokens=False)

    def decode_task_prompt_ids(self, token_ids: torch.Tensor) -> list[str]:
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
    def infer(
        self,
        s_seq: torch.Tensor,
        obs_z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        r_seq: torch.Tensor,
        rnn_state: torch.Tensor,
        task_prompts: list[str],
    ) -> dict:
        state, action, q_value = self._infer(s_seq, task_prompts)

        next_image, next_reward = self.prediction_head.predict_next_state(
            self._state_for_predictor(state),
            action[:, 0],
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        return {
            "action": action,
            "a_logp": torch.zeros(s_seq.shape[0], 1, device=self.device),
            "value": q_value.item(),
            "x": state,
            "rnn_state": rnn_state,
            "next_image": next_image,
            "next_reward": next_reward,
            "action_token_ids": [],
            "parse_success": True,
        }

    def compute_loss(self, data) -> tuple[torch.Tensor, dict, dict]:
        # Decode task prompts from buffer: use last timestep's prompt for next-state
        next_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -1])
        # Use prompt at the boundary between seq and horizon for current state
        curr_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -self.horizon - 1])

        _, _, next_q = self._infer(data.observations[:, self.horizon :], next_prompts)
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self._compute_target_value(next_q, chunk_rewards, chunk_dones)

        curr_obs = data.observations[:, : -self.horizon]
        state, _ = self._forward_state(curr_obs, curr_prompts)
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)

        # Critic loss
        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        # Actor loss (advantage + DACER)
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(state)

        # Sequence (state prediction) loss
        seq_loss, seq_activations, seq_info = self._compute_sequence_loss(data, state)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        activations_dict = {
            "state": state,
            **critic_activations,
            **actor_activations,
            **seq_activations,
        }
        info_dict = {**critic_info, **actor_info, **seq_info}

        return total_loss, activations_dict, info_dict

    def infer_and_compute_loss(self, data) -> tuple[dict, torch.Tensor, dict, dict]:
        next_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -1])
        curr_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -self.horizon - 1])

        _, next_action, next_q = self._infer(data.observations[:, self.horizon :], next_prompts)
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self._compute_target_value(next_q, chunk_rewards, chunk_dones)

        curr_obs = data.observations[:, : -self.horizon]
        state, _ = self._forward_state(curr_obs, curr_prompts)
        action_chunk = data.actions[:, -self.horizon :]

        # Critic loss
        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        # Actor loss (advantage + DACER)
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(state)

        # Sequence (state prediction) loss
        seq_loss, seq_activations, seq_info = self._compute_sequence_loss(data, state)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        # Actor-only loss (no critic component)
        actor_entropy_loss = actor_loss + seq_loss

        # -Q(s,a) for eligibility trace backward (detached from encoder)
        et_critic_dict = self.value_head(state.detach(), action_chunk.detach())
        if self.num_bins > 1:
            neg_value_detached = -self.hl_gauss_loss(et_critic_dict["output"]).mean()
        else:
            neg_value_detached = -et_critic_dict["output"].mean()

        next_image, next_reward = self.prediction_head.predict_next_state(
            self._state_for_predictor(state),
            next_action[:, 0],
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        infer_dict = {
            "action": next_action,
            "value": next_q.item(),
            "rnn_state": self._dummy_state.clone(),
            "next_image": next_image,
            "next_reward": next_reward,
        }

        activations_dict = {
            "state": state,
            **critic_activations,
            **actor_activations,
            **seq_activations,
        }
        info_dict = {**critic_info, **actor_info, **seq_info}

        et_info = {
            "actor_entropy_loss": actor_entropy_loss,
            "neg_value": neg_value_detached,
            "delta": critic_info["delta"],
        }

        return infer_dict, total_loss, activations_dict, info_dict, et_info

    ####################
    # Internal methods #
    ####################

    def _weighted_hidden(self, hidden_states) -> torch.Tensor:
        """Softmax-weighted sum over all VLM hidden layers (incl. embedding output).

        Hidden values are detached (the VLM is frozen); gradient flows only into
        ``self.layer_logits`` through the softmax weights.
        """
        stacked = torch.stack([h.to(torch.float32).detach() for h in hidden_states], dim=0)
        weights = F.softmax(self.layer_logits, dim=0)
        return (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)

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

    def _vlm_forward(self, images: torch.Tensor, task_prompts: list[str]):
        """Run VLM forward and return (state, past_key_values)."""
        inputs = self._input_cache(images, task_prompts)
        inputs_embeds = self._build_inputs_embeds(inputs)

        # When the VLM weights themselves are frozen we can save a lot of memory
        # by skipping autograd through them.
        with nullcontext() if self.use_lora else torch.no_grad():
            outputs = self._vlm_language_forward(inputs, inputs_embeds)

        # Store last input_id for text generation seeding
        self._last_input_ids = inputs["input_ids"]

        hidden = self._weighted_hidden(outputs.hidden_states)
        state = self.state_out_proj(hidden)  # (B, T, state_out_dim)
        # AdaptiveAvgPool1d folds the variable T into a fixed num_state_queries
        # so the downstream policy/critic sees a constant state dim.
        state = F.adaptive_avg_pool1d(state.transpose(1, 2), self.num_state_queries).transpose(
            1, 2
        )  # (B, num_state_queries, state_out_dim)
        return state.flatten(start_dim=1), outputs.past_key_values

    def _forward_state(
        self, obs: torch.Tensor, task_prompts: list[str]
    ) -> tuple[torch.Tensor, object]:
        """Run VLM forward and compute state. Returns (state, vlm_past_kv)."""
        return self._vlm_forward(obs, task_prompts)

    def _extract_kv(self, vlm_past_kv) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
        """Extract (K, V) pairs for the VLM attention layers (used by text generation).

        Returns:
            vlm_kv_list: list of (K, V) tuples, one per attention layer
            vlm_seq_len: sequence length of the VLM KV cache
        """
        kv_list = []
        if self.is_qwen35:
            for idx in self.attn_layer_indices:
                kv_list.append((vlm_past_kv.key_cache[idx], vlm_past_kv.value_cache[idx]))
            seq_len = vlm_past_kv.key_cache[self.attn_layer_indices[0]].shape[2]
        else:
            for j in range(self.num_layers):
                layer = vlm_past_kv.layers[j]
                kv_list.append((layer.keys, layer.values))
            seq_len = vlm_past_kv.layers[0].keys.shape[2]
        return kv_list, seq_len

    def _generate_text_and_extend_kv(self, prompt: str, vlm_past_kv, max_new_tokens: int):
        """Generate text via manual forward loop (supports batched KV cache).

        Uses greedy decoding (argmax) with manual model.forward() calls
        instead of generate() to avoid rope_deltas batch mismatch issues.
        Returns (first_item_text, extended_kv_cache).
        """
        tokenizer = self.processor.tokenizer
        _, kv_len = self._extract_kv(vlm_past_kv)
        eos_token_id = tokenizer.eos_token_id

        if prompt:
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            if self.is_qwen35:
                B = vlm_past_kv.key_cache[self.attn_layer_indices[0]].shape[0]
            else:
                B = vlm_past_kv.layers[0].keys.shape[0]
            next_ids = prompt_ids.expand(B, -1)  # (B, prompt_len)
            cur_pos = kv_len
        else:
            next_ids = self._last_input_ids[:, -1:].to(self.device)  # (B, 1)
            B = next_ids.shape[0]
            cur_pos = kv_len - 1  # re-feed last cached token

        # rope_deltas from the initial VLM forward (accounts for image token positions)
        rope_deltas = self.vlm_model.model.rope_deltas  # (B, 1)

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
            if step == 0 and not prompt:
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
        q_dict = self.value_head(state, action)
        q = q_dict["output"]
        return self.hl_gauss_loss(q).view(-1) if self.num_bins > 1 else q.view(-1)

    @torch.inference_mode()
    def _infer(
        self, obs: torch.Tensor, task_prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, vlm_past_kv = self._forward_state(obs, task_prompts)
        mode = self.text_action_mode

        if mode == "high_level":
            generated_text, _ = self._generate_text_and_extend_kv(
                "", vlm_past_kv, max_new_tokens=30
            )
            print(f"[HighLevel] {generated_text}")
        elif mode == "text_action":
            generated_text, _ = self._generate_text_and_extend_kv(
                "", vlm_past_kv, max_new_tokens=self.max_new_tokens
            )
            print(f"[TextAction] {generated_text}")
        elif mode == "pi_fast":
            raise NotImplementedError("pi_fast mode is not yet implemented")
        elif mode != "none":
            raise ValueError(f"Unknown text_action_mode: {mode}")

        diff_action, _ = self.policy_head.get_action(state)
        diff_q = self._compute_q(state, diff_action)

        if mode == "text_action":
            action_array, parse_success = self.parse_action_text(generated_text)
            text_action = torch.from_numpy(action_array).unsqueeze(0).to(obs.device)
            text_q = self._compute_q(state, text_action)
            use_text = text_q > diff_q + self.text_q_margin
            action = torch.where(use_text.unsqueeze(-1).unsqueeze(-1), text_action, diff_action)
            q = torch.where(use_text, text_q, diff_q)
            print(
                f"[ActionSelect] diff_q={diff_q.item():.3f}, text_q={text_q.item():.3f}, "
                f"use_text={use_text.item()}, parse_success={parse_success}, "
                f"action_text={generated_text}"
            )
        else:
            action = diff_action
            q = diff_q

        return state, action, q

    @torch.no_grad()
    def _compute_target_value(
        self,
        next_q: torch.Tensor,
        chunk_rewards: torch.Tensor,
        chunk_dones: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = chunk_rewards.size(0)
        discounted_reward = torch.zeros(batch_size, device=self.device)
        gamma_power = 1.0
        continuing = torch.ones(batch_size, device=self.device)
        for i in range(self.horizon):
            discounted_reward += continuing * gamma_power * chunk_rewards[:, i].flatten()
            gamma_power *= self.gamma
            continuing *= 1 - chunk_dones[:, i].flatten()
        return discounted_reward + continuing * gamma_power * next_q

    def _compute_critic_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        target_value: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:
        if self.detach_critic:
            state = state.detach()
        curr_critic_output_dict = self.value_head(state, action_chunk)

        if self.num_bins > 1:
            maybe_update_hl_gauss_range(self, target_value)
            curr_critic_value = self.hl_gauss_loss(curr_critic_output_dict["output"]).view(-1)
            critic_loss = self.hl_gauss_loss(curr_critic_output_dict["output"], target_value)
        else:
            curr_critic_value = curr_critic_output_dict["output"].view(-1)
            critic_loss = F.mse_loss(curr_critic_value, target_value)

        delta = target_value - curr_critic_value

        activations_dict = {}

        info_dict = {
            "delta": delta.mean().item(),
            "critic_loss": critic_loss.item(),
            "curr_critic_value": curr_critic_value.mean().item(),
            "target_value": target_value.mean().item(),
            "value_range": self.value_range,
        }

        return critic_loss, activations_dict, info_dict

    def _compute_actor_loss(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Advantage-based loss + DACER loss, matching actor_critic_with_action_value."""
        if self.detach_actor:
            state = state.detach()
        action, log_pi = self.policy_head.get_action(state)
        bs = action.shape[0]
        actor_activation = {}

        def predict_fn(a_t, t):
            a_flat = a_t.view(bs, -1)
            result = self.policy_head.forward(a_flat, t, state)
            actor_activation["value"] = result["activation"]
            return result["output"].view(bs, self.horizon, self.action_dim)

        total_actor_loss, advantage_dict, info_dict = compute_actor_loss_with_dacer(
            state,
            action,
            self.value_head,
            self.hl_gauss_loss if self.num_bins > 1 else None,
            self.num_bins,
            self.dacer_loss_weight,
            predict_fn,
        )

        activations_dict = {
            "actor": actor_activation["value"],
            "critic": advantage_dict["activation"],
        }
        info_dict["log_pi"] = log_pi.mean().item()

        return total_actor_loss, activations_dict, info_dict

    def _state_for_predictor(self, state: torch.Tensor) -> torch.Tensor:
        """Reshape and project state for StatePredictionHead context."""
        B = state.shape[0]
        x = state.view(B, self.num_state_queries, -1)
        return self.state_to_predictor_proj(x)

    def _compute_sequence_loss(self, data, curr_state):
        if self.disable_state_predictor:
            dummy_loss = torch.tensor(0.0, device=curr_state.device, requires_grad=True)
            activations_dict = {"state_predictor": curr_state}
            info_dict = {"seq_loss": 0.0}
            return dummy_loss, activations_dict, info_dict

        predictor_state = self._state_for_predictor(curr_state)
        if self.detach_predictor:
            predictor_state = predictor_state.detach()

        curr_action = data.actions[:, -1]  # (B, action_dim)

        with torch.no_grad():
            last_obs = data.observations[:, -1]  # (B, C, H, W)
            target_state_next = self.image_processor.encode(last_obs)  # (B, C', H', W')
            B, C, H, W = target_state_next.shape
            target_state_next = target_state_next.flatten(2).permute(0, 2, 1)  # (B, H'*W', C')

        reward_next = data.rewards[:, -1]  # (B, 1)
        target_reward_next = self.reward_processor.encode(reward_next)  # (B, 1, C')
        target_reward_next = target_reward_next.squeeze(1)  # (B, C')
        x1 = torch.cat(
            [target_state_next, target_reward_next.unsqueeze(1)], dim=1
        )  # (B, H'*W'+1, C')

        pred_loss, activation, info_dict = self.prediction_head.compute_loss(
            predictor_state, curr_action, x1
        )

        activations_dict = {"state_predictor": activation}
        return pred_loss, activations_dict, info_dict
