# SPDX-License-Identifier: MIT
import torch
from torch import nn
from torch.nn import functional as F

from vla_streaming_rl.networks.backbone import SpatialTemporalEncoder, TemporalOnlyEncoder
from vla_streaming_rl.networks.image_processor import ImageProcessor
from vla_streaming_rl.networks.policy_head import BetaPolicy, CategoricalPolicy
from vla_streaming_rl.networks.prediction_head import StatePredictionHead
from vla_streaming_rl.networks.reward_processor import RewardProcessor
from vla_streaming_rl.networks.value_head import StateValueHead


class ActorCriticWithStateValue(nn.Module):
    def __init__(
        self,
        *,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        gamma: float,
        clip_param_policy: float,
        clip_param_value: float,
        num_bins: int,
        predictor_step_num: int,
        critic_loss_weight: float,
        encoder: str,
        seq_len: int,
        encoder_block_num: int,
        temporal_model_type: str,
        horizon: int,
        critic_block_num: int,
        policy_type: str,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        disable_state_predictor: bool,
        predictor_type: str,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.clip_param_policy = clip_param_policy
        self.clip_param_value = clip_param_value
        self.action_dim = action_space_shape[0]
        self.num_bins = num_bins
        self.observation_space_shape = observation_space_shape
        self.predictor_step_num = predictor_step_num
        self.critic_loss_weight = critic_loss_weight

        self.image_processor = ImageProcessor(observation_space_shape)
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        if encoder == "spatial_temporal":
            self.encoder = SpatialTemporalEncoder(
                image_processor=self.image_processor,
                reward_processor=self.reward_processor,
                seq_len=seq_len,
                n_layer=encoder_block_num,
                action_dim=self.action_dim,
                temporal_model_type=temporal_model_type,
                use_image_only=True,
            )
        elif encoder == "temporal_only":
            self.encoder = TemporalOnlyEncoder(
                image_processor=self.image_processor,
                reward_processor=self.reward_processor,
                seq_len=seq_len,
                n_layer=encoder_block_num,
                action_dim=self.action_dim,
                temporal_model_type=temporal_model_type,
                use_image_only=True,
            )
        else:
            raise ValueError(f"Unknown encoder: {encoder=}")

        hidden_dim = self.encoder.output_dim
        self.horizon = horizon

        self.value_head = StateValueHead(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            block_num=1,
            num_bins=num_bins,
            sparsity=0.0,
        )

        if policy_type == "beta":
            self.policy_head = BetaPolicy(hidden_dim, self.action_dim, horizon)
        elif policy_type == "categorical":
            self.policy_head = CategoricalPolicy(hidden_dim, self.action_dim, horizon)
        else:
            raise ValueError("Invalid policy type")

        self.prediction_head = StatePredictionHead(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            action_dim=self.action_dim,
            predictor_hidden_dim=predictor_hidden_dim,
            predictor_block_num=predictor_block_num,
            predictor_type=predictor_type,
        )

        self.disable_state_predictor = disable_state_predictor

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with orthogonal initialization.

        Arguments:
            module {nn.Module} -- Module to initialize
        """
        for name, param in module.named_parameters():
            if "ae." in name:
                continue
            if param.dim() != 2:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_state()

    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
        return []

    def decode_task_prompt_ids(self, token_ids: torch.Tensor) -> list[str]:
        return [""] * token_ids.shape[0]

    @torch.inference_mode()
    def infer(
        self,
        s_seq: torch.Tensor,  # (B, T, C, H, W)
        obs_z_seq: torch.Tensor,  # (B, T, C', H', W') - pre-encoded observations
        a_seq: torch.Tensor,  # (B, T, action_dim)
        r_seq: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # SpatialTemporal: (B, space_len, state_size, n_layer); TemporalOnly: (B, state_size, n_layer)
        task_prompts: list[str] | None = None,
    ) -> dict:
        x, rnn_state = self.encoder(s_seq, obs_z_seq, a_seq, r_seq, rnn_state)  # (B, hidden_dim)

        value_dict = self.value_head(x)

        policy_dict = self.policy_head(x, None)

        next_image, next_reward = self.prediction_head.predict_next_state(
            x,
            policy_dict["action"][:, 0],  # use first action in chunk for prediction
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        return {
            "action": policy_dict["action"],  # (B, horizon, action_dim)
            "a_logp": policy_dict["a_logp"],  # (B, 1)
            "entropy": policy_dict["entropy"],  # (B, 1)
            "value": value_dict["output"],  # (B, num_bins) or (B, 1)
            "x": x,  # (B, hidden_dim)
            "rnn_state": rnn_state,  # (B, ...)
            "next_image": next_image,  # predicted next image
            "next_reward": next_reward,  # predicted next reward
            "action_token_ids": [],  # empty for non-VLM networks
            "parse_success": True,  # always True for non-VLM networks
        }

    def compute_loss(self, data, curr_target_v, curr_adv) -> tuple[torch.Tensor, dict, dict]:
        # Encode state: use seq_len frames (excluding last horizon frames)
        curr_obs = data.observations[:, : -self.horizon]
        curr_obs_z = data.obs_z[:, : -self.horizon]
        curr_actions = data.actions[:, : -self.horizon]
        curr_rewards = data.rewards[:, : -self.horizon]
        curr_rnn_state = data.rnn_state[:, 0]

        curr_state, _ = self.encoder.forward(
            curr_obs, curr_obs_z, curr_actions, curr_rewards, curr_rnn_state
        )

        # Get policy output with action chunk (B, horizon, action_dim)
        target_actions = data.actions[:, -self.horizon :]
        policy_dict = self.policy_head(curr_state, action=target_actions)
        a_logp = policy_dict["a_logp"]
        entropy = policy_dict["entropy"]
        policy_activation = policy_dict["activation"]

        # Get value output (state value at chunk start, i.e., last frame of input sequence)
        value_dict = self.value_head(curr_state)
        value = value_dict["output"]
        value_activation = value_dict["activation"]

        # Compute policy loss
        ratio = torch.exp(a_logp - data.log_probs[:, -self.horizon])
        surr1 = ratio * curr_adv
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param_policy, 1.0 + self.clip_param_policy)
            * curr_adv
        )
        action_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss
        if self.num_bins > 1:
            self.value_head.update_value_range(curr_target_v.squeeze(1))
            value_loss = self.value_head.value_loss(value, curr_target_v.squeeze(1))
        else:
            value_clipped = torch.clamp(
                value,
                data.values[:, -self.horizon] - self.clip_param_value,
                data.values[:, -self.horizon] + self.clip_param_value,
            )
            value_loss_unclipped = F.mse_loss(value, curr_target_v)
            value_loss_clipped = F.mse_loss(value_clipped, curr_target_v)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

        loss = action_loss + self.critic_loss_weight * value_loss - 0.02 * entropy.mean()

        activations_dict = {
            "state": curr_state,
            "actor": policy_activation,
            "critic": value_activation,
            "state_predictor": curr_state,
        }

        info_dict = {
            "actor_loss": action_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }

        return loss, activations_dict, info_dict

    def infer_and_compute_loss(self, data) -> tuple[dict, torch.Tensor, dict, dict]:
        """Combined inference and loss computation for streaming agent."""
        # --- Target value computation (no grad) ---
        with torch.no_grad():
            next_obs = data.observations[:, self.horizon :]
            next_obs_z = data.obs_z[:, self.horizon :]
            next_actions = data.actions[:, self.horizon :]
            next_rewards = data.rewards[:, self.horizon :]
            next_rnn_state = data.rnn_state[:, self.horizon]

            next_state, rnn_state_out = self.encoder.forward(
                next_obs, next_obs_z, next_actions, next_rewards, next_rnn_state
            )

            next_value_dict = self.value_head(next_state)
            next_value = self.value_head.to_value(next_value_dict["output"]).view(-1)

            # Generate inference action from next state
            policy_dict_next = self.policy_head(next_state, None)
            infer_action = policy_dict_next["action"]

            # Discounted reward over horizon
            chunk_rewards = data.rewards[:, -self.horizon :]
            chunk_dones = data.dones[:, -self.horizon :]
            batch_size = chunk_rewards.size(0)
            device = chunk_rewards.device
            discounted_reward = torch.zeros(batch_size, device=device)
            gamma_power = 1.0
            continuing = torch.ones(batch_size, device=device)
            for i in range(self.horizon):
                discounted_reward += continuing * gamma_power * chunk_rewards[:, i].flatten()
                gamma_power *= self.gamma
                continuing *= 1 - chunk_dones[:, i].flatten()
            target_value = discounted_reward + continuing * gamma_power * next_value

        # --- Current state for loss computation ---
        curr_obs = data.observations[:, : -self.horizon]
        curr_obs_z = data.obs_z[:, : -self.horizon]
        curr_actions = data.actions[:, : -self.horizon]
        curr_rewards = data.rewards[:, : -self.horizon]
        curr_rnn_state = data.rnn_state[:, 0]

        curr_state, _ = self.encoder.forward(
            curr_obs, curr_obs_z, curr_actions, curr_rewards, curr_rnn_state
        )

        # Policy evaluation with target actions
        target_actions = data.actions[:, -self.horizon :]
        policy_dict = self.policy_head(curr_state, action=target_actions)
        a_logp = policy_dict["a_logp"]
        entropy = policy_dict["entropy"]
        policy_activation = policy_dict["activation"]

        # Value evaluation
        value_dict = self.value_head(curr_state)
        value = value_dict["output"]
        value_activation = value_dict["activation"]

        # Advantage (no normalization for streaming B=1)
        with torch.no_grad():
            curr_v = self.value_head.to_value(value).view(-1)
            advantage = target_value - curr_v

        # Policy loss (REINFORCE-style, online so no PPO clipping needed)
        action_loss = -(a_logp * advantage.unsqueeze(1)).mean()

        # Value loss
        self.value_head.update_value_range(target_value)
        value_loss = self.value_head.value_loss(value, target_value)

        total_loss = action_loss + self.critic_loss_weight * value_loss - 0.02 * entropy.mean()

        # Actor-only loss (no critic component)
        actor_entropy_loss = action_loss - 0.02 * entropy.mean()

        # -V(s) for eligibility trace backward (detached from encoder)
        value_for_et_dict = self.value_head(curr_state.detach())
        neg_value_detached = -self.value_head.to_value(value_for_et_dict["output"]).mean()

        # Prediction for inference visualization
        next_image, next_reward = self.prediction_head.predict_next_state(
            next_state,
            infer_action[:, 0],
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        infer_dict = {
            "action": infer_action,
            "value": next_value.item(),
            "rnn_state": rnn_state_out,
            "next_image": next_image,
            "next_reward": next_reward,
        }

        activations_dict = {
            "state": curr_state,
            "actor": policy_activation,
            "critic": value_activation,
            "state_predictor": curr_state,
        }

        loss_info = {
            "actor_loss": action_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "delta": advantage.mean().item(),
        }

        et_info = {
            "actor_entropy_loss": actor_entropy_loss,
            "neg_value": neg_value_detached,
            "delta": advantage.mean().item(),
        }

        return infer_dict, total_loss, activations_dict, loss_info, et_info
