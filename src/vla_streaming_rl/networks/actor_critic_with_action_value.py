# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn

from vla_streaming_rl.networks.backbone import SpatialTemporalEncoder
from vla_streaming_rl.networks.image_processor import ImageProcessor
from vla_streaming_rl.networks.infer_result import ActivationFeatures, InferResult
from vla_streaming_rl.networks.loss_result import (
    EligibilityTraceInfo,
    InferLossResult,
    LossResult,
)
from vla_streaming_rl.networks.policy_head import (
    CFGDiffusionPolicy,
    DiffusionPolicy,
    MeanFlowPolicy,
)
from vla_streaming_rl.networks.prediction_head import StatePredictionHead
from vla_streaming_rl.networks.reward_processor import RewardProcessor
from vla_streaming_rl.networks.value_head import ActionValueHead, HypersphericalActionValueHead


class ActorCriticWithActionValue(nn.Module):
    def __init__(
        self,
        *,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        gamma: float,
        num_bins: int,
        sparsity: float,
        seq_len: int,
        dacer_loss_weight: float,
        critic_loss_weight: float,
        predictor_step_num: int,
        encoder_block_num: int,
        temporal_model_type: str,
        horizon: int,
        policy_type: str,
        actor_hidden_dim: int,
        actor_block_num: int,
        denoising_time: float,
        denoising_steps: int,
        som_alpha: float,
        som_w: float,
        critic_arch: str,
        critic_hidden_dim: int,
        critic_block_num: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        detach_actor: bool,
        detach_critic: bool,
        detach_predictor: bool,
        disable_state_predictor: bool,
        predictor_type: str,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.sparsity = sparsity
        self.seq_len = seq_len
        self.critic_loss_weight = critic_loss_weight

        self.action_dim = action_space_shape[0]
        self.predictor_step_num = predictor_step_num
        self.observation_space_shape = observation_space_shape

        self.image_processor = ImageProcessor(observation_space_shape)
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        self.encoder = SpatialTemporalEncoder(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            seq_len=self.seq_len,
            n_layer=encoder_block_num,
            action_dim=self.action_dim,
            temporal_model_type=temporal_model_type,
            use_image_only=True,
        )

        self.horizon = horizon
        self.policy_type = policy_type
        if self.policy_type == "diffusion":
            self.policy_head = DiffusionPolicy(
                state_dim=self.encoder.output_dim,
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
                state_dim=self.encoder.output_dim,
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
                state_dim=self.encoder.output_dim,
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
        # Critic architecture (the network owns hl_gauss, so ``num_bins`` from
        # config is used as-is for both):
        #   - ``simbav2`` : SimbaV2 hyperspherical critic (arXiv:2502.15280).
        #     Weights / features stay on the unit hypersphere so the critic's
        #     norm cannot blow up under the TD loss.
        #   - ``dueling`` : the original LayerNorm dueling V/A critic.
        if critic_arch == "simbav2":
            self.value_head = HypersphericalActionValueHead(
                in_channels=self.encoder.output_dim,
                action_dim=self.action_dim,
                horizon=horizon,
                gamma=gamma,
                hidden_dim=critic_hidden_dim,
                block_num=critic_block_num,
                num_bins=self.num_bins,
            )
        elif critic_arch == "dueling":
            self.value_head = ActionValueHead(
                in_channels=self.encoder.output_dim,
                action_dim=self.action_dim,
                horizon=horizon,
                gamma=gamma,
                hidden_dim=critic_hidden_dim,
                block_num=critic_block_num,
                num_bins=self.num_bins,
                sparsity=sparsity,
            )
        else:
            raise ValueError(f"Unknown critic_arch: {critic_arch!r} (expected 'simbav2'/'dueling')")
        self.prediction_head = StatePredictionHead(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            action_dim=self.action_dim,
            predictor_hidden_dim=predictor_hidden_dim,
            predictor_block_num=predictor_block_num,
            predictor_type=predictor_type,
        )

        self.detach_actor = detach_actor
        self.detach_critic = detach_critic
        self.detach_predictor = detach_predictor
        self.disable_state_predictor = disable_state_predictor

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
        obs_z_seq: torch.Tensor,  # (B, T, C', H', W')
        a_seq: torch.Tensor,  # (B, T, action_dim)
        r_seq: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,
        task_prompts: list[str],
    ) -> InferResult:
        assert s_seq.shape[0] == 1, "Batch size must be 1 for inference"

        x, rnn_state = self.encoder(s_seq, obs_z_seq, a_seq, r_seq, rnn_state)  # (B, hidden_dim)

        # Get action chunk from policy_head
        action, actor_activation = self.policy_head.get_action(x)  # (B, horizon, action_dim)

        # Get action-value from value_head
        q_dict = self.value_head(x, action)
        q_value = self.value_head.to_value(q_dict.output).item()

        # Get predicted next state
        next_image, next_reward, predictor_activation = self.prediction_head.predict_next_state(
            x,
            action[:, 0],  # use first action in chunk for prediction
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        activations = ActivationFeatures(
            state=x,
            actor=actor_activation,
            critic=q_dict.activation,
            state_predictor=predictor_activation,
        )

        return InferResult(
            action=action,
            value=q_value,
            rnn_state=rnn_state,
            next_image=next_image,
            next_reward=next_reward,
            action_token_ids=[],
            activations=activations,
        )

    def compute_loss(self, data) -> LossResult:
        _, _, next_q, _, _, _ = self._infer(
            data.observations[:, self.horizon :],
            data.obs_z[:, self.horizon :],
            data.actions[:, self.horizon :],
            data.rewards[:, self.horizon :],
            data.rnn_state[:, self.horizon],
        )
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self.value_head.compute_target_value(next_q, chunk_rewards, chunk_dones)

        # Use seq_len frames (excluding last horizon frames)
        curr_obs = data.observations[:, : -self.horizon]
        curr_obs_z = data.obs_z[:, : -self.horizon]
        curr_actions = data.actions[:, : -self.horizon]
        curr_rewards = data.rewards[:, : -self.horizon]
        curr_rnn_state = data.rnn_state[:, 0]  # (B, ...)

        curr_state, _ = self.encoder.forward(
            curr_obs, curr_obs_z, curr_actions, curr_rewards, curr_rnn_state
        )  # (B, state_dim)

        # Action chunk: (B, horizon, action_dim)
        action_chunk = data.actions[:, -self.horizon :]

        critic_loss, critic_info = self._compute_critic_loss(curr_state, action_chunk, target_value)
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            curr_state,
            action_chunk,
            value_head=self.value_head,
            detach_actor=self.detach_actor,
        )
        seq_loss, seq_info = self._compute_sequence_loss(data, curr_state)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        info_dict = {
            **critic_info,
            **actor_info,
            **seq_info,
        }

        return LossResult(loss=total_loss, info=info_dict)

    def infer_and_compute_loss(self, data) -> InferLossResult:
        """Combined inference and loss computation."""
        next_state, next_action, next_q, next_rnn_state, actor_activation, critic_activation = (
            self._infer(
                data.observations[:, self.horizon :],
                data.obs_z[:, self.horizon :],
                data.actions[:, self.horizon :],
                data.rewards[:, self.horizon :],
                data.rnn_state[:, self.horizon],
            )
        )
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self.value_head.compute_target_value(next_q, chunk_rewards, chunk_dones)

        prev_obs = data.observations[:, : -self.horizon]
        prev_obs_z = data.obs_z[:, : -self.horizon]
        prev_actions = data.actions[:, : -self.horizon]
        prev_rewards = data.rewards[:, : -self.horizon]
        prev_rnn_state = data.rnn_state[:, 0]

        prev_state, _ = self.encoder.forward(
            prev_obs, prev_obs_z, prev_actions, prev_rewards, prev_rnn_state
        )

        action_chunk = data.actions[:, -self.horizon :]

        critic_loss, critic_info = self._compute_critic_loss(prev_state, action_chunk, target_value)
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            prev_state,
            action_chunk,
            value_head=self.value_head,
            detach_actor=self.detach_actor,
        )
        seq_loss, seq_info = self._compute_sequence_loss(data, prev_state)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        # Actor-only loss (no critic component)
        actor_entropy_loss = actor_loss + seq_loss

        # -Q(s,a) for eligibility trace backward (detached from encoder)
        et_critic_dict = self.value_head(prev_state.detach(), action_chunk.detach())
        neg_value_detached = -self.value_head.to_value(et_critic_dict.output).mean()

        next_image, next_reward, predictor_activation = self.prediction_head.predict_next_state(
            next_state,
            next_action[:, 0],
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        activations = ActivationFeatures(
            state=next_state,
            actor=actor_activation,
            critic=critic_activation,
            state_predictor=predictor_activation,
        )

        infer_result = InferResult(
            action=next_action,
            value=next_q.item(),
            rnn_state=next_rnn_state,
            next_image=next_image,
            next_reward=next_reward,
            action_token_ids=[],
            activations=activations,
        )

        info_dict = {
            **critic_info,
            **actor_info,
            **seq_info,
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

    @torch.inference_mode()
    def _infer(
        self,
        obs: torch.Tensor,
        obs_z: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, rnn_state_out = self.encoder.forward(obs, obs_z, actions, rewards, rnn_state)
        action, actor_activation = self.policy_head.get_action(state)
        q_dict = self.value_head(state, action)
        q = self.value_head.to_value(q_dict.output).view(-1)
        return state, action, q, rnn_state_out, actor_activation, q_dict.activation

    def _compute_critic_loss(self, curr_state, action_chunk, target_value):
        """
        Args:
            curr_state: (B, state_dim)
            action_chunk: (B, horizon, action_dim)
            target_value: (B,)
        """
        if self.detach_critic:
            curr_state = curr_state.detach()

        curr_critic_output_dict = self.value_head(curr_state, action_chunk)
        logits = curr_critic_output_dict.output

        self.value_head.update_value_range(target_value)
        curr_critic_value = self.value_head.to_value(logits).view(-1)
        critic_loss = self.value_head.value_loss(logits, target_value)

        delta = target_value - curr_critic_value

        info_dict = {
            "delta": delta.mean().item(),
            "critic_loss": critic_loss.item(),
            "curr_critic_value": curr_critic_value.mean().item(),
            "target_value": target_value.mean().item(),
            "value_range": self.value_head.value_range,
        }

        return critic_loss, info_dict

    def _compute_sequence_loss(self, data, curr_state):
        if self.disable_state_predictor:
            # Return dummy loss when state_predictor is disabled
            dummy_loss = torch.tensor(0.0, device=curr_state.device, requires_grad=True)
            info_dict = {"seq_loss": 0.0}
            return dummy_loss, info_dict

        if self.detach_predictor:
            curr_state = curr_state.detach()

        # Get last action (actions[:, -1] corresponds to current_state)
        curr_action = data.actions[:, -1]  # (B, action_dim)

        # Encode next state
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

        curr_state = curr_state.view(B, -1, C)
        pred_loss, _, info_dict = self.prediction_head.compute_loss(curr_state, curr_action, x1)

        return pred_loss, info_dict
