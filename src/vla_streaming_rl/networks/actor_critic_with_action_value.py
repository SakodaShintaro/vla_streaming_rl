# SPDX-License-Identifier: MIT
from collections.abc import Callable

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
from vla_streaming_rl.networks.value_head import DistributionalValueHead


class ActorCriticWithActionValue(nn.Module):
    def __init__(
        self,
        *,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        value_head_factory: Callable[[int], DistributionalValueHead],
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
        predictor_hidden_dim: int,
        predictor_block_num: int,
        detach_actor: bool,
        detach_critic: bool,
        detach_predictor: bool,
        disable_state_predictor: bool,
        predictor_type: str,
    ) -> None:
        super().__init__()
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
        # The critic (action-value head) is injected as a factory so that all
        # value-related construction lives in the network builder, not here. We
        # only supply the state width the encoder produces.
        self.value_head = value_head_factory(self.encoder.output_dim)
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
        q_out = self.value_head(x, action)
        value_report = self.value_head.value_report(q_out.output)

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
            critic=q_out.activation,
            state_predictor=predictor_activation,
        )

        return InferResult(
            action=action,
            value_report=value_report,
            rnn_state=rnn_state,
            next_image=next_image,
            next_reward=next_reward,
            action_token_ids=[],
            activations=activations,
        )

    def compute_loss(self, data) -> LossResult:
        # Bootstrap value: Q(s', μ(s')) on the next-state window, no grad.
        with torch.inference_mode():
            next_state, _ = self.encoder.forward(
                data.observations[:, self.horizon :],
                data.obs_z[:, self.horizon :],
                data.actions[:, self.horizon :],
                data.rewards[:, self.horizon :],
                data.rnn_state[:, self.horizon],
            )
            next_action, _ = self.policy_head.get_action(next_state)
            # Per-gamma bootstrap values (B, num_gammas) for the multi-gamma TD target.
            next_q = self.value_head.to_values(self.value_head(next_state, next_action).output)
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

        critic_loss, critic_info = self.value_head.compute_critic_loss(
            curr_state, action_chunk, target_value, self.detach_critic
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            curr_state,
            action_chunk,
            value_head=self.value_head,
            detach_actor=self.detach_actor,
        )
        seq_loss, seq_info = self.prediction_head.compute_loss(
            curr_state,
            data.actions[:, -1],
            data.observations[:, -1],
            data.rewards[:, -1],
            self.detach_predictor,
            self.disable_state_predictor,
        )

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        info_dict = {
            **critic_info,
            **actor_info,
            **seq_info,
        }

        return LossResult(loss=total_loss, info=info_dict)

    def infer_and_compute_loss(self, data) -> InferLossResult:
        """Combined inference and loss computation."""
        # Next-step inference (no grad): the action the agent will take, its Q,
        # and the activations carried into the InferResult.
        with torch.inference_mode():
            next_state, next_rnn_state = self.encoder.forward(
                data.observations[:, self.horizon :],
                data.obs_z[:, self.horizon :],
                data.actions[:, self.horizon :],
                data.rewards[:, self.horizon :],
                data.rnn_state[:, self.horizon],
            )
            next_action, actor_activation = self.policy_head.get_action(next_state)
            next_q_out = self.value_head(next_state, next_action)
            # Per-gamma bootstrap values (B, num_gammas) for the multi-gamma TD target.
            next_q = self.value_head.to_values(next_q_out.output)
            critic_activation = next_q_out.activation
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

        critic_loss, critic_info = self.value_head.compute_critic_loss(
            prev_state, action_chunk, target_value, self.detach_critic
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            prev_state,
            action_chunk,
            value_head=self.value_head,
            detach_actor=self.detach_actor,
        )
        seq_loss, seq_info = self.prediction_head.compute_loss(
            prev_state,
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
        et_critic_out = self.value_head(prev_state.detach(), action_chunk.detach())
        neg_value_detached = -self.value_head.to_value(et_critic_out.output).mean()

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
            value_report=self.value_head.value_report(next_q_out.output),
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
