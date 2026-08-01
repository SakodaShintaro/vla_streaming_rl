# SPDX-License-Identifier: MIT
from collections.abc import Callable

import numpy as np
import torch

from vla_streaming_rl.networks.interface import (
    ActivationFeatures,
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from vla_streaming_rl.networks.modules.backbone import SpatialTemporalEncoder
from vla_streaming_rl.networks.modules.image_processor import ImageProcessor
from vla_streaming_rl.networks.modules.policy_head import (
    CFGDiffusionPolicy,
    DiffusionPolicy,
    MeanFlowPolicy,
)
from vla_streaming_rl.networks.modules.prediction_head import StatePredictionHead
from vla_streaming_rl.networks.modules.reward_processor import RewardProcessor
from vla_streaming_rl.networks.modules.value_head import DistributionalValueHead
from vla_streaming_rl.replay_buffer import ReplayBufferData
from vla_streaming_rl.reward_processor import RunningNormalizer


class ActorCriticWithActionValue(NetworkInterface):
    def __init__(
        self,
        *,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        value_head_factory: Callable[[int, int], DistributionalValueHead],
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
        self.image_numel = int(np.prod(observation_space_shape))

        self.image_processor = ImageProcessor(observation_space_shape)
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        self.scalar_obs_dim = 5
        self.scalar_obs_normalizer = RunningNormalizer(self.scalar_obs_dim)
        self.encoder = SpatialTemporalEncoder(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            seq_len=self.seq_len,
            n_layer=encoder_block_num,
            action_dim=self.action_dim,
            scalar_obs_dim=self.scalar_obs_dim,
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

        self.value_head = value_head_factory(self.encoder.output_dim, self.action_dim)
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

    @property
    def packed_obs_shape(self) -> tuple[int, ...]:
        return (self.image_numel + self.scalar_obs_dim,)

    def observe_scalar_obs(self, scalar_obs: np.ndarray) -> None:
        self.scalar_obs_normalizer.update(scalar_obs)

    def pack_obs(self, image: torch.Tensor, scalar_obs: torch.Tensor) -> torch.Tensor:
        # Store raw scalar_obs next to the flattened image; normalization is applied
        # at unpack with the latest running statistics (as with the reward).
        return torch.cat([image.reshape(*image.shape[:-3], -1), scalar_obs], dim=-1)

    def _unpack_obs(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = packed[..., : self.image_numel].reshape(
            *packed.shape[:-1], *self.observation_space_shape
        )
        scalar_obs = self.scalar_obs_normalizer.normalize(packed[..., self.image_numel :])
        return image, scalar_obs

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        assert data.s_seq.shape[0] == 1, "Batch size must be 1 for inference"

        image, scalar_obs = self._unpack_obs(data.s_seq)
        x, rnn_state = self.encoder(
            image, data.a_seq, data.r_seq, data.rnn_state, scalar_obs
        )  # (B, hidden_dim)

        # Get action chunk from policy_head
        action, actor_activation = self.policy_head.get_action(x)  # (B, horizon, action_dim)

        # Get action-value from value_head
        q_out = self.value_head(x, action)
        value_report = self.value_head.value_report(q_out.output)

        # Get predicted next state (image + reward, both in latent space)
        next_image_latent, next_reward_latent, predictor_activation = (
            self.prediction_head.predict_next_state(
                x,
                action[:, 0],  # use first action in chunk for prediction
                self.predictor_step_num,
                self.disable_state_predictor,
            )
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
            next_image_latent=next_image_latent,
            next_reward_latent=next_reward_latent,
            activations=activations,
            features=x,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        # Bootstrap value: Q(s', μ(s')) on the next-state window, no grad.
        with torch.inference_mode():
            next_image, next_scalar_obs = self._unpack_obs(data.observations[:, self.horizon :])
            next_state, _ = self.encoder.forward(
                next_image,
                data.actions[:, self.horizon :],
                data.rewards[:, self.horizon :],
                data.rnn_state[:, self.horizon],
                next_scalar_obs,
            )
            next_action, _ = self.policy_head.get_action(next_state)
            next_output = self.value_head(next_state, next_action).output
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self.value_head.compute_target_value(next_output, chunk_rewards, chunk_dones)

        # Use seq_len frames (excluding last horizon frames)
        curr_image, curr_scalar_obs = self._unpack_obs(data.observations[:, : -self.horizon])
        curr_actions = data.actions[:, : -self.horizon]
        curr_rewards = data.rewards[:, : -self.horizon]
        curr_rnn_state = data.rnn_state[:, 0]  # (B, ...)

        curr_state, _ = self.encoder.forward(
            curr_image, curr_actions, curr_rewards, curr_rnn_state, curr_scalar_obs
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
        target_image, _ = self._unpack_obs(data.observations[:, -1])
        seq_loss, seq_info = self.prediction_head.compute_loss(
            curr_state,
            data.actions[:, -1],
            target_image,
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
        """Combined inference and loss computation."""
        # Next-step inference (no grad): the action the agent will take, its Q,
        # and the activations carried into the InferResult.
        with torch.inference_mode():
            next_image, next_scalar_obs = self._unpack_obs(data.observations[:, self.horizon :])
            next_state, next_rnn_state = self.encoder.forward(
                next_image,
                data.actions[:, self.horizon :],
                data.rewards[:, self.horizon :],
                data.rnn_state[:, self.horizon],
                next_scalar_obs,
            )
            next_action, actor_activation = self.policy_head.get_action(next_state)
            next_q_out = self.value_head(next_state, next_action)
            critic_activation = next_q_out.activation
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self.value_head.compute_target_value(
            next_q_out.output, chunk_rewards, chunk_dones
        )

        prev_image, prev_scalar_obs = self._unpack_obs(data.observations[:, : -self.horizon])
        prev_actions = data.actions[:, : -self.horizon]
        prev_rewards = data.rewards[:, : -self.horizon]
        prev_rnn_state = data.rnn_state[:, 0]

        prev_state, _ = self.encoder.forward(
            prev_image, prev_actions, prev_rewards, prev_rnn_state, prev_scalar_obs
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
        target_image, _ = self._unpack_obs(data.observations[:, -1])
        seq_loss, seq_info = self.prediction_head.compute_loss(
            prev_state,
            data.actions[:, -1],
            target_image,
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

        next_image_latent, next_reward_latent, predictor_activation = (
            self.prediction_head.predict_next_state(
                next_state,
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
            value_report=self.value_head.value_report(next_q_out.output),
            rnn_state=next_rnn_state,
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
