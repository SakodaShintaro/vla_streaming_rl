# SPDX-License-Identifier: MIT
from collections.abc import Callable

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoConfig

from vla_streaming_rl.networks.interface import (
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from vla_streaming_rl.networks.modules.backbone import SpatialTemporalEncoder
from vla_streaming_rl.networks.modules.image_processor import ImageProcessor
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
        critic_loss_weight: float,
        prediction_head_factory,
        actor_critic_config: DictConfig,
        horizon: int,
        policy_head_factory,
        high_level_config: DictConfig,
    ) -> None:
        super().__init__()
        self.seq_len = actor_critic_config.seq_len
        self.critic_loss_weight = critic_loss_weight

        self.action_dim = action_space_shape[0]

        self.image_processor = ImageProcessor(
            observation_space_shape, actor_critic_config.image_encoder_type
        )
        hidden_image_dim = actor_critic_config.image_encoder_output_dim
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        assert 0.0 <= actor_critic_config.subtask_dropout < 1.0, actor_critic_config.subtask_dropout
        self.subtask_dropout = actor_critic_config.subtask_dropout
        assert 0.0 <= actor_critic_config.token_dropout < 1.0, actor_critic_config.token_dropout
        self.token_dropout = actor_critic_config.token_dropout
        # サブタスクを残した系列のうち、この割合でサブタスク以外の入力をすべて落とし、
        # テキストの指示だけから行動と価値を学ばせる
        assert 0.0 <= actor_critic_config.observation_dropout < 1.0, (
            actor_critic_config.observation_dropout
        )
        assert actor_critic_config.observation_dropout == 0.0 or (
            high_level_config.subtask_tokens_num > 0
        ), "observation_dropout leaves only the subtask, so it needs subtask_tokens_num > 0"
        self.observation_dropout = actor_critic_config.observation_dropout
        self.bc_loss_weight = actor_critic_config.bc_loss_weight
        self.scalar_obs_dim = 9
        self.scalar_obs_normalizer = RunningNormalizer(self.scalar_obs_dim)
        # ``subtask_tokens_num = 0`` is the ablation: the same body, the same heads
        # and the same loss with the chain's tokens taken out of the space axis,
        # which is what isolates what the chain contributes. No chain means no
        # VLM to load, so the width comes from the config, not a loaded model.
        subtask_tokens_num = high_level_config.subtask_tokens_num
        text_config = AutoConfig.from_pretrained(high_level_config.model_id).text_config
        subtask_dim = text_config.hidden_size
        # The embedding plus every layer's output.
        subtask_layers = text_config.num_hidden_layers + 1
        self.pool_subtask = actor_critic_config.subtask_pool == "mean" and subtask_tokens_num > 0
        self.subtask_shape = (
            1 if self.pool_subtask else subtask_tokens_num,
            subtask_layers,
            subtask_dim,
        )

        self.encoder = SpatialTemporalEncoder(
            image_features_shape=tuple(self.image_processor.output_shape),
            image_latent_dim=hidden_image_dim,
            reward_processor=self.reward_processor,
            seq_len=self.seq_len,
            n_layer=actor_critic_config.encoder_block_num,
            action_dim=self.action_dim,
            scalar_obs_dim=self.scalar_obs_dim,
            temporal_model_type=actor_critic_config.temporal_model_type,
            subtask_tokens_num=subtask_tokens_num,
            subtask_layers=subtask_layers,
            subtask_dim=subtask_dim,
            subtask_pool=actor_critic_config.subtask_pool,
            steps_per_reply=high_level_config.steps_per_reply,
            layer_scale_init=actor_critic_config.layer_scale_init,
        )

        self.horizon = horizon
        self.policy_head = policy_head_factory(
            state_dim=self.encoder.output_dim, action_dim=self.action_dim
        )

        self.value_head = value_head_factory(self.encoder.output_dim, self.action_dim)
        self.prediction_head = prediction_head_factory(
            image_latent_shape=(hidden_image_dim, self.encoder.hidden_h, self.encoder.hidden_w),
            reward_processor=self.reward_processor,
            action_dim=self.action_dim,
        )

        self.detach_predictor = actor_critic_config.detach_predictor
        self.disable_state_predictor = actor_critic_config.disable_state_predictor

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_state()

    def action_value(self, features: torch.Tensor, action_chunk: np.ndarray) -> float:
        """Q(s, a) the critic gives one ``(horizon, action_dim)`` chunk at the
        state ``infer`` handed back as ``features``."""
        chunk = torch.from_numpy(action_chunk).to(features.device, features.dtype).unsqueeze(0)
        return self.value_head.scalar_value(features, chunk).item()

    def stored_image_shape(self) -> tuple[int, ...]:
        """The frozen encoder's output for the frame, not the frame."""
        return tuple(self.image_processor.output_shape)

    def to_stored_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_processor.encode(image.unsqueeze(0)).squeeze(0)

    def to_stored_subtask(self, activations: torch.Tensor) -> torch.Tensor:
        """平均でまとめる設定なら、ステップのトークンを1つに均してから保存する。"""
        if self.pool_subtask:
            return activations.float().mean(dim=0, keepdim=True).to(activations.dtype)
        return activations

    def _subtask_keep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Which sequences of a batch keep their chain, the rest being the share
        ``subtask_dropout`` it is taken away from. Handed to the encoder as a mask
        rather than applied to the activations here, which would copy them.

        Only on the learning path. Without it the critic has never scored a
        state whose chain is missing, so asking it what an action is worth
        without one is asking about an input it was never trained on -- which is
        exactly what the chain's own contribution has to be measured against.
        Dropping whole sequences rather than single steps keeps a sampled window
        internally consistent.
        """
        if self.subtask_dropout == 0.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return torch.rand(batch_size, device=device) >= self.subtask_dropout

    def _token_keep(self, batch_size: int, steps: int, device: torch.device) -> torch.Tensor:
        """Which tokens of a batch's windows the encoder gets to read, the rest
        being the share ``token_dropout`` takes away: every token alike, whatever
        it carries, so no one kind of input can be leaned on alone. Only on the
        learning path, as the chain's own dropout is."""
        shape = (batch_size, steps, self.encoder.space_len)
        return torch.rand(shape, device=device) >= self.token_dropout

    def _window(self, data: ReplayBufferData, start, stop) -> tuple:
        """The ``(image, action, reward, rnn_state, scalar_obs, subtask, subtask_age,
        subtask_keep, token_keep)`` the encoder reads, sliced out of a replay batch
        over ``[start, stop)`` steps."""
        observations = data.observations[:, start:stop]
        batch_size = observations.shape[0]
        subtask_keep = self._subtask_keep(batch_size, observations.device)
        token_keep = self._token_keep(batch_size, observations.shape[1], observations.device)
        # サブタスクのトークンは1ステップのトークンの最後に並ぶので、それより前をすべて落とす
        subtask_only = subtask_keep & (
            torch.rand(batch_size, device=observations.device) < self.observation_dropout
        )
        token_keep[subtask_only, :, : -self.subtask_shape[0]] = False
        return (
            observations,
            data.actions[:, start:stop],
            data.rewards[:, start:stop],
            data.rnn_state[:, start],
            self._scalar_obs(
                data.velocity_x[:, start:stop],
                data.velocity_y[:, start:stop],
                data.velocity_z[:, start:stop],
                data.episode_return[:, start:stop],
                data.pass_mark[:, start:stop],
                data.remaining_return[:, start:stop],
                data.global_step[:, start:stop],
                data.episode_step[:, start:stop],
                data.health[:, start:stop],
            ),
            data.subtask_activations[:, start:stop],
            data.subtask_age[:, start:stop],
            subtask_keep,
            token_keep,
        )

    def tokenize(self, text: str) -> list[int]:
        del text
        return []

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
        scalar_obs = np.array(
            [
                velocity_x,
                velocity_y,
                velocity_z,
                episode_return,
                pass_mark,
                remaining_return,
                global_step,
                episode_step,
                health,
            ],
            dtype=np.float32,
        )
        self.scalar_obs_normalizer.update(scalar_obs)

    def _scalar_obs(
        self,
        velocity_x: torch.Tensor,
        velocity_y: torch.Tensor,
        velocity_z: torch.Tensor,
        episode_return: torch.Tensor,
        pass_mark: torch.Tensor,
        remaining_return: torch.Tensor,
        global_step: torch.Tensor,
        episode_step: torch.Tensor,
        health: torch.Tensor,
    ) -> torch.Tensor:
        raw = torch.cat(
            [
                velocity_x,
                velocity_y,
                velocity_z,
                episode_return,
                pass_mark,
                remaining_return,
                global_step,
                episode_step,
                health,
            ],
            dim=-1,
        )
        return self.scalar_obs_normalizer.normalize(raw)

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        assert data.s_seq.shape[0] == 1, "Batch size must be 1 for inference"

        scalar_obs = self._scalar_obs(
            data.velocity_x_seq,
            data.velocity_y_seq,
            data.velocity_z_seq,
            data.episode_return_seq,
            data.pass_mark_seq,
            data.remaining_return_seq,
            data.global_step_seq,
            data.episode_step_seq,
            data.health_seq,
        )
        x, rnn_state = self.encoder(
            data.s_seq,
            data.a_seq,
            data.r_seq,
            data.rnn_state,
            scalar_obs,
            data.subtask_activations_seq,
            data.subtask_age_seq,
            torch.ones(1, dtype=torch.bool, device=data.subtask_activations_seq.device),
            torch.ones(
                (1, data.s_seq.shape[1], self.encoder.space_len),
                dtype=torch.bool,
                device=data.s_seq.device,
            ),
        )  # (B, state_dim)

        # Get action chunk from policy_head
        action, _ = self.policy_head.get_action(x)  # (B, horizon, action_dim)

        # Get action-value from value_head
        q_out = self.value_head(x, action)
        value_report = self.value_head.value_report(q_out.output)

        return InferResult(
            action=action,
            value_report=value_report,
            rnn_state=rnn_state,
            features=x,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        # Bootstrap value: Q(s', μ(s')) on the next-state window, no grad.
        with torch.inference_mode():
            next_state, _ = self.encoder(*self._window(data, self.horizon, None))
            next_action, _ = self.policy_head.get_action(next_state)
            next_output = self.value_head(next_state, next_action).output
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self.value_head.compute_target_value(next_output, chunk_rewards, chunk_dones)

        # Use seq_len frames (excluding last horizon frames)
        curr_state, _ = self.encoder(*self._window(data, 0, -self.horizon))

        # Action chunk: (B, horizon, action_dim)
        action_chunk = data.actions[:, -self.horizon :]

        critic_loss, critic_info = self.value_head.compute_critic_loss(
            curr_state, action_chunk, target_value
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            curr_state,
            action_chunk,
            value_head=self.value_head,
        )
        head_action, _ = self.policy_head.get_action(curr_state)
        bc_gap = (head_action - data.vlm_actions[:, -self.horizon :]).pow(2).mean(dim=-1)
        bc_holds = data.vlm_holds[:, -self.horizon :, 0]
        bc_loss = (bc_gap * bc_holds).sum() / bc_holds.sum().clamp(min=1.0)
        with torch.no_grad():
            next_image_latent = self.encoder.image_projection(data.observations[:, -self.horizon])
        seq_loss, seq_info = self.prediction_head.compute_loss(
            curr_state,
            data.actions[:, -self.horizon],
            next_image_latent,
            data.rewards[:, -self.horizon],
            self.detach_predictor,
            self.disable_state_predictor,
        )

        total_loss = (
            self.critic_loss_weight * critic_loss
            + actor_loss
            + seq_loss
            + self.bc_loss_weight * bc_loss
        )

        info_dict = {
            f"losses/{key}": value
            for key, value in {
                **critic_info,
                **actor_info,
                **seq_info,
                "bc_loss": bc_loss.item(),
            }.items()
        }

        return LossResult(loss=total_loss, info=info_dict)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        """Combined inference and loss computation."""
        # Next-step inference (no grad): the action the agent will take and its Q.
        with torch.inference_mode():
            next_state, next_rnn_state = self.encoder(*self._window(data, self.horizon, None))
            next_action, _ = self.policy_head.get_action(next_state)
            next_q_out = self.value_head(next_state, next_action)
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self.value_head.compute_target_value(
            next_q_out.output, chunk_rewards, chunk_dones
        )

        prev_state, _ = self.encoder(*self._window(data, 0, -self.horizon))

        action_chunk = data.actions[:, -self.horizon :]

        critic_loss, critic_info = self.value_head.compute_critic_loss(
            prev_state, action_chunk, target_value
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            prev_state,
            action_chunk,
            value_head=self.value_head,
        )
        with torch.no_grad():
            next_image_latent = self.encoder.image_projection(data.observations[:, -self.horizon])
        seq_loss, seq_info = self.prediction_head.compute_loss(
            prev_state,
            data.actions[:, -self.horizon],
            next_image_latent,
            data.rewards[:, -self.horizon],
            self.detach_predictor,
            self.disable_state_predictor,
        )

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        # Actor-only loss (no critic component)
        actor_entropy_loss = actor_loss + seq_loss

        # -Q(s,a) for eligibility trace backward (detached from encoder)
        neg_value_detached = -self.value_head.scalar_value(
            prev_state.detach(), action_chunk.detach()
        ).mean()

        infer_result = InferResult(
            action=next_action,
            value_report=self.value_head.value_report(next_q_out.output),
            rnn_state=next_rnn_state,
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
