# SPDX-License-Identifier: MIT
"""The Animal-AI backbone with a diffusion policy and an action-value head.

The body is the winning network's, unchanged: :class:`AnimalBackbone` -- the
Fixup residual tower with channel attention, the dense branch and the recurrent
cell. What sits on top is not PPO's categorical logits and state value but the
pair the off-policy and streaming agents train, a diffusion policy head over an
action chunk and a distributional Q(s, a), so this is a :class:`NetworkInterface` and
:class:`OffPolicyAgent` and :class:`StreamingAgent` both drive it.

The two differences from ``AnimalPPONetwork``'s use of the same body:

- the dense branch reads the scalar observations the replay buffer stores (with
  the previous action and reward appended) rather than the hand-scaled
  velocity/clock vector the PPO agent builds, and
- there is no state predictor. The prediction fields of :class:`InferResult` are
  part of that contract, so they are filled with zeros.
"""

import numpy as np
import torch

from vla_streaming_rl.networks.animal_ppo import TEMPORAL_UNITS, AnimalBackbone
from vla_streaming_rl.networks.interface import (
    ActivationFeatures,
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from vla_streaming_rl.networks.modules.policy_head import build_policy_head
from vla_streaming_rl.replay_buffer import ReplayBufferData
from vla_streaming_rl.reward_processor import RunningNormalizer

SCALAR_OBS_DIM = 9


class AnimalEncoder(torch.nn.Module):
    """:class:`AnimalBackbone` behind the encoder contract the networks use:
    a ``(B, T, ...)`` window in, the last step's recurrent output out.

    The backbone's batch is environment-major (environment e's step t at
    ``e * T + t``), which is exactly what flattening a ``(B, T, ...)`` window
    gives, so the window maps onto it by a reshape. Episode boundaries inside the
    window are not replayed -- the mask is zero throughout -- as with the other
    encoders here: the state carried in is the one recorded at the window's first
    step, and the agent's windows are short.
    """

    def __init__(
        self,
        observation_space_shape: tuple[int, ...],
        action_dim: int,
        scalar_obs_dim: int,
        image_encoder_type: str,
        image_encoder_output_dim: int,
        image_encode_mode: str,
        image_encoder_trainable: bool,
        temporal_model_type: str,
    ) -> None:
        super().__init__()
        self.backbone = AnimalBackbone(
            observation_space_shape,
            scalar_obs_dim + action_dim + 1,
            image_encoder_type,
            image_encoder_output_dim,
            image_encode_mode,
            image_encoder_trainable,
            temporal_model_type,
        )
        self.output_dim = TEMPORAL_UNITS

    def init_state(self) -> torch.Tensor:
        return self.backbone.init_state(1, torch.device("cpu"))

    def forward_steps(
        self,
        images: torch.Tensor,  # (B, T, C, H, W)
        actions: torch.Tensor,  # (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (B, state_size)
        scalar_obs: torch.Tensor,  # (B, T, scalar_obs_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every step of the window rather than only its last: the recurrent
        output ``(B, T, TEMPORAL_UNITS)``, the per-step visual latent
        ``(B, T, HIDDEN_NODES)``, and the state."""
        batch_size, steps_num = images.shape[:2]
        flat_num = batch_size * steps_num
        visual = images.reshape(flat_num, *images.shape[2:])
        vels = torch.cat([scalar_obs, actions, rewards], dim=-1).reshape(flat_num, -1)
        masks = torch.zeros(flat_num, device=images.device, dtype=images.dtype)
        visual_latent, hidden = self.backbone.embed(visual, vels)
        temporal_out, state = self.backbone.recurrent(hidden, rnn_state, masks, batch_size)
        return (
            temporal_out.reshape(batch_size, steps_num, -1),
            visual_latent.reshape(batch_size, steps_num, -1),
            state,
        )

    def forward(
        self,
        images: torch.Tensor,  # (B, T, C, H, W)
        actions: torch.Tensor,  # (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (B, state_size)
        scalar_obs: torch.Tensor,  # (B, T, scalar_obs_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal_out, _, state = self.forward_steps(images, actions, rewards, rnn_state, scalar_obs)
        return temporal_out[:, -1], state


class AnimalActorCriticWithActionValue(NetworkInterface):
    def __init__(
        self,
        *,
        observation_space_shape: tuple[int, ...],
        action_space_shape: tuple[int, ...],
        value_head_factory,
        horizon: int,
        policy_type: str,
        actor_hidden_dim: int,
        actor_block_num: int,
        denoising_time: float,
        denoising_steps: int,
        dacer_loss_weight: float,
        som_alpha: float,
        som_w: float,
        sparsity: float,
        critic_loss_weight: float,
        detach_actor: bool,
        detach_critic: bool,
        image_encoder_type: str,
        image_encoder_output_dim: int,
        image_encode_mode: str,
        image_encoder_trainable: bool,
        temporal_model_type: str,
    ) -> None:
        super().__init__()
        self.observation_space_shape = observation_space_shape
        self.action_dim = action_space_shape[0]
        self.horizon = horizon
        self.critic_loss_weight = critic_loss_weight
        self.detach_actor = detach_actor
        self.detach_critic = detach_critic

        self.scalar_obs_normalizer = RunningNormalizer(SCALAR_OBS_DIM)
        self.encoder = AnimalEncoder(
            observation_space_shape,
            self.action_dim,
            SCALAR_OBS_DIM,
            image_encoder_type,
            image_encoder_output_dim,
            image_encode_mode,
            image_encoder_trainable,
            temporal_model_type,
        )

        self.policy_type = policy_type
        self.policy_head = build_policy_head(
            policy_type=policy_type,
            state_dim=self.encoder.output_dim,
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
        self.value_head = value_head_factory(self.encoder.output_dim, self.action_dim)

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_state()

    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
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
        self.scalar_obs_normalizer.update(
            np.array(
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
        )

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

    def _empty_prediction(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The prediction fields of the shared result types, for a network that
        has no state predictor."""
        zeros = torch.zeros((state.shape[0], 1), device=state.device)
        return zeros, zeros

    def _window(self, data: ReplayBufferData, start: int, stop) -> tuple:
        """The ``(image, action, reward, rnn_state, scalar_obs)`` the encoder
        reads, sliced out of a replay batch over ``[start, stop)`` steps."""
        return (
            data.observations[:, start:stop],
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
        )

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
            data.s_seq, data.a_seq, data.r_seq, data.rnn_state, scalar_obs
        )  # (B, TEMPORAL_UNITS)

        action, actor_activation = self.policy_head.get_action(x)  # (B, horizon, action_dim)
        q_out = self.value_head(x, action)
        next_image_latent, next_reward_latent = self._empty_prediction(x)

        return InferResult(
            action=action,
            value_report=self.value_head.value_report(q_out.output),
            rnn_state=rnn_state,
            next_image_latent=next_image_latent,
            next_reward_latent=next_reward_latent,
            activations=ActivationFeatures(
                state=x,
                actor=actor_activation,
                critic=q_out.activation,
                state_predictor=next_image_latent,
            ),
            features=x,
        )

    def _encode_current_window(
        self, data: ReplayBufferData
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """The state the actor and critic read, plus whatever auxiliary loss the
        network trains on that same window -- none here."""
        curr_state, _ = self.encoder(*self._window(data, 0, -self.horizon))
        return curr_state, torch.zeros((), device=curr_state.device), {}

    def _losses(
        self,
        curr_state: torch.Tensor,
        action_chunk: torch.Tensor,
        target_value: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        auxiliary_info: dict,
    ) -> tuple[LossResult, torch.Tensor, float]:
        """The training loss, plus the two quantities the eligibility-trace
        critic step needs on its own: the actor-only loss and the TD error."""
        critic_loss, critic_info = self.value_head.compute_critic_loss(
            curr_state, action_chunk, target_value, self.detach_critic
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            curr_state,
            action_chunk,
            value_head=self.value_head,
            detach_actor=self.detach_actor,
        )
        total_loss = self.critic_loss_weight * critic_loss + actor_loss + auxiliary_loss
        info_dict = {
            f"losses/{key}": value
            for key, value in {**critic_info, **actor_info, **auxiliary_info}.items()
        }
        return LossResult(loss=total_loss, info=info_dict), actor_loss, critic_info["delta"]

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        # Bootstrap value: Q(s', mu(s')) on the next-state window, no grad.
        with torch.inference_mode():
            next_state, _ = self.encoder(*self._window(data, self.horizon, None))
            next_action, _ = self.policy_head.get_action(next_state)
            next_output = self.value_head(next_state, next_action).output
        target_value = self.value_head.compute_target_value(
            next_output, data.rewards[:, -self.horizon :], data.dones[:, -self.horizon :]
        )

        curr_state, auxiliary_loss, auxiliary_info = self._encode_current_window(data)
        loss_result, _, _ = self._losses(
            curr_state,
            data.actions[:, -self.horizon :],
            target_value,
            auxiliary_loss,
            auxiliary_info,
        )
        return loss_result

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        """Combined inference and loss computation: the action the agent takes
        next comes from the same next-state window the TD target bootstraps on."""
        with torch.inference_mode():
            next_state, next_rnn_state = self.encoder(*self._window(data, self.horizon, None))
            next_action, actor_activation = self.policy_head.get_action(next_state)
            next_q_out = self.value_head(next_state, next_action)
        target_value = self.value_head.compute_target_value(
            next_q_out.output, data.rewards[:, -self.horizon :], data.dones[:, -self.horizon :]
        )

        prev_state, auxiliary_loss, auxiliary_info = self._encode_current_window(data)
        action_chunk = data.actions[:, -self.horizon :]
        loss_result, actor_loss, delta = self._losses(
            prev_state, action_chunk, target_value, auxiliary_loss, auxiliary_info
        )

        # -Q(s, a) for the eligibility-trace critic step, detached from the encoder.
        et_critic_out = self.value_head(prev_state.detach(), action_chunk.detach())
        neg_value_detached = -self.value_head.to_value(et_critic_out.output).mean()

        next_image_latent, next_reward_latent = self._empty_prediction(next_state)

        infer_result = InferResult(
            action=next_action,
            value_report=self.value_head.value_report(next_q_out.output),
            rnn_state=next_rnn_state,
            next_image_latent=next_image_latent,
            next_reward_latent=next_reward_latent,
            activations=ActivationFeatures(
                state=next_state,
                actor=actor_activation,
                critic=next_q_out.activation,
                state_predictor=next_image_latent,
            ),
            features=next_state,
        )

        return InferLossResult(
            infer_result=infer_result,
            loss_result=loss_result,
            et_info=EligibilityTraceInfo(
                actor_entropy_loss=actor_loss,
                neg_value=neg_value_detached,
                delta=delta,
            ),
        )
