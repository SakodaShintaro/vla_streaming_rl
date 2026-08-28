# SPDX-License-Identifier: MIT
"""Actor-critic over an image, the interoceptive scalars and a running chain of thought.

Two loops at two rates. The slow one is :class:`CoTStream`: a frozen VLM reasons
about a frame for as long as its chain lasts, handing out a fixed budget of
lm_head-input activations per environment step and reading a new frame only when
the chain ends. The fast one is this network: every step it sees the current
frame, the interoceptive scalars and whatever the chain issued on that step,
lays them flat on the space axis of :class:`CoTEncoder`, and reads the last
step's slice as the state its diffusion policy and action-value critic consume.

The VLM never trains and its sampling is not differentiable, so the chain is an
observation, not a decision this network can shape. That is what makes it
storable: the activations go into the replay buffer with the image, and no
replayed update ever runs the VLM again.
"""

import numpy as np
import torch
from transformers import AutoConfig

from vla_streaming_rl.networks.interface import (
    ActivationFeatures,
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from vla_streaming_rl.networks.modules.cot_backbone import CoTEncoder
from vla_streaming_rl.networks.modules.cot_stream import CoTStream
from vla_streaming_rl.networks.modules.image_processor import ImageProcessor
from vla_streaming_rl.networks.modules.policy_head import build_policy_head
from vla_streaming_rl.networks.modules.reward_processor import RewardProcessor
from vla_streaming_rl.replay_buffer import ReplayBufferData
from vla_streaming_rl.reward_processor import RunningNormalizer
from vla_streaming_rl.utils import render_text_panel

SCALAR_OBS_DIM = 9


class CoTActorCritic(NetworkInterface):
    # Fixed so the render strip keeps one shape for the whole run; wide enough
    # to read a chain of ``cot_max_len`` tokens.
    COT_PANEL_WIDTH = 420
    COT_PANEL_HEIGHT = 300

    def __init__(
        self,
        *,
        observation_space_shape: tuple[int, ...],
        action_space_shape: tuple[int, ...],
        value_head_factory,
        seq_len: int,
        horizon: int,
        encoder_block_num: int,
        temporal_model_type: str,
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
        cot_model_id: str,
        cot_tokens_num: int,
        cot_max_len: int,
        cot_temperature: float,
        cot_cuda_graph: bool,
    ) -> None:
        super().__init__()
        self.observation_space_shape = observation_space_shape
        self.action_dim = action_space_shape[0]
        self.horizon = horizon
        self.critic_loss_weight = critic_loss_weight
        self.detach_actor = detach_actor
        self.detach_critic = detach_critic

        # The chain's tokens join the patch grid on the space axis, so a single
        # pooled image token would leave the spatial attention nothing to relate
        # them to.
        assert image_encode_mode == "grid"
        self.image_processor = ImageProcessor(
            observation_space_shape,
            image_encoder_type,
            image_encoder_output_dim,
            image_encode_mode,
            image_encoder_trainable,
        )
        self.reward_processor = RewardProcessor(embed_dim=self.image_processor.output_shape[0])

        # ``cot_tokens_num = 0`` is the ablation: the same network and the same
        # loss with the chain's tokens taken out of the space axis, which is what
        # isolates what the chain contributes. No chain means no VLM to load, so
        # the width comes from the config instead of the loaded model.
        cot_dim = AutoConfig.from_pretrained(cot_model_id).text_config.hidden_size
        self.cot_shape = (cot_tokens_num, cot_dim)
        # Not a submodule: the frozen VLM must stay out of parameters()/state_dict().
        self.cot_stream = None
        if cot_tokens_num > 0:
            self.cot_stream = CoTStream(
                model_id=cot_model_id,
                tokens_per_step=cot_tokens_num,
                max_len=cot_max_len,
                temperature=cot_temperature,
                use_cuda_graph=cot_cuda_graph,
                device=torch.device("cuda"),
            )

        self.scalar_obs_normalizer = RunningNormalizer(SCALAR_OBS_DIM)
        self.encoder = CoTEncoder(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            seq_len=seq_len,
            n_layer=encoder_block_num,
            action_dim=self.action_dim,
            scalar_obs_dim=SCALAR_OBS_DIM,
            temporal_model_type=temporal_model_type,
            cot_tokens_num=cot_tokens_num,
            cot_dim=cot_dim,
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

    def advance_cot(
        self, image: torch.Tensor, task_prompt: str, episode_done: bool
    ) -> torch.Tensor:
        """This step's chain-of-thought activations, or nothing when the chain is
        off. A finished episode ends the chain, so the next one starts its
        commentary from its own first frame."""
        if self.cot_stream is None:
            return torch.zeros(self.cot_shape)
        if episode_done:
            self.cot_stream.reset()
        return self.cot_stream.advance(image, task_prompt)

    def render_panels(self) -> dict[str, np.ndarray]:
        """The chain as it currently stands, drawn for the render strip. It runs
        from the frame the chain started on, so it empties whenever the chain
        restarts. The ablation contributes no panel at all rather than a blank
        one, which keeps its render strip the width of what it actually has."""
        if self.cot_stream is None:
            return {}
        return {
            "chain_of_thought": render_text_panel(
                self.cot_stream.text(), self.COT_PANEL_WIDTH, self.COT_PANEL_HEIGHT
            )
        }

    def render_texts(self) -> dict[str, str]:
        if self.cot_stream is None:
            return {}
        return {"chain_of_thought": self.cot_stream.text()}

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
        state, rnn_state = self.encoder(
            data.s_seq,
            data.a_seq,
            data.r_seq,
            data.rnn_state,
            scalar_obs,
            data.cot_activations_seq,
        )

        action, actor_activation = self.policy_head.get_action(state)
        q_out = self.value_head(state, action)
        next_image_latent, next_reward_latent = self._empty_prediction(state)

        return InferResult(
            action=action,
            value_report=self.value_head.value_report(q_out.output),
            rnn_state=rnn_state,
            next_image_latent=next_image_latent,
            next_reward_latent=next_reward_latent,
            activations=ActivationFeatures(
                state=state,
                actor=actor_activation,
                critic=q_out.activation,
                state_predictor=next_image_latent,
            ),
            features=state,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        # Bootstrap value: Q(s', mu(s')) on the next-state window, no grad.
        with torch.inference_mode():
            next_state, _ = self.encoder(*self._window(data, self.horizon, None))
            next_action, _ = self.policy_head.get_action(next_state)
            next_output = self.value_head(next_state, next_action).output
        target_value = self.value_head.compute_target_value(
            next_output, data.rewards[:, -self.horizon :], data.dones[:, -self.horizon :]
        )

        curr_state, _ = self.encoder(*self._window(data, 0, -self.horizon))
        loss_result, _, _ = self._losses(curr_state, data.actions[:, -self.horizon :], target_value)
        return loss_result

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        """Combined inference and loss: the action the agent takes next comes from
        the same next-state window the TD target bootstraps on."""
        with torch.inference_mode():
            next_state, next_rnn_state = self.encoder(*self._window(data, self.horizon, None))
            next_action, actor_activation = self.policy_head.get_action(next_state)
            next_q_out = self.value_head(next_state, next_action)
        target_value = self.value_head.compute_target_value(
            next_q_out.output, data.rewards[:, -self.horizon :], data.dones[:, -self.horizon :]
        )

        curr_state, _ = self.encoder(*self._window(data, 0, -self.horizon))
        action_chunk = data.actions[:, -self.horizon :]
        loss_result, actor_loss, delta = self._losses(curr_state, action_chunk, target_value)

        # -Q(s, a) for the eligibility-trace critic step, detached from the encoder.
        et_critic_out = self.value_head(curr_state.detach(), action_chunk.detach())
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

    ####################
    # Internal methods #
    ####################

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
        """The ``(image, action, reward, rnn_state, scalar_obs, cot)`` the encoder
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
            data.cot_activations[:, start:stop],
        )

    def _losses(
        self,
        curr_state: torch.Tensor,
        action_chunk: torch.Tensor,
        target_value: torch.Tensor,
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
        total_loss = self.critic_loss_weight * critic_loss + actor_loss
        info_dict = {f"losses/{key}": value for key, value in {**critic_info, **actor_info}.items()}
        return LossResult(loss=total_loss, info=info_dict), actor_loss, critic_info["delta"]
