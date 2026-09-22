# SPDX-License-Identifier: MIT
from collections.abc import Callable

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
from vla_streaming_rl.networks.modules.backbone import SpatialTemporalEncoder
from vla_streaming_rl.networks.modules.cot_batch import CoTBatch
from vla_streaming_rl.networks.modules.cot_stream import CoTStream
from vla_streaming_rl.networks.modules.image_processor import ImageProcessor
from vla_streaming_rl.networks.modules.policy_head import build_policy_head
from vla_streaming_rl.networks.modules.prediction_head import StatePredictionHead
from vla_streaming_rl.networks.modules.reward_processor import RewardProcessor
from vla_streaming_rl.networks.modules.value_head import DistributionalValueHead
from vla_streaming_rl.replay_buffer import ReplayBufferData
from vla_streaming_rl.reward_processor import RunningNormalizer
from vla_streaming_rl.utils import render_conversation_panel


def build_cot(
    mode: str,
    model_id: str,
    load_in_4bit: bool,
    tokens_per_step: int,
    max_len: int,
    temperature: float,
    steps_per_chain: int,
    use_cuda_graph: bool,
    prompt_budget: int,
    prompt_builder,
    device: torch.device,
):
    """The chain generator named by `mode`, both of which advance() the same way.

    "stream" keeps one chain mid-thought and issues `tokens_per_step` of it per
    environment step; "batch" writes a whole chain every `steps_per_chain` steps
    and holds it in between. Every mode's parameters are always supplied; a mode
    ignores the ones that do not apply to it.
    """
    builders = {
        "stream": lambda: CoTStream(
            model_id=model_id,
            load_in_4bit=load_in_4bit,
            tokens_per_step=tokens_per_step,
            max_len=max_len,
            temperature=temperature,
            use_cuda_graph=use_cuda_graph,
            prompt_builder=prompt_builder,
            device=device,
        ),
        "batch": lambda: CoTBatch(
            model_id=model_id,
            load_in_4bit=load_in_4bit,
            tokens_per_step=tokens_per_step,
            max_len=max_len,
            temperature=temperature,
            steps_per_chain=steps_per_chain,
            use_cuda_graph=use_cuda_graph,
            prompt_budget=prompt_budget,
            prompt_builder=prompt_builder,
            device=device,
        ),
    }
    assert mode in builders, f"unknown cot_mode {mode!r}; expected one of {sorted(builders)}"
    return builders[mode]()


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
        image_encoder_type: str,
        image_encoder_output_dim: int,
        vlm_model_id: str,
        vlm_load_in_4bit: bool,
        cot_tokens_num: int,
        max_new_tokens: int,
        temperature: float,
        cot_mode: str,
        cot_steps_per_chain: int,
        cot_dropout: float,
        token_dropout: float,
        cot_pool: str,
        cot_cuda_graph: bool,
        cot_prompt_budget: int,
        prompt_builder,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.sparsity = sparsity
        self.seq_len = seq_len
        self.critic_loss_weight = critic_loss_weight

        self.action_dim = action_space_shape[0]
        self.predictor_step_num = predictor_step_num
        self.observation_space_shape = observation_space_shape

        self.image_processor = ImageProcessor(observation_space_shape, image_encoder_type)
        hidden_image_dim = image_encoder_output_dim
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        assert 0.0 <= cot_dropout < 1.0, cot_dropout
        self.cot_dropout = cot_dropout
        assert 0.0 <= token_dropout < 1.0, token_dropout
        self.token_dropout = token_dropout
        self.scalar_obs_dim = 9
        self.scalar_obs_normalizer = RunningNormalizer(self.scalar_obs_dim)
        # ``cot_tokens_num = 0`` is the ablation: the same body, the same heads
        # and the same loss with the chain's tokens taken out of the space axis,
        # which is what isolates what the chain contributes. No chain means no
        # VLM to load, so the width comes from the config, not a loaded model.
        text_config = AutoConfig.from_pretrained(vlm_model_id).text_config
        cot_dim = text_config.hidden_size
        # The embedding plus every layer's output.
        cot_layers = text_config.num_hidden_layers + 1
        self.pool_cot = cot_pool == "mean" and cot_tokens_num > 0
        self.cot_shape = (1 if self.pool_cot else cot_tokens_num, cot_layers, cot_dim)
        # Not a submodule: the frozen VLM must stay out of parameters()/state_dict().
        self.cot_module = None
        if cot_tokens_num > 0:
            self.cot_module = build_cot(
                mode=cot_mode,
                model_id=vlm_model_id,
                load_in_4bit=vlm_load_in_4bit,
                tokens_per_step=cot_tokens_num,
                max_len=max_new_tokens,
                temperature=temperature,
                steps_per_chain=cot_steps_per_chain,
                use_cuda_graph=cot_cuda_graph,
                prompt_budget=cot_prompt_budget,
                prompt_builder=prompt_builder,
                device=torch.device("cuda"),
            )

        self.encoder = SpatialTemporalEncoder(
            image_features_shape=tuple(self.image_processor.output_shape),
            image_latent_dim=hidden_image_dim,
            reward_processor=self.reward_processor,
            seq_len=self.seq_len,
            n_layer=encoder_block_num,
            action_dim=self.action_dim,
            scalar_obs_dim=self.scalar_obs_dim,
            temporal_model_type=temporal_model_type,
            cot_tokens_num=cot_tokens_num,
            cot_layers=cot_layers,
            cot_dim=cot_dim,
            cot_pool=cot_pool,
            cot_steps_per_chain=cot_steps_per_chain,
            layer_scale_init=layer_scale_init,
        )

        self.horizon = horizon
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
        self.prediction_head = StatePredictionHead(
            image_latent_shape=(hidden_image_dim, self.encoder.hidden_h, self.encoder.hidden_w),
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

    # Fixed so the render strip keeps one shape for the whole run; wide enough
    # to read a chain of ``max_new_tokens`` tokens.
    # Wide and tall enough for several turns of the conversation at once: the
    # panel is the only place a run shows what the chain was actually asked.
    COT_PANEL_WIDTH = 680
    COT_PANEL_HEIGHT = 560

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_state()

    def stored_image_shape(self) -> tuple[int, ...]:
        """The frozen encoder's output for the frame, not the frame."""
        return tuple(self.image_processor.output_shape)

    def to_stored_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_processor.encode(image.unsqueeze(0)).squeeze(0)

    def advance_cot(
        self, episode_started: bool, episode_done: bool, window: ReplayBufferData
    ) -> tuple[torch.Tensor, int]:
        """This step's chain-of-thought activations and how many steps ago they
        were generated, or nothing when the chain is off. The first tick of an
        episode ends whatever chain was running, so an episode's commentary
        starts on its own first frame rather than carrying the one written about
        the frame the last episode ended on, and its last tick is what the chain
        judges the subtask it cut short on. The chain reads the conversation the
        builder holds, not ``window``."""
        del window
        if self.cot_module is None:
            return torch.zeros(self.cot_shape), 0
        if episode_started:
            self.cot_module.reset()
        # Advanced first: `age` is about the chain the call hands back, which is
        # a fresh one on the steps that write.
        activations = self.cot_module.advance(episode_done)
        if self.pool_cot:
            activations = activations.float().mean(dim=0, keepdim=True).to(activations.dtype)
        return activations, self.cot_module.age()

    def render_panels(self) -> dict[str, np.ndarray]:
        """The conversation as it currently stands, drawn for the render strip:
        the turn the agent was shown this tick and the chains written about the
        ones before it, under what the last run of the VLM cost. Without a chain
        there is no panel at all rather than a blank one, which keeps that run's
        strip the width of what it has."""
        if self.cot_module is None:
            return {}
        stats = self.cot_module.stats()
        status = (
            f"in {stats['input_tokens']} tok   out {stats['output_tokens']} tok   "
            f"{stats['msec']:.0f} ms"
        )
        return {
            "conversation": render_conversation_panel(
                self.cot_module.exchange(),
                status,
                self.COT_PANEL_WIDTH,
                self.COT_PANEL_HEIGHT,
            )
        }

    def render_texts(self) -> dict[str, str]:
        if self.cot_module is None:
            return {}
        return {"chain_of_thought": self.cot_module.text()}

    def thought_text(self) -> str:
        if self.cot_module is None:
            return ""
        return self.cot_module.text()

    def _cot_keep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Which sequences of a batch keep their chain, the rest being the share
        ``cot_dropout`` it is taken away from. Handed to the encoder as a mask
        rather than applied to the activations here, which would copy them.

        Only on the learning path. Without it the critic has never scored a
        state whose chain is missing, so asking it what an action is worth
        without one is asking about an input it was never trained on -- which is
        exactly what the chain's own contribution has to be measured against.
        Dropping whole sequences rather than single steps keeps a sampled window
        internally consistent.
        """
        if self.cot_dropout == 0.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return torch.rand(batch_size, device=device) >= self.cot_dropout

    def _token_keep(self, batch_size: int, steps: int, device: torch.device) -> torch.Tensor:
        """Which tokens of a batch's windows the encoder gets to read, the rest
        being the share ``token_dropout`` takes away: every token alike, whatever
        it carries, so no one kind of input can be leaned on alone. Only on the
        learning path, as the chain's own dropout is."""
        shape = (batch_size, steps, self.encoder.space_len)
        return torch.rand(shape, device=device) >= self.token_dropout

    def _window(self, data: ReplayBufferData, start, stop) -> tuple:
        """The ``(image, action, reward, rnn_state, scalar_obs, cot, cot_age,
        cot_keep, token_keep)`` the encoder reads, sliced out of a replay batch
        over ``[start, stop)`` steps."""
        observations = data.observations[:, start:stop]
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
            data.cot_activations[:, start:stop],
            data.cot_age[:, start:stop],
            self._cot_keep(data.cot_activations.shape[0], data.cot_activations.device),
            self._token_keep(observations.shape[0], observations.shape[1], observations.device),
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
            data.cot_activations_seq,
            data.cot_age_seq,
            torch.ones(1, dtype=torch.bool, device=data.cot_activations_seq.device),
            torch.ones(
                (1, data.s_seq.shape[1], self.encoder.space_len),
                dtype=torch.bool,
                device=data.s_seq.device,
            ),
        )  # (B, state_dim)

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
            curr_state, action_chunk, target_value, self.detach_critic
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            curr_state,
            action_chunk,
            value_head=self.value_head,
            detach_actor=self.detach_actor,
        )
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
            next_state, next_rnn_state = self.encoder(*self._window(data, self.horizon, None))
            next_action, actor_activation = self.policy_head.get_action(next_state)
            next_q_out = self.value_head(next_state, next_action)
            critic_activation = next_q_out.activation
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self.value_head.compute_target_value(
            next_q_out.output, chunk_rewards, chunk_dones
        )

        prev_state, _ = self.encoder(*self._window(data, 0, -self.horizon))

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
