# SPDX-License-Identifier: MIT
"""Cosmos3-Edge backbone + new ``navigation`` action domain, trained with DACER2.

The ``navigation`` embodiment is a NEW domain occupying a free slot (id 31) of
the transformer's 32-slot ``DomainAwareLinear`` action projections. Its action
row is the direct 2D env control ``[steer, gas_or_brake]`` at the model frame
rate (``action_fps``), so the same action space can later serve other envs
(e.g. Animal-AI at a different ``frame_stride``). Only the new slot's
projection rows (+ the shared action modality embed) learn; the trunk
(und/gen streams), VAE and text stay frozen — the bet is that the frozen trunk
already computes future ego-motion in the action-token positions and a linear
readout per domain suffices (that is exactly how the ~20 pretrained
embodiments are wired).

Chunk semantics — one config value ``chunk_size`` (== critic horizon):
  * The joint sequence carries ``future_start + chunk_size`` action rows: the
    ``future_start`` history rows are anchored to the really-executed controls,
    the ``chunk_size`` future rows are generated.
  * The agent executes the future rows open-loop (one row per model frame,
    held for ``frame_stride`` env ticks), then replans. The replay stores one
    slot per model frame, so a sampled window of ``history_len + chunk_size``
    slots holds the conditioning clip, the executed rows and the
    ``chunk_size``-step reward window the critic bootstraps over
    (``libero_pi05`` pattern).
"""

import numpy as np
import torch
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
    _EMBODIMENT_TO_DOMAIN_ID,
    _EMBODIMENT_TO_RAW_ACTION_DIM,
)
from PIL import Image
from torch.nn import functional as F

from vla_streaming_rl.cosmos3.policy import CosmosEdgePolicy
from vla_streaming_rl.networks.interface import (
    ActivationFeatures,
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from vla_streaming_rl.networks.modules.value_head import DistributionalValueHead
from vla_streaming_rl.replay_buffer import ReplayBufferData

REAL_REPO = "nvidia/Cosmos3-Edge"
MAX_PROMPT_TOKENS = 64

# New embodiment domain: 2D direct control [steer, gas_or_brake]. Slot 31 is
# unused by the pretrained checkpoint (registered ids stop at 20, the
# projection banks hold 32 slots); registering here makes the diffusers
# pipeline accept the domain name end-to-end.
NAVIGATION_DOMAIN_NAME = "navigation"
NAVIGATION_DOMAIN_ID = 31
NAVIGATION_ACTION_DIM = 2

assert all(
    domain_id != NAVIGATION_DOMAIN_ID or name == NAVIGATION_DOMAIN_NAME
    for name, domain_id in _EMBODIMENT_TO_DOMAIN_ID.items()
), f"Cosmos3 domain id {NAVIGATION_DOMAIN_ID} is already taken"
_EMBODIMENT_TO_DOMAIN_ID[NAVIGATION_DOMAIN_NAME] = NAVIGATION_DOMAIN_ID
_EMBODIMENT_TO_RAW_ACTION_DIM[NAVIGATION_DOMAIN_NAME] = NAVIGATION_ACTION_DIM


class CosmosEdgeNetwork(NetworkInterface):
    def __init__(
        self,
        *,
        device: torch.device,
        value_head_factory,
        chunk_size: int,
        resolution_tier: int,
        num_inference_steps: int,
        actor_denoising_steps: int,
        num_cond_latent_frames: int,
        action_fps: float,
        frame_stride: int,
        q_grad_eta: float,
        dacer_loss_weight: float,
        critic_loss_weight: float,
        detach_critic: bool,
    ) -> None:
        super().__init__()
        self.device = device
        self.policy = CosmosEdgePolicy.from_pretrained(
            REAL_REPO, torch_dtype=torch.bfloat16, enable_safety_checker=False
        ).to(device)
        # WanVAE is unstable in bf16 (trained without amp); keep it in fp32.
        self.policy.vae.to(torch.float32)

        # ``chunk_size`` is the number of *generated future* rows the agent
        # executes == the critic reward window (horizon). The model's joint
        # sequence additionally carries the ``future_start`` anchored history
        # rows; the VAE temporal factor 4 requires the total row count to be a
        # multiple of 4 (num_frames = rows + 1 -> integer latent frames).
        self.chunk_size = int(chunk_size)
        self.num_cond_latent_frames = int(num_cond_latent_frames)
        self.history_len = (self.num_cond_latent_frames - 1) * 4 + 1
        self.future_start = (self.num_cond_latent_frames - 1) * 4
        self.model_chunk = self.future_start + self.chunk_size
        assert self.model_chunk % 4 == 0, (
            f"future_start ({self.future_start}) + chunk_size ({self.chunk_size}) must be a "
            f"multiple of 4 (VAE temporal factor)"
        )
        # Model frames are ``action_fps`` Hz while the env ticks faster; one row
        # spans ``frame_stride`` env ticks (agent-side cadence).
        self.action_fps = float(action_fps)
        self.frame_stride = int(frame_stride)
        self.policy.setup(
            chunk_size=self.model_chunk,
            domain_name=NAVIGATION_DOMAIN_NAME,
            resolution_tier=int(resolution_tier),
            num_inference_steps=int(num_inference_steps),
            num_cond_latent_frames=self.num_cond_latent_frames,
            fps=self.action_fps,
        )
        # Fresh projection rows for the new domain — the checkpoint values of an
        # unused slot are arbitrary pretraining leftovers.
        transformer = self.policy.transformer
        for proj in (transformer.action_proj_in, transformer.action_proj_out):
            torch.nn.init.normal_(proj.fc.weight.data[NAVIGATION_DOMAIN_ID], std=0.02)
            torch.nn.init.zeros_(proj.bias.weight.data[NAVIGATION_DOMAIN_ID])
        # NOTE: the DomainAwareLinear banks are single embedding tensors, so the
        # optimizer sees every domain's rows even though only the navigation
        # rows receive gradients (embedding lookup). Keep the actor optimizer's
        # weight_decay at 0 or the other domains' pretrained rows decay.
        self.actor_parameters = self.policy.freeze_backbone()

        pad = self.policy.text_tokenizer.pad_token_id
        self.pad_id = int(pad) if pad is not None else 0
        self.num_inference_steps = int(num_inference_steps)
        self.actor_denoising_steps = int(actor_denoising_steps)
        self.q_grad_eta = float(q_grad_eta)
        self.dacer_loss_weight = float(dacer_loss_weight)
        self.critic_loss_weight = float(critic_loss_weight)
        self.detach_critic = bool(detach_critic)

        self.action_dim = NAVIGATION_ACTION_DIM
        self.state_dim = self.policy.pooled_state_dim

        # The critic scores the executed future rows (chunk_size x action_dim,
        # flattened inside the head) against the chunk_size-step reward window.
        self.critic: DistributionalValueHead = value_head_factory(
            self.state_dim, self.action_dim
        ).to(device)
        assert self.critic.horizon == self.chunk_size, (
            f"critic horizon ({self.critic.horizon}) must equal chunk_size "
            f"({self.chunk_size}); set `horizon: ${{chunk_size}}` in the agent config"
        )

        self._obs_schema: list | None = None
        self._obs_flat_dim: int | None = None

    @property
    def seq_len(self) -> int:
        """Replay slots (one per model frame) a sampled window must span: the
        conditioning clip plus the executed chunk / reward window."""
        return self.history_len + self.chunk_size

    # --- NetworkInterface contract -----------------------------------------

    def init_state(self) -> torch.Tensor:
        return torch.zeros(1)

    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
        return self.policy.text_tokenizer.encode(task_prompt, add_special_tokens=False)[
            :MAX_PROMPT_TOKENS
        ]

    def _detokenize(self, token_ids: torch.Tensor) -> str:
        ids = [int(t) for t in token_ids.tolist() if int(t) != self.pad_id]
        return self.policy.text_tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def infer(self, data: InferInput) -> InferResult:
        # data.a_seq carries the executed history control rows (future_start, 2).
        enc = self._encode(data.s_seq, data.task_prompts[0], data.a_seq)
        full, vision_latents = self.policy.sample_action_chunk(enc, self.num_inference_steps)
        action = full[self.future_start :].float()  # (chunk_size, 2) generated future rows
        critic_out = self.critic(enc.state[None], action[None]).output
        return InferResult(
            action=action[None],
            value_report=self.critic.value_report(critic_out),
            rnn_state=torch.zeros(1),
            # The world model's farthest future-frame prediction (last decoded frame).
            next_image=self.policy.decode_vision_latents(vision_latents, -1),
            next_reward=0.0,
            activations=ActivationFeatures(
                state=enc.state[None],
                actor=action.reshape(1, -1),
                critic=critic_out,
                state_predictor=enc.state[None],
            ),
            features=enc.state[None],
        )

    def pack_obs(self, frame: torch.Tensor) -> torch.Tensor:
        """Flatten one resized RGB frame (C, H, W) into a float vector for the buffer.
        The navigation prompt travels separately in the buffer's token-id slot."""
        if self._obs_schema is None:
            self._obs_schema = (tuple(frame.shape), frame.dtype)
            self._obs_flat_dim = int(np.prod(frame.shape))
        return frame.reshape(-1).to(torch.float32)

    @property
    def obs_flat_dim(self) -> int:
        return self._obs_flat_dim

    def _unpack_obs(self, flat: torch.Tensor) -> torch.Tensor:
        shape, dtype = self._obs_schema
        return flat.reshape(*shape).to(dtype)

    def _unpack_window(self, data: ReplayBufferData) -> ReplayBufferData:
        # seq_len = history_len + chunk_size stored model frames. Slot t holds
        # (frame_t, row executed during the previous model frame, reward of that
        # row, done). The current clip is slots [0, history_len); its history
        # rows are the executed rows at slots [1, history_len) (the transitions
        # connecting the clip frames). The executed chunk / rewards / dones live
        # in slots [history_len, seq_len); the bootstrap (next) clip is the last
        # history_len slots.
        num = self.history_len
        cur_frames = torch.stack([self._unpack_obs(data.observations[0, i]) for i in range(num)])
        next_frames = torch.stack(
            [
                self._unpack_obs(data.observations[0, i])
                for i in range(self.chunk_size, self.chunk_size + num)
            ]
        )
        cur_prompt = self._detokenize(data.task_prompt_token_ids[0, num - 1])
        next_prompt = self._detokenize(data.task_prompt_token_ids[0, -1])
        rows = data.actions[0].float().to(self.device)  # (seq_len, 2)
        cur_hist = rows[1:num]
        next_hist = rows[self.chunk_size + 1 : self.chunk_size + num]
        return ReplayBufferData(
            observations=[
                (cur_frames, cur_prompt, cur_hist),
                (next_frames, next_prompt, next_hist),
            ],
            actions=data.actions[:, num:].contiguous(),  # (B, chunk_size, 2) executed rows
            rewards=data.rewards[:, num:, 0],
            dones=data.dones[:, num:, 0],
            obs_z=data.obs_z,
            rnn_state=data.rnn_state,
            task_prompt_token_ids=data.task_prompt_token_ids,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        data = self._unpack_window(data)
        cur_obs, next_obs = data.observations
        action_chunk = data.actions
        chunk_rewards = data.rewards
        chunk_dones = data.dones

        enc = self._encode(*cur_obs)
        with torch.no_grad():
            next_enc = self._encode(*next_obs)
            next_full, _ = self.policy.sample_action_chunk(next_enc, self.actor_denoising_steps)
            next_action = next_full[self.future_start :].float()
            next_output = self.critic(next_enc.state[None], next_action[None]).output
        target_value = self.critic.compute_target_value(next_output, chunk_rewards, chunk_dones)
        critic_loss, critic_info = self.critic.compute_critic_loss(
            enc.state[None], action_chunk, target_value, self.detach_critic
        )
        actor_loss, actor_info = self._dacer2_actor_loss(enc)

        info = {
            "losses/critic_loss": critic_info["critic_loss"],
            "losses/q_value": critic_info["curr_critic_value"],
            "losses/target_q": critic_info["target_value"],
            "losses/value_range": critic_info["value_range"],
            **actor_info,
        }
        return LossResult(loss=self.critic_loss_weight * critic_loss + actor_loss, info=info)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        data = self._unpack_window(data)
        cur_obs, next_obs = data.observations
        action_chunk = data.actions
        chunk_rewards = data.rewards
        chunk_dones = data.dones

        enc = self._encode(*cur_obs)
        with torch.no_grad():
            next_enc = self._encode(*next_obs)
            next_full, _ = self.policy.sample_action_chunk(next_enc, self.actor_denoising_steps)
            next_action = next_full[self.future_start :].float()
            next_output = self.critic(next_enc.state[None], next_action[None]).output
            exec_full, _ = self.policy.sample_action_chunk(next_enc, self.num_inference_steps)
            exec_action = exec_full[self.future_start :].float()
        target_value = self.critic.compute_target_value(next_output, chunk_rewards, chunk_dones)
        critic_loss, critic_info = self.critic.compute_critic_loss(
            enc.state[None], action_chunk, target_value, self.detach_critic
        )
        actor_loss, actor_info = self._dacer2_actor_loss(enc)

        neg_value = (
            -self.critic.to_value(self.critic(enc.state[None], action_chunk).output).view(-1).mean()
        )
        et_info = EligibilityTraceInfo(
            actor_entropy_loss=actor_loss, neg_value=neg_value, delta=critic_info["delta"]
        )
        info = {
            "losses/critic_loss": critic_info["critic_loss"],
            "losses/q_value": critic_info["curr_critic_value"],
            "losses/target_q": critic_info["target_value"],
            "losses/value_range": critic_info["value_range"],
            "losses/delta": critic_info["delta"],
            **actor_info,
        }
        loss_result = LossResult(loss=self.critic_loss_weight * critic_loss + actor_loss, info=info)
        with torch.no_grad():
            value_report = self.critic.value_report(
                self.critic(next_enc.state[None], exec_action[None]).output
            )
        infer_result = InferResult(
            action=exec_action[None],
            value_report=value_report,
            rnn_state=torch.zeros(1),
            next_image=np.zeros((1, 1, 3), dtype=np.uint8),
            next_reward=0.0,
            features=enc.state[None],
            activations=ActivationFeatures(
                state=enc.state[None],
                actor=action_chunk.reshape(1, -1),
                critic=enc.state[None],
                state_predictor=enc.state[None],
            ),
        )
        return InferLossResult(infer_result=infer_result, loss_result=loss_result, et_info=et_info)

    # --- Cosmos internals (private) ----------------------------------------

    def _to_pil(self, frame: torch.Tensor) -> Image.Image:
        arr = (frame.detach().float().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        return Image.fromarray(np.transpose(arr, (1, 2, 0)))

    def _encode(self, frames: torch.Tensor, prompt: str, history_rows: torch.Tensor):
        # ``frames`` is the history clip (T, C, H, W); the first num_cond_latent_frames
        # latent frames are anchored as history. ``history_rows`` (future_start, 2)
        # anchors the history action rows with the really-executed controls, so
        # only the future rows are generated.
        pils = [self._to_pil(f) for f in frames]
        return self.policy.encode(pils, prompt, history_rows)

    def _flow_matching_velocity_loss(self, enc, target_future: torch.Tensor) -> torch.Tensor:
        """Regress the future-row action velocity toward a Q-improved target
        (DACER2 score term). History rows stay clean (they are conditioning);
        the loss reads only the generated future rows and raw action dims."""
        target = enc.x0_action.clone()  # (model_chunk, action_dim): real history rows, 0 elsewhere
        target[self.future_start :, : self.action_dim] = target_future.to(target)
        noise = torch.randn_like(target)
        noise[:, self.action_dim :] = 0.0
        noise[: self.future_start] = 0.0
        flow_time = float(torch.rand(()).item())
        x_t = flow_time * noise + (1.0 - flow_time) * target
        u_t = noise - target
        v_pred = self.policy.action_velocity(enc, x_t, flow_time)
        return F.mse_loss(
            v_pred[self.future_start :, : self.action_dim],
            u_t[self.future_start :, : self.action_dim],
        )

    def _dacer2_actor_loss(self, enc) -> tuple[torch.Tensor, dict]:
        full, _ = self.policy.sample_action_chunk(enc, self.actor_denoising_steps)
        action_pi = full[self.future_start :].float()
        state = enc.state.detach()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        advantage = self.critic.to_value(
            self.critic.get_advantage(state[None], action_pi[None]).output
        ).view(-1)
        actor_adv_loss = -advantage.mean()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        a = action_pi.detach().requires_grad_(True)
        q = self.critic.to_value(self.critic(state[None], a[None]).output)
        (q_grad,) = torch.autograd.grad(q.sum(), a)
        a_star = (a.detach() + self.q_grad_eta * q_grad).detach()
        flow_loss = self._flow_matching_velocity_loss(enc, a_star)

        actor_loss = actor_adv_loss + self.dacer_loss_weight * flow_loss
        info = {
            "losses/actor_adv_loss": float(actor_adv_loss.item()),
            "losses/flow_loss": float(flow_loss.item()),
            "losses/advantage": float(advantage.mean().item()),
        }
        return actor_loss, info
