# SPDX-License-Identifier: MIT
"""Cosmos3-Edge av policy + injected value head, trained with DACER2.

Mirrors :class:`LiberoPi05Network`: the Cosmos3-Edge backbone / VAE are frozen,
only the action projection heads (the policy μ) and the injected action-value
critic learn. The action is the av embodiment's ego-pose chunk (raw dim 9); the
state scored by the critic is the pooled backbone understanding stream.

Rollout and training both sample through the same differentiable action denoiser
(:class:`CosmosEdgePolicy`), with more integration steps at rollout than in the
DACER2 actor unroll. The navigation-aware prompt is built by the environment and
arrives via ``obs["language"]``; the agent stores its token ids in the replay so
this network can rebuild the same conditioning when it re-encodes at loss time.
"""

import numpy as np
import torch
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_RAW_ACTION_DIM
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


def ego_history_av_deltas(poses: np.ndarray) -> np.ndarray:
    """Real av-convention pose deltas connecting consecutive ego poses.

    ``poses`` is ``(N, 3)`` of world ``(x, y, yaw_rad)`` (the recorded ego trajectory
    of the history clip). Returns ``(N - 1, 9)``: per transition, translation
    ``[x=right, y=down=0, z=forward]`` in the ego frame at the start pose, plus the
    ``rotation_6d`` (first two SO(3) columns) of the heading change toward the ego's
    right (identity == ``[1,0,0,0,1,0]``). These fix the history rows so the model
    generates only the future, grounded in the actual recent motion (not inferred)."""
    rows = []
    for a, b in zip(poses[:-1], poses[1:]):
        d = np.array([b[0] - a[0], b[1] - a[1]])
        fwd = np.array([np.cos(a[2]), np.sin(a[2])])
        right = np.array([-np.sin(a[2]), np.cos(a[2])])  # CARLA left-handed ego right
        fwd_b = np.array([np.cos(b[2]), np.sin(b[2])])
        theta = float(np.arctan2(float(right @ fwd_b), float(fwd @ fwd_b)))
        c, s = np.cos(theta), np.sin(theta)
        rows.append([float(d @ right), 0.0, float(d @ fwd), c, 0.0, -s, 0.0, 1.0, 0.0])
    return np.asarray(rows, dtype=np.float32)


class CosmosEdgeNetwork(NetworkInterface):
    def __init__(
        self,
        *,
        device: torch.device,
        value_head_factory,
        chunk_size: int,
        domain_name: str,
        resolution_tier: int,
        num_inference_steps: int,
        actor_denoising_steps: int,
        num_cond_latent_frames: int,
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
        self.actor_parameters = self.policy.freeze_backbone()

        self.chunk_size = int(chunk_size)
        self.domain_name = str(domain_name)
        self.resolution_tier = int(resolution_tier)
        self.num_inference_steps = int(num_inference_steps)
        # History-conditioned joint generation: the first ``num_cond_latent_frames``
        # latent frames are the clean history clip. VAE temporal factor is 4, so that
        # is the first ``history_len`` pixel frames; ``future_start`` is the action
        # row (== current-frame pixel index) where the *generated future* begins.
        self.num_cond_latent_frames = int(num_cond_latent_frames)
        self.history_len = (self.num_cond_latent_frames - 1) * 4 + 1
        self.future_start = (self.num_cond_latent_frames - 1) * 4
        pad = self.policy.text_tokenizer.pad_token_id
        self.pad_id = int(pad) if pad is not None else 0
        self.actor_denoising_steps = int(actor_denoising_steps)
        self.q_grad_eta = float(q_grad_eta)
        self.dacer_loss_weight = float(dacer_loss_weight)
        self.critic_loss_weight = float(critic_loss_weight)
        self.detach_critic = bool(detach_critic)

        self.action_dim = int(_EMBODIMENT_TO_RAW_ACTION_DIM[self.domain_name])
        self.state_dim = self.policy.pooled_state_dim

        # The critic scores the whole flattened trajectory as one action
        # (chunk_size * action_dim wide), with a single-step reward window
        # (horizon == 1), since the trajectory is re-planned every tick.
        self.critic: DistributionalValueHead = value_head_factory(
            self.state_dim, self.chunk_size * self.action_dim
        ).to(device)
        assert self.critic.horizon == 1, (
            f"critic horizon ({self.critic.horizon}) must be 1 (single-step reward "
            f"window); set `horizon: 1` in the agent config"
        )

        self._obs_schema: list | None = None
        self._obs_flat_dim: int | None = None

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
        # data.a_seq carries the real history action rows (from the agent's ego poses).
        enc = self._encode(data.s_seq, data.task_prompts[0], data.a_seq)
        action, vision_latents = self.policy.sample_action_chunk(enc, self.num_inference_steps)
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
        # seq_len = history_len + 1 consecutive stored frames. The current history clip
        # is the first ``history_len`` frames; the bootstrap (next) clip is shifted by
        # one. The transition's action / reward / done live in the last stored slot
        # (the action taken at the current frame, its reward, done).
        num = self.history_len
        cur_frames = torch.stack([self._unpack_obs(data.observations[0, i]) for i in range(num)])
        next_frames = torch.stack(
            [self._unpack_obs(data.observations[0, i]) for i in range(1, num + 1)]
        )
        cur_prompt = self._detokenize(data.task_prompt_token_ids[0, num - 1])
        next_prompt = self._detokenize(data.task_prompt_token_ids[0, num])
        # Real history action rows from the stored ego poses (obs_z = (x, y, yaw)).
        poses = data.obs_z[0].float().cpu().numpy()
        cur_hist = torch.from_numpy(ego_history_av_deltas(poses[:num])).to(self.device)
        next_hist = torch.from_numpy(ego_history_av_deltas(poses[1 : num + 1])).to(self.device)
        return ReplayBufferData(
            observations=[
                (cur_frames, cur_prompt, cur_hist),
                (next_frames, next_prompt, next_hist),
            ],
            actions=data.actions[:, -1:].contiguous(),
            rewards=data.rewards[:, -1:, 0],
            dones=data.dones[:, -1:, 0],
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
            next_action, _ = self.policy.sample_action_chunk(next_enc, self.actor_denoising_steps)
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
            next_action, _ = self.policy.sample_action_chunk(next_enc, self.actor_denoising_steps)
            next_output = self.critic(next_enc.state[None], next_action[None]).output
            exec_action, _ = self.policy.sample_action_chunk(next_enc, self.num_inference_steps)
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

    def _encode(self, frames: torch.Tensor, prompt: str, history_action: torch.Tensor):
        # ``frames`` is the history clip (T, C, H, W); the first num_cond_latent_frames
        # latent frames are anchored as history. ``history_action`` (future_start rows)
        # anchors the history action rows with the real ego-pose deltas, so only the
        # future rows are generated.
        pils = [self._to_pil(f) for f in frames]
        return self.policy.encode(
            frames=pils,
            prompt=prompt,
            chunk_size=self.chunk_size,
            domain_name=self.domain_name,
            resolution_tier=self.resolution_tier,
            num_inference_steps=self.num_inference_steps,
            generator=None,
            num_cond_latent_frames=self.num_cond_latent_frames,
            history_action=history_action,
        )

    def _flow_matching_velocity_loss(self, enc, target_action: torch.Tensor) -> torch.Tensor:
        """Regress the action velocity toward a Q-improved target (DACER2 score term)."""
        full_dim = enc.action_latents.shape[-1]
        target = F.pad(target_action, (0, full_dim - self.action_dim))
        noise = torch.randn_like(target)
        flow_time = float(torch.rand(()).item())
        x_t = flow_time * noise + (1.0 - flow_time) * target
        u_t = noise - target
        v_pred = self.policy.action_velocity(enc, x_t, flow_time)
        return F.mse_loss(v_pred[:, : self.action_dim], u_t[:, : self.action_dim])

    def _dacer2_actor_loss(self, enc) -> tuple[torch.Tensor, dict]:
        action_pi, _ = self.policy.sample_action_chunk(enc, self.actor_denoising_steps)
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
