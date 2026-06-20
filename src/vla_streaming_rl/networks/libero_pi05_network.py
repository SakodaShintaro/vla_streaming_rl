# SPDX-License-Identifier: MIT
"""The learnable network behind :class:`LiberoPi05Agent`.

Wraps LeRobot's pi0.5 (``PI05Policy``) flow-matching VLA together with its
pretrained pre/post-processor pipelines (which carry the LIBERO dataset
normalization statistics) in a single :class:`NetworkInterface`, and injects an
action-value critic so the frozen-backbone action expert can be fine-tuned by
RL — mirroring how ``SimLingoNetwork``/``ActorCriticWithActionValue`` bundle a
policy with a ``value_head`` critic. Like every other network it exposes exactly
the five ``NetworkInterface`` methods; everything else is a private helper.

The PaliGemma vision/LLM backbone is frozen; only the action expert and its
projections stay trainable (the policy μ), plus the injected critic. Learning
follows the project's flow-matching policy-RL recipe:

  * **Critic** — TD bootstrap on action chunks (``DistributionalValueHead``):
    ``target = Σ γ^i r_i + γ^H (1-done) Q(s', π(s'))``. The bootstrap action
    ``π(s')`` is sampled from pi0.5 itself (no grad).
  * **Actor** — DACER2 (arXiv:2505.23426), the flow-matching policy's RL loss,
    exactly as ``DiffusionPolicy.compute_actor_loss``: an advantage-maximizing
    term ``-E[Q(s, π(s))]`` (gradient flows through pi0.5's *differentiable*
    denoiser into the expert) plus a Q-gradient-guided flow-matching term that
    regresses the expert's velocity prediction toward a Q-improved action,
    keeping μ a valid flow while steering it toward higher Q.

``compute_loss`` returns the single combined ``critic + actor`` loss (the actor
loss freezes the critic during its Q forwards, so one backward keeps the two on
disjoint parameters); the agent steps a separate optimizer per param group, just
like ``SimLingoAgent._train_offpolicy``.

Everything the critic / actor losses touch lives in pi0.5's *normalized* action
space: the differentiable denoiser emits normalized actions, the executed chunk
is read normalized straight off the preprocessor (``batch[ACTION]``), and the
flow-matching target is built in the same space — so no (un)normalization is
needed at train time. The postprocessor is only used on the rollout path.

Because pi0.5's observation is a multimodal dict (two camera images + proprio +
tokenized language), it cannot ride the project's tensor ``ReplayBuffer``; the
agent keeps its own chunk buffer and passes preprocessed pi0.5 batches through
``infer`` (in ``InferInput.s_seq``, as ``SimLingoAgent`` passes its
``DrivingInput``) and ``compute_loss`` (in ``ReplayBufferData.observations`` as
a length-2 ``[cur, next]`` list of batches).
"""

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch.nn import functional as F

from vla_streaming_rl.networks.interface import (
    ActivationFeatures,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from vla_streaming_rl.networks.modules.value_head import DistributionalValueHead
from vla_streaming_rl.replay_buffer import ReplayBufferData

# LeRobot observation/action keys for the LIBERO pi0.5 checkpoint.
OBS_IMAGE_AGENTVIEW = "observation.images.image"
OBS_IMAGE_WRIST = "observation.images.image2"
OBS_STATE = "observation.state"
TASK_KEY = "task"
ACTION_KEY = "action"


class LiberoPi05Network(NetworkInterface):
    """pi0.5 policy (frozen backbone + trainable action expert) + Q critic.

    Exposed attributes used by ``LiberoPi05Agent``:
      - ``policy`` — the ``PI05Policy`` (pre/post-processors live on it too).
      - ``preprocessor`` / ``postprocessor`` — pipelines that normalize/tokenize
        a raw observation dict and un-normalize predicted actions.
      - ``critic`` — the injected ``DistributionalValueHead`` Q(s, a-chunk).
      - ``actor_parameters`` — the action-expert params (policy μ) the agent's
        actor optimizer owns; ``critic.parameters()`` is the critic optimizer's.
      - ``state_dim`` / ``action_dim`` / ``chunk_size`` — shapes the agent needs.

    The public surface is the five ``NetworkInterface`` methods; the pi0.5
    internals (prefix forward, differentiable denoiser, flow-matching loss) are
    private helpers.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        value_head_factory,
        actor_denoising_steps: int,
        q_grad_eta: float,
        dacer_loss_weight: float,
        critic_loss_weight: float,
        detach_critic: bool,
    ) -> None:
        super().__init__()
        checkpoint_repo = "lerobot/pi05_libero_finetuned_quantiles_v044"
        cfg = PreTrainedConfig.from_pretrained(checkpoint_repo)
        cfg.device = str(device)
        # Freeze the PaliGemma vision + LLM; keep only the action expert and its
        # projections trainable. PI05Pytorch reads these flags at construction.
        cfg.freeze_vision_encoder = True
        cfg.train_expert_only = True
        # torch.compile (max-autotune) is disabled: it traces a fixed shape and
        # breaks on the loss path. Gradient checkpointing is kept ON: even with a
        # frozen backbone, the training forward runs the full PaliGemma in the
        # same graph as the trainable action expert, so its activations are
        # retained for the expert's backward. Recomputing them (checkpointing)
        # rather than storing them is what makes the model's update fit in GPU
        # memory — and it matters even more here because the DACER2 actor unrolls
        # the expert ``actor_denoising_steps`` times with grad. (Requires the
        # policy be in train() mode at loss time — see LiberoPi05Agent._train.)
        cfg.compile_model = False
        cfg.gradient_checkpointing = True
        self.cfg = cfg
        self.device = device

        self.policy = PI05Policy.from_pretrained(checkpoint_repo, config=cfg).to(device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint_repo
        )

        self.state_dim = int(cfg.input_features[OBS_STATE].shape[0])
        self.action_dim = int(cfg.output_features[ACTION_KEY].shape[0])
        self.chunk_size = int(cfg.chunk_size)
        self.max_action_dim = int(cfg.max_action_dim)
        self.num_inference_steps = int(cfg.num_inference_steps)

        # DACER2 / TD knobs.
        self.actor_denoising_steps = int(actor_denoising_steps)
        self.q_grad_eta = float(q_grad_eta)
        self.dacer_loss_weight = float(dacer_loss_weight)
        self.critic_loss_weight = float(critic_loss_weight)
        self.detach_critic = bool(detach_critic)

        # ``train_expert_only`` already cleared requires_grad on the frozen
        # backbone; the remaining trainable params *are* the action expert (the
        # policy μ) the actor optimizer owns.
        self.actor_parameters = [p for p in self.policy.parameters() if p.requires_grad]

        # Inject the critic. The state it scores is the mean-pooled VLM prefix
        # (vision + language context) concatenated with the raw proprio — visual
        # context plus the kinematic state pi0.5 routes through its suffix. The
        # action is the full chunk (horizon == chunk_size, bound into the factory
        # by config), in pi0.5's normalized action space.
        prefix_dim = int(
            self.policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_size
        )
        self.prefix_dim = prefix_dim
        self.critic: DistributionalValueHead = value_head_factory(
            prefix_dim + self.state_dim, self.action_dim
        ).to(device)
        assert self.critic.horizon == self.chunk_size, (
            f"critic horizon ({self.critic.horizon}) must equal pi0.5 chunk_size "
            f"({self.chunk_size}); set `horizon: {self.chunk_size}` in the agent config"
        )

    # --- NetworkInterface contract -----------------------------------------

    def init_state(self) -> torch.Tensor:
        # pi0.5 inference is feed-forward (no recurrent state); the agent carries
        # this dummy only to satisfy the shared agent/network protocol.
        return torch.zeros(1)

    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
        # Language is tokenized inside the pi0.5 preprocessor, not here.
        del task_prompt
        return []

    @torch.no_grad()
    def infer(self, data: InferInput) -> InferResult:
        """Rollout inference: one prefix forward → action chunk + critic value.

        ``data.s_seq`` is a preprocessed pi0.5 batch (B == 1), mirroring how
        ``SimLingoAgent`` passes its ``DrivingInput`` through ``s_seq``. Sampling
        and scoring share the single (frozen) prefix forward, so the value report
        comes for free — no second backbone pass. The returned action chunk is in
        pi0.5's *normalized* space; the agent un-normalizes it with the
        postprocessor before stepping the env.
        """
        state, prefix_pad_masks, past_key_values = self._encode_state(data.s_seq)
        action = self._sample_action_chunk(
            prefix_pad_masks, past_key_values, self.num_inference_steps
        )
        critic_out = self.critic(state, action).output
        return InferResult(
            action=action,
            value_report=self.critic.value_report(critic_out),
            rnn_state=torch.zeros(1),
            next_image=np.zeros((1, 1, 3), dtype=np.uint8),
            next_reward=0.0,
            action_token_ids=[],
            activations=ActivationFeatures(
                state=state,
                actor=action.reshape(action.shape[0], -1),
                critic=critic_out,
                state_predictor=state,
            ),
            features=state,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        """TD-critic + DACER2-actor loss for one replay batch.

        pi0.5's multimodal observation does not fit the tensor ``ReplayBuffer``,
        so the agent overloads ``ReplayBufferData`` minimally: ``observations`` is
        a length-2 ``[cur_batch, next_batch]`` list of preprocessed pi0.5 batches,
        ``actions`` is the executed *normalized* chunk ``(B, H, action_dim)``, and
        ``rewards`` / ``dones`` are per-env-step ``(B, H)`` over the chunk.

        Returns the single combined loss (``critic_loss_weight·critic + actor``);
        the actor loss froze the critic during its Q forwards, so one backward
        leaves the two on disjoint params and the agent steps each optimizer.
        """
        cur_batch, next_batch = data.observations
        action_chunk = data.actions
        chunk_rewards = data.rewards
        chunk_dones = data.dones

        state, prefix_pad_masks, past_key_values = self._encode_state(cur_batch)

        # --- critic: TD bootstrap from the *next* state, no grad ------------
        with torch.no_grad():
            next_state, next_ppm, next_pkv = self._encode_state(next_batch)
            next_action = self._sample_action_chunk(next_ppm, next_pkv, self.actor_denoising_steps)
            next_output = self.critic(next_state, next_action).output
        target_value = self.critic.compute_target_value(next_output, chunk_rewards, chunk_dones)
        critic_loss, critic_info = self.critic.compute_critic_loss(
            state, action_chunk, target_value, self.detach_critic
        )

        # --- actor: DACER2 (advantage maximization + Q-guided flow) ---------
        actor_loss, actor_info = self._dacer2_actor_loss(state, prefix_pad_masks, past_key_values)

        info = {
            "losses/critic_loss": critic_info["critic_loss"],
            "losses/q_value": critic_info["curr_critic_value"],
            "losses/target_q": critic_info["target_value"],
            "losses/value_range": critic_info["value_range"],
            **actor_info,
        }
        return LossResult(loss=self.critic_loss_weight * critic_loss + actor_loss, info=info)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        # Streaming / eligibility-trace training is not supported for the pi0.5
        # agent (it trains off-policy on random replay batches). The contract
        # method exists so the network is a full NetworkInterface citizen.
        del data
        raise NotImplementedError(
            "LiberoPi05Network supports off-policy compute_loss only, not the "
            "streaming infer_and_compute_loss path."
        )

    # --- pi0.5 internals (private) -----------------------------------------

    def _encode_state(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, list]:
        model = self.policy.model
        images, img_masks = self.policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
            "eager"
        )
        (prefix_out, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        mask = prefix_pad_masks.unsqueeze(-1).to(prefix_out.dtype)
        pooled = (prefix_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        proprio = batch[OBS_STATE].to(self.device, torch.float32)
        state = torch.cat([pooled.to(torch.float32), proprio], dim=-1)
        return state, prefix_pad_masks, past_key_values

    def _sample_action_chunk(
        self, prefix_pad_masks: torch.Tensor, past_key_values: list, num_steps: int
    ) -> torch.Tensor:
        """Euler-integrate pi0.5's flow field to a normalized action chunk.

        Same integration as ``PI05Pytorch.sample_actions`` but **without** the
        ``@torch.no_grad`` decorator: when called under grad, the gradient flows
        through every ``denoise_step`` (the action expert) so the DACER2
        advantage term ``-Q(s, π(s))`` can update μ. The frozen prefix lives in
        ``past_key_values`` (detached), so only the expert receives gradient.
        Returns the unpadded chunk ``(B, chunk_size, action_dim)``, normalized.
        """
        model = self.policy.model
        bsize = prefix_pad_masks.shape[0]
        shape = (bsize, self.chunk_size, self.max_action_dim)
        x_t = model.sample_noise(shape, self.device)
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = 1.0 + step * dt
            time = torch.tensor(t, dtype=torch.float32, device=self.device).expand(bsize)
            v_t = model.denoise_step(prefix_pad_masks, past_key_values, x_t, time)
            x_t = x_t + dt * v_t
        return x_t[:, :, : self.action_dim].contiguous()

    def _flow_matching_velocity_loss(
        self, prefix_pad_masks: torch.Tensor, past_key_values: list, target_action: torch.Tensor
    ) -> torch.Tensor:
        """pi0.5's native flow-matching loss regressing μ toward ``target_action``.

        One differentiable expert forward (``denoise_step``) at a random flow
        time, in pi0.5's convention ``x_t = t·noise + (1-t)·a``, target velocity
        ``u_t = noise - a``. With ``target_action`` a Q-improved chunk this is
        DACER2's score-matching term in the policy's own parameterization: it
        pulls the flow toward higher-Q actions while keeping it a valid flow.
        ``target_action`` is normalized ``(B, chunk, action_dim)``.
        """
        model = self.policy.model
        bsize = prefix_pad_masks.shape[0]
        target = F.pad(target_action, (0, self.max_action_dim - self.action_dim))
        noise = model.sample_noise(target.shape, self.device)
        time = model.sample_time(bsize, self.device)
        t_exp = time[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * target
        u_t = noise - target
        v_pred = model.denoise_step(prefix_pad_masks, past_key_values, x_t, time)
        # Supervise only the real action dims (the rest is zero-pad).
        return F.mse_loss(v_pred[:, :, : self.action_dim], u_t[:, :, : self.action_dim])

    def _dacer2_actor_loss(
        self, state: torch.Tensor, prefix_pad_masks: torch.Tensor, past_key_values: list
    ) -> tuple[torch.Tensor, dict]:
        """DACER2 actor loss for the pi0.5 flow policy (mirrors DiffusionPolicy).

        (1) Advantage term ``-E[Q(s, π(s))]``: differentiable sampling pushes the
        gradient through the denoiser into the expert; the critic is frozen so
        only μ moves. (2) Q-gradient-guided flow matching: one ascent step on Q in
        action space forms a Q-improved target ``a*``, then μ's velocity is
        regressed onto the flow toward ``a*`` (pi0.5's own loss). ``a*`` is
        detached, keeping the critic out of the actor graph.
        """
        action_pi = self._sample_action_chunk(
            prefix_pad_masks, past_key_values, self.actor_denoising_steps
        )
        for p in self.critic.parameters():
            p.requires_grad_(False)
        advantage = self.critic.to_value(
            self.critic.get_advantage(state.detach(), action_pi).output
        ).view(-1)
        actor_adv_loss = -advantage.mean()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        a = action_pi.detach().requires_grad_(True)
        q = self.critic.to_value(self.critic(state.detach(), a).output)
        (q_grad,) = torch.autograd.grad(q.sum(), a)
        a_star = (a.detach() + self.q_grad_eta * q_grad).clamp(-1.0, 1.0)
        flow_loss = self._flow_matching_velocity_loss(prefix_pad_masks, past_key_values, a_star)

        actor_loss = actor_adv_loss + self.dacer_loss_weight * flow_loss
        info = {
            "losses/actor_adv_loss": float(actor_adv_loss.item()),
            "losses/flow_loss": float(flow_loss.item()),
            "losses/advantage": float(advantage.mean().item()),
        }
        return actor_loss, info
