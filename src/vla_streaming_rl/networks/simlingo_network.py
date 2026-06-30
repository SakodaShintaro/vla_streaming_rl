# SPDX-License-Identifier: MIT
"""The learnable network behind :class:`SimLingoAgent`.

Bundles the (frozen) SimLingo VLM driving model, the trainable waypoint heads
(the deterministic policy μ), and the Q critic into a single ``nn.Module`` so
the agent receives it from outside via ``build_network`` — exactly like every
other agent — instead of constructing the model itself. The tokenizer / config
byproducts the agent's inference pipeline needs are exposed as attributes.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from torch import nn
from transformers import Qwen2Tokenizer

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
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.simlingo.models.driving import DrivingModel
from vla_streaming_rl.simlingo.models.encoder.internvl2_vendored.configuration_internvl_chat import (
    InternVLChatConfig,
)
from vla_streaming_rl.simlingo.utils.internvl2_utils import (
    SIMLINGO_ADDITIONAL_SPECIAL_TOKENS,
)

# SimLingo's driving adaptor emits route waypoints + speed waypoints, each 2-D —
# see DrivingAdaptor in simlingo_training/models/adaptors.py.
ROUTE_WPS_LEN = 20
SPEED_WPS_LEN = 10
WP_DIM = 2
# Number of waypoint-query positions in ``driving_features`` (route then speed).
# The DrivingAdaptor heads consume one feature vector per query.
NUM_WP_QUERIES = ROUTE_WPS_LEN + SPEED_WPS_LEN
ACTION_DIM = NUM_WP_QUERIES * WP_DIM


class SimLingoNetwork(NetworkInterface):
    """SimLingo VLM + waypoint heads (policy μ) + Q critic.

    ``infer`` runs the (frozen) VLM forward on the live ``DrivingInput`` and
    returns the action plus the per-query ``features`` the agent caches into the
    replay buffer (so training never re-runs the VLM). The whole VLM is frozen
    except the two waypoint heads (``actor_heads``); those and the ``critic`` are
    the only trainable parameters.

    Exposed attributes used by ``SimLingoAgent``:
      - ``actor_heads`` / ``critic`` / ``driving_adaptor`` — for training.
      - ``feature_dim`` — shape.
      - ``tokenizer`` / ``num_image_token`` / ``cfg`` — to build the agent's
        prompt / inference pipeline.
    """

    def __init__(
        self,
        *,
        value_head_factory: Callable[[int, int], DistributionalValueHead],
        actor_loss_type: str,
        awr_num_samples: int,
        awr_temperature: float,
        awr_sample_noise: float,
    ) -> None:
        super().__init__()

        # Which actor loss trains the waypoint heads: ``ddpg`` (deterministic
        # policy gradient, −Q(s, μ(s))) or ``awr`` (advantage-weighted
        # regression). The agent only asks the network for ``actor_loss`` and
        # does not branch on this.
        self.actor_loss_type = actor_loss_type

        # Advantage-weighted-regression actor knobs (used by ``awr_actor_loss``):
        # how many actions to sample per state, the softmax temperature on the
        # per-state-normalized Q advantages, and the Gaussian std of the
        # candidate actions around μ(s). Unused by the off_policy / streaming
        # actor losses.
        self.awr_num_samples = int(awr_num_samples)
        self.awr_temperature = float(awr_temperature)
        self.awr_sample_noise = float(awr_sample_noise)

        HF_REPO_ID = "RenzKa/simlingo"
        HF_CKPT_NAME = "pytorch_model.pt"
        snapshot = Path(snapshot_download(HF_REPO_ID))
        candidates = [p for p in snapshot.rglob(HF_CKPT_NAME) if "/blobs/" not in str(p)]
        assert candidates, f"No {HF_CKPT_NAME} found in HF snapshot of {HF_REPO_ID} at {snapshot}"
        config_path = str(candidates[0])

        # load config from the checkpoint's .hydra folder
        config_load_path = Path(config_path).parent.parent.parent / ".hydra" / "config.yaml"
        with open(config_load_path, "r") as file:
            cfg = OmegaConf.load(file)
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        self.cfg = cfg

        # tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(cfg.model.vision_model.variant)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SIMLINGO_ADDITIONAL_SPECIAL_TOKENS)}
        )
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        # ``AutoConfig`` + ``trust_remote_code=True`` resolves to
        # ``InternVLChatConfig``; use the vendored class directly.
        tmp_config = InternVLChatConfig.from_pretrained(cfg.model.vision_model.variant)
        image_size = tmp_config.force_image_size or tmp_config.vision_config.image_size
        patch_size = tmp_config.vision_config.patch_size
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (tmp_config.downsample_ratio**2)
        )
        cache_dir = f"pretrained/{(cfg.model.vision_model.variant.split('/')[1])}"

        # DrivingModel's submodules are built in the default dtype; SimLingo runs
        # in bfloat16. The waypoint heads are promoted to float32 below.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.vlm = DrivingModel(
            cfg_data_module=cfg.data_module,
            processor=tokenizer,
            cache_dir=cache_dir,
            vision_model_cfg=cfg.model.vision_model,
            language_model_cfg=cfg.model.language_model,
        )
        torch.set_default_dtype(default_dtype)
        self.vlm.load_state_dict(torch.load(config_path))

        # Freeze the whole VLM, then re-enable just the two SimLingo waypoint
        # heads — those *are* the deterministic policy μ and the only part of
        # the VLM that learns. Promote them to float32 so the tiny actor LR isn't
        # swallowed by bf16 rounding and they line up with the float32 critic.
        for p in self.vlm.parameters():
            p.requires_grad_(False)
        self.driving_adaptor = self.vlm.adaptors.driving
        self.actor_heads = nn.ModuleList(
            [self.driving_adaptor.route_head, self.driving_adaptor.speed_wps_head]
        ).float()
        for p in self.actor_heads.parameters():
            p.requires_grad_(True)

        self.feature_dim = int(self.vlm.language_model.hidden_size)
        self.critic = value_head_factory(self.feature_dim, ACTION_DIM)

        # ``compute_loss`` normalizes the raw env reward with the same stateless
        # "carla" transform the agent uses for telemetry, so the value the critic
        # trains on is self-contained in the contract method.
        self.reward_processor = RewardProcessor("carla", 1.0)

    # --- NetworkInterface contract ----------------------------------------

    def init_state(self) -> torch.Tensor:
        return torch.zeros(1)

    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
        del task_prompt
        return []

    @torch.no_grad()
    def infer(self, data: InferInput) -> InferResult:
        *_, driving_features = self.vlm(data.s_seq)
        features = driving_features.to(torch.float32)  # (1, NUM_WP_QUERIES, hidden)
        action = self.driving_adaptor.get_predictions(features)
        s = features.mean(dim=1)
        critic_out = self.critic(s, action.unsqueeze(1)).output
        return self._infer_result(action, s, critic_out, features)

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        # Buffer window is two columns: col 0 = state t, col 1 = state t+1. The
        # combined critic + actor loss is one tensor; the critic-freeze in the
        # actor loss keeps them on disjoint parameters, so a single backward works.
        feat = data.observations[:, 0]  # (B, NUM_WP_QUERIES, hidden)
        feat_next = data.observations[:, 1]
        a = data.actions[:, 1]
        r = self.reward_processor.normalize(data.rewards[:, 1, 0])
        done = data.dones[:, 1, 0]
        s = feat.mean(dim=1)
        s_next = feat_next.mean(dim=1)

        # Bootstrap value Q(s', μ(s')) on the next-state column, no grad.
        with torch.no_grad():
            a_next = self.driving_adaptor.get_predictions(feat_next)
            next_output = self.critic(s_next, a_next.unsqueeze(1)).output
        target_q = self.critic.compute_target_value(next_output, r.unsqueeze(1), done.unsqueeze(1))

        # The value head owns forward → range update → categorical loss; simlingo
        # just hands it (state, action chunk, per-gamma target).
        critic_loss, critic_info = self.critic.compute_critic_loss(
            s, a.unsqueeze(1), target_q, False
        )
        actor_loss, actor_info = self._actor_loss(feat, s)

        info = {
            "losses/critic_loss": critic_info["critic_loss"],
            "losses/actor_loss": float(actor_loss.item()),
            "losses/q_value": critic_info["curr_critic_value"],
            "losses/target_q": critic_info["target_value"],
            "losses/value_range": critic_info["value_range"],
            **actor_info,
        }
        return LossResult(loss=critic_loss + actor_loss, info=info)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        feat = data.observations[:, 0]
        feat_next = data.observations[:, 1]
        a = data.actions[:, 1]
        r = self.reward_processor.normalize(data.rewards[:, 1, 0])
        done = data.dones[:, 1, 0]
        s = feat.mean(dim=1)
        s_next = feat_next.mean(dim=1)

        # Bootstrap value Q(s', μ(s')) on the next-state column, no grad.
        with torch.no_grad():
            a_next = self.driving_adaptor.get_predictions(feat_next)
            next_output = self.critic(s_next, a_next.unsqueeze(1)).output
        target_q = self.critic.compute_target_value(next_output, r.unsqueeze(1), done.unsqueeze(1))

        critic_loss, critic_info = self.critic.compute_critic_loss(
            s, a.unsqueeze(1), target_q, False
        )
        actor_loss, actor_info = self._actor_loss(feat_next, s_next)

        # AdamET trace inputs: actor loss → heads, -Q(s,a) → critic, TD error.
        neg_value = -self.critic.to_value(self.critic(s, a.unsqueeze(1)).output).view(-1).mean()
        et_info = EligibilityTraceInfo(
            actor_entropy_loss=actor_loss, neg_value=neg_value, delta=critic_info["delta"]
        )
        info = {
            "losses/critic_loss": critic_info["critic_loss"],
            "losses/actor_loss": float(actor_loss.item()),
            "losses/q_value": critic_info["curr_critic_value"],
            "losses/target_q": critic_info["target_value"],
            "losses/delta": critic_info["delta"],
            "losses/value_range": critic_info["value_range"],
            **actor_info,
        }

        loss_result = LossResult(loss=critic_loss + actor_loss, info=info)

        with torch.no_grad():
            action = self.driving_adaptor.get_predictions(feat_next)
            critic_out = self.critic(s_next, action.unsqueeze(1)).output
            infer_result = self._infer_result(action, s_next, critic_out, feat_next)
        return InferLossResult(infer_result=infer_result, loss_result=loss_result, et_info=et_info)

    # --- internals --------------------------------------------------------

    def _infer_result(
        self,
        action: torch.Tensor,
        s: torch.Tensor,
        critic_out: torch.Tensor,
        features: torch.Tensor,
    ) -> InferResult:
        return InferResult(
            action=action,
            value_report=self.critic.value_report(critic_out),
            rnn_state=torch.zeros(1),
            next_image=np.zeros((1, 1, 3), dtype=np.uint8),
            next_reward=0.0,
            action_token_ids=[],
            features=features,
            activations=ActivationFeatures(
                state=s, actor=action, critic=critic_out, state_predictor=s
            ),
        )

    def _actor_loss(self, feat: torch.Tensor, s: torch.Tensor) -> tuple:
        """Actor loss for the waypoint heads + telemetry, per ``actor_loss_type``.

        Returns ``(loss, info)``. ``ddpg`` is the deterministic policy gradient
        (−Q(s, μ(s)), no extra telemetry); ``awr`` is advantage-weighted
        regression (reports the softmax weight entropy). The agent calls this
        without knowing which loss is configured.
        """
        if self.actor_loss_type == "ddpg":
            return self._ddpg_actor_loss(feat, s), {}
        if self.actor_loss_type == "awr":
            loss, weight_entropy = self._awr_actor_loss(feat, s)
            return loss, {"losses/awr_weight_entropy": float(weight_entropy.item())}
        raise ValueError(f"Unknown actor_loss_type: {self.actor_loss_type}")

    def _ddpg_actor_loss(self, feat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """L_actor = − Q(s, μ(s)) (deterministic policy gradient).

        Freeze the critic's params during this Q forward so the −Q gradient
        flows only into the waypoint heads, not the critic. ``s`` is a buffer
        read with no grad, so the heads are the sole path to the loss.
        """
        a_pred = self.driving_adaptor.get_predictions(feat)
        for p in self.critic.parameters():
            p.requires_grad_(False)
        actor_q = self.critic.to_value(self.critic(s, a_pred.unsqueeze(1)).output).view(-1)
        for p in self.critic.parameters():
            p.requires_grad_(True)
        return -actor_q.mean()

    def _awr_actor_loss(self, feat: torch.Tensor, s: torch.Tensor) -> tuple:
        """Exp-weighted (advantage-weighted) supervised regression actor.

        Sample ``awr_num_samples`` candidate actions around the current
        deterministic policy μ(s), score each with the critic Q(s, a_i),
        normalize the scores *per state* (unit-std advantages) and turn them
        into softmax weights ``w_i = softmax(adv_i / temperature)``, then regress
        μ(s) toward the candidates weighted by ``w_i``. This is
        reward/advantage-weighted regression (AWR): a supervised loss that pulls
        the policy toward the higher-Q samples *without* backpropagating through
        the critic — the Q forward (and hence the weights and the candidate
        actions) is fully detached, so the only gradient path is μ(s).

        Returns ``(actor_loss, weight_entropy)`` where ``weight_entropy`` is a
        telemetry scalar (mean per-state Shannon entropy of the softmax weights,
        nats) — low entropy means the weights collapsed onto a single candidate.
        """
        mu = self.driving_adaptor.get_predictions(feat)  # (B, A), grad flows into the heads
        b, action_dim = mu.shape
        n = self.awr_num_samples

        # Candidates + their Q scores live entirely under no_grad: they are
        # regression *targets*, must not carry gradient into the heads/critic.
        with torch.no_grad():
            noise = torch.randn(b, n, action_dim, device=mu.device) * self.awr_sample_noise
            # Keep μ(s) itself as the first candidate so the current policy is
            # always represented in the mix (zero noise on sample 0).
            noise[:, 0, :] = 0.0
            cand = mu.unsqueeze(1) + noise  # (B, N, A)

            # Q(s, a_i): broadcast the state over the N samples and score the
            # whole (B*N) batch in one critic forward.
            s_rep = s.unsqueeze(1).expand(b, n, s.shape[-1]).reshape(b * n, -1)
            a_rep = cand.reshape(b * n, action_dim).unsqueeze(1)  # (B*N, 1, A)
            q = self.critic.to_value(self.critic(s_rep, a_rep).output).view(b, n)

            # Per-state exp weights via softmax (handles the exp and the
            # normalize-to-sum-1). softmax is shift-invariant, so centering the
            # Q's would be a no-op — we only rescale by the per-state std, which
            # keeps the temperature scale-invariant to the critic's magnitude.
            adv = q / (q.std(dim=1, keepdim=True) + 1e-8)
            weights = torch.softmax(adv / self.awr_temperature, dim=1)  # (B, N)
            weight_entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()

        # Weighted regression of μ(s) toward the candidates.
        sq = ((mu.unsqueeze(1) - cand) ** 2).sum(dim=-1)  # (B, N)
        actor_loss = (weights * sq).sum(dim=1).mean()
        return actor_loss, weight_entropy
