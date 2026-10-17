# SPDX-License-Identifier: MIT
import copy

import torch
import torch.nn as nn

from .blocks import SimbaBlock
from .flux_dit import FluxDiT
from .reward_embed import embed_reward


class StatePredictionHead(nn.Module):
    """将来の (画像, 報酬) を予測する生成モデル。flow_matching では、割引率 ``gamma`` で
    割り引いた将来の占有分布（successor representation）を TD^2-CFM (Farebrother et al.,
    Temporal Difference Flows) で学習する。mean_flow は1ステップ先を予測する。"""

    # 目標の予測器が予測器を指数移動平均で追う刻み
    TARGET_EMA_RATE = 0.001
    # ブートストラップの点を作るとき、目標の予測器の ODE を中点法で解くステップ数
    ODE_STEPS = 10

    def __init__(
        self,
        image_latent_dim: int,
        state_dim: int,
        action_dim: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        predictor_type: str,
        detach_predictor: bool,
        gamma: float,
    ) -> None:
        super().__init__()
        # 予測先は画像と報酬なので、状態の幅は使わない
        del state_dim
        # 予測の損失の勾配をエンコーダに流さない
        self.detach_predictor = detach_predictor
        self.image_latent_dim = image_latent_dim
        self.predictor_type = predictor_type
        hidden_image_dim = image_latent_dim
        self.state_predictor = FluxDiT(
            in_channels=hidden_image_dim,
            out_channels=hidden_image_dim,
            vec_in_dim=action_dim,
            context_in_dim=hidden_image_dim,
            hidden_size=predictor_hidden_dim,
            mlp_ratio=4.0,
            num_heads=8,
            depth_double_blocks=predictor_block_num,
            depth_single_blocks=predictor_block_num,
            axes_dim=[64],
            theta=10000,
            qkv_bias=True,
            use_r_time=(predictor_type == "mean_flow"),
        )
        self.gamma = gamma
        # ブートストラップに使う目標の予測器。勾配では学習せず、予測器を指数移動平均で追う
        self.target_predictor = copy.deepcopy(self.state_predictor).requires_grad_(False)

    def _loss_flow_matching(
        self,
        predictor_state: torch.Tensor,
        action: torch.Tensor,
        x1: torch.Tensor,
        next_predictor_state: torch.Tensor,
        next_action: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """1ステップ先の項と、目標の予測器から作るブートストラップの項を
        ``1 - gamma`` 対 ``gamma`` で混ぜる。"""
        shape_t = (x1.shape[0],) + (1,) * (len(x1.shape) - 1)
        t = torch.rand(shape_t, device=x1.device)

        # 1ステップ先の項。ノイズから次の観測へ向かう直線の経路の速度を回帰する
        x0 = torch.randn_like(x1)
        xt = (1.0 - t) * x0 + t * x1
        pred_dict = self.state_predictor.forward(xt, t, predictor_state, action, None)
        one_step_loss = (pred_dict.output - (x1 - x0)).pow(2).mean(dim=(1, 2))

        # ブートストラップの項。次の状態と次の行動を条件に目標の予測器の ODE を時刻 t まで
        # 解いた点で、目標の予測器がそこで出す速度を回帰する
        with torch.no_grad():
            boot_xt = self._solve_target(torch.randn_like(x1), t, next_predictor_state, next_action)
            boot_vt = self.target_predictor.forward(
                boot_xt, t, next_predictor_state, next_action, None
            ).output
        boot_pred = self.state_predictor.forward(boot_xt, t, predictor_state, action, None).output
        bootstrap_loss = (boot_pred - boot_vt).pow(2).mean(dim=(1, 2))

        # エピソードが終わった遷移には続く状態がないので、1ステップ先の項だけで学習する
        boot_weight = self.gamma * (1.0 - done.view(-1))
        loss = ((1.0 - boot_weight) * one_step_loss + boot_weight * bootstrap_loss).mean()
        info = {
            "seq_loss": loss.item(),
            "seq_one_step_loss": one_step_loss.mean().item(),
            "seq_bootstrap_loss": bootstrap_loss.mean().item(),
        }
        return loss, pred_dict.activation, info

    def _solve_target(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """目標の予測器の ODE を、時刻 0 の ``x`` から各サンプルの時刻 ``t`` まで中点法で解く。"""
        dt = t / self.ODE_STEPS
        for step in range(self.ODE_STEPS):
            time = dt * step
            v = self.target_predictor.forward(x, time, cond, action, None).output
            v = self.target_predictor.forward(
                x + 0.5 * dt * v, time + 0.5 * dt, cond, action, None
            ).output
            x = x + dt * v
        return x

    @torch.no_grad()
    def _update_target(self) -> None:
        for target, online in zip(
            self.target_predictor.parameters(), self.state_predictor.parameters(), strict=True
        ):
            target.lerp_(online, self.TARGET_EMA_RATE)

    def _loss_mean_flow(
        self,
        predictor_state: torch.Tensor,
        action: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        B = x1.shape[0]
        device = x1.device
        x0 = torch.randn_like(x1)
        # Logit-normal time sampling (MeanFlow paper recommendation): μ=-0.4, σ=1.0
        # biases t toward smaller values where modelling is harder.
        normal_samples = torch.randn(B, 2, device=device) * 1.0 + (-0.4)
        time_samples = torch.sigmoid(normal_samples)
        # Lower time t is closer to noise (=0); upper time r is closer to target (=1).
        t = time_samples.min(dim=1).values
        r = time_samples.max(dim=1).values
        # r = t for 25% of samples (paper-recommended) so the network also learns FM at r=t.
        fm_mask = torch.rand(B, device=device) < 0.25
        r = torch.where(fm_mask, t, r)

        v = x1 - x0
        t_b = t.view(-1, 1, 1)
        xt = (1.0 - t_b) * x0 + t_b * x1

        def f(x: torch.Tensor, t_: torch.Tensor) -> torch.Tensor:
            return self.state_predictor.forward(x, t_, predictor_state, action, r).output

        # JVP computes total derivative along the trajectory: du/dt = ∂u/∂x · v + ∂u/∂t.
        # SDPA MATH backend because Flash Attention lacks double-backward / forward-mode AD.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            u, du_dt = torch.func.jvp(f, (xt, t), (v, torch.ones_like(t)))

        # In our convention (t ≤ r) the identity gives u_target = v + (r - t) * du/dt.
        gap = (r - t).view(-1, 1, 1)
        u_target = v + gap * du_dt
        # Adaptive loss weighting (MeanFlow paper): without it (r-t)·du/dt self-amplifies
        # and training diverges.
        delta_sq = (u - u_target.detach()).pow(2)
        w = 1.0 / (delta_sq.detach().mean(dim=(1, 2), keepdim=True) + 1e-3)
        loss = (w * delta_sq).mean()

        # Tracking-only: restrict to the r=t subset, where u_target = v exactly.
        per_sample_uv_sq = (u - v).detach().pow(2).mean(dim=(1, 2))
        fm_count = fm_mask.sum().clamp(min=1)
        fm_mse = (per_sample_uv_sq * fm_mask.float()).sum() / fm_count
        du_dt_mag = du_dt.detach().abs().mean()

        activation = torch.zeros((B, 1), device=device)
        info = {
            "seq_loss": loss.item(),
            "mf_fm_mse": fm_mse.item(),
            "mf_du_dt": du_dt_mag.item(),
        }
        return loss, activation, info

    def encode_target(
        self, next_image_tokens: torch.Tensor, next_reward: torch.Tensor
    ) -> torch.Tensor:
        """Build the flow-matching regression target ``x1`` (B, N+1, C')
        from the next step's image tokens (B, N, C') -- every image of the
        observation, one after another -- which the caller encodes without
        gradient (fixed target), and the next reward spread over the channels,
        appended as the final position.
        """
        target_reward_next = embed_reward(next_reward, self.image_latent_dim).squeeze(1)  # (B, C')
        return torch.cat([next_image_tokens, target_reward_next.unsqueeze(1)], dim=1)

    def compute_loss(
        self,
        predictor_state: torch.Tensor,
        action: torch.Tensor,
        next_image_tokens: torch.Tensor,
        next_reward: torch.Tensor,
        next_predictor_state: torch.Tensor,
        next_action: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Prediction loss: train the predictor so that ``(predictor_state, action)``
        maps to the encoded future ``(image, reward)`` -- the successor
        representation for flow_matching, bootstrapped off ``next_predictor_state``
        and ``next_action`` unless ``done``; the next one for mean_flow.

        Mirrors ``PolicyHead.compute_actor_loss`` / ``ValueHead.compute_critic_loss``
        — the head owns its loss and logging. ``detach_predictor`` (set at
        construction) stops the gradient into the encoder. The caller supplies
        the already-prepared ``predictor_state`` (its layout is network-specific).
        """
        if self.detach_predictor:
            predictor_state = predictor_state.detach()

        # Context layout: (B, tokens, C) where C is the image channel dim
        # (no-op for callers already in that shape).
        C = self.image_latent_dim
        predictor_state = predictor_state.view(predictor_state.size(0), -1, C)
        next_predictor_state = next_predictor_state.view(next_predictor_state.size(0), -1, C)

        self._update_target()
        x1 = self.encode_target(next_image_tokens, next_reward)
        if self.predictor_type == "flow_matching":
            loss, _, info = self._loss_flow_matching(
                predictor_state, action, x1, next_predictor_state, next_action, done
            )
        elif self.predictor_type == "mean_flow":
            loss, _, info = self._loss_mean_flow(predictor_state, action, x1)
        else:
            raise ValueError(f"Unknown predictor_type: {self.predictor_type}")
        return loss, info


class _MLP(nn.Module):
    """入力を隠れ層の幅に写し、SimbaBlock を重ねて出力の幅に写す決定的な関数。"""

    def __init__(self, in_dim: int, hidden_dim: int, block_num: int, out_dim: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.norm(self.fc_mid(self.fc_in(x))))


class LatentTDPredictionHead(nn.Module):
    """ヘッドが読む状態そのものを予測先にした TD-JEPA (Bagatella et al., 2025) の対称版。

    予測器 T(φ(s), a) は、割引率 ``gamma`` で割り引いた将来の状態の正規化した和
    (1 - gamma) Σ_k gamma^k φ(s_{t+1+k}) を決定的に回帰する。目標は次の状態と、次の状態で
    方策が選ぶ行動での目標の予測器の出力から TD で作る。状態は価値の TD での報酬の役を、
    予測器は行動価値の役をする。あわせて、状態と次の状態から行動を当てる逆運動モデルの
    損失を足す。次の状態は入力に行動のトークンを含むので、写すだけで解ける損失でもある。"""

    # 目標の予測器が予測器を指数移動平均で追う刻み
    TARGET_EMA_RATE = 0.001

    def __init__(
        self,
        image_latent_dim: int,
        state_dim: int,
        action_dim: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        detach_predictor: bool,
        gamma: float,
        td_loss_weight: float,
        idm_loss_weight: float,
    ) -> None:
        super().__init__()
        # 予測先は状態なので、画像のトークンの幅は使わない
        del image_latent_dim
        # 予測の損失の勾配をエンコーダに流さない
        self.detach_predictor = detach_predictor
        self.gamma = gamma
        self.td_loss_weight = td_loss_weight
        self.idm_loss_weight = idm_loss_weight
        self.predictor = _MLP(
            state_dim + action_dim, predictor_hidden_dim, predictor_block_num, state_dim
        )
        # ブートストラップに使う目標の予測器。勾配では学習せず、予測器を指数移動平均で追う
        self.target_predictor = copy.deepcopy(self.predictor).requires_grad_(False)
        self.inverse_dynamics = _MLP(
            2 * state_dim, predictor_hidden_dim, predictor_block_num, action_dim
        )

    @torch.no_grad()
    def _update_target(self) -> None:
        for target, online in zip(
            self.target_predictor.parameters(), self.predictor.parameters(), strict=True
        ):
            target.lerp_(online, self.TARGET_EMA_RATE)

    @staticmethod
    def _collapse_info(state: torch.Tensor, target: torch.Tensor) -> dict:
        """表現が潰れていないかを見るための、バッチでの次元ごとの標準偏差の平均と実効階数、
        および目標の次元ごとの分散の平均。バッチが1つの状態だけのときは測れないので空。"""
        if state.shape[0] == 1:
            return {}
        target_var = target.to(torch.float32).var(dim=0).mean().item()
        state = state.detach().to(torch.float32)
        centered = state - state.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered)
        p = singular_values / singular_values.sum().clamp(min=1e-12)
        entropy = -(p * torch.log(p.clamp(min=1e-12))).sum()
        return {
            "seq_state_std": state.std(dim=0).mean().item(),
            "seq_state_effective_rank": torch.exp(entropy).item(),
            "seq_td_target_var": target_var,
        }

    def compute_loss(
        self,
        predictor_state: torch.Tensor,
        action: torch.Tensor,
        next_image_tokens: torch.Tensor,
        next_reward: torch.Tensor,
        next_predictor_state: torch.Tensor,
        next_action: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """``predictor_state`` (B, state_dim) と ``action`` (B, action_dim) から、
        ``next_predictor_state`` (B, state_dim) と ``next_action`` (B, action_dim) で
        ブートストラップした状態の割引和を予測する損失と、逆運動モデルの損失。
        ``next_predictor_state`` は勾配なしで作ったもの。"""
        # 予測先は状態なので、次の画像と報酬は使わない
        del next_image_tokens, next_reward
        if self.detach_predictor:
            predictor_state = predictor_state.detach()

        self._update_target()
        with torch.no_grad():
            boot = self.target_predictor(torch.cat([next_predictor_state, next_action], dim=-1))
            # エピソードが終わった遷移には続く状態がないので、次の状態だけを目標にする
            boot_weight = self.gamma * (1.0 - done.view(-1, 1))
            target = (1.0 - boot_weight) * next_predictor_state + boot_weight * boot
        pred = self.predictor(torch.cat([predictor_state, action], dim=-1))
        td_loss = (pred - target).pow(2).mean()
        # 目標の 99% は目標の予測器の出力で、隣り合うステップの状態はほぼ同じなので、
        # td_loss は何も学ばなくても小さい。今の状態をそのまま予測にしたときの損失と、
        # 目標のバッチの分散を物差しとして残す
        copy_loss = (predictor_state.detach() - target).pow(2).mean()

        # 行動は [-1, 1] なので tanh で範囲を合わせる
        idm_pred = torch.tanh(
            self.inverse_dynamics(torch.cat([predictor_state, next_predictor_state], dim=-1))
        )
        idm_loss = (idm_pred - action).pow(2).mean()

        loss = self.td_loss_weight * td_loss + self.idm_loss_weight * idm_loss
        info = {
            "seq_loss": loss.item(),
            "seq_td_loss": td_loss.item(),
            "seq_idm_loss": idm_loss.item(),
            "seq_td_copy_loss": copy_loss.item(),
            **self._collapse_info(predictor_state, target),
        }
        return loss, info
