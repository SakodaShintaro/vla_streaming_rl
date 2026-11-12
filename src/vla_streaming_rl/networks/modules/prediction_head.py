# SPDX-License-Identifier: MIT
import copy

import torch
import torch.nn as nn

from .blocks import SimbaBlock


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

    予測器 T(φ(s), a) は、ステップごとに割引率 ``gamma`` で割り引いた将来の状態の正規化した和
    (1 - gamma) Σ_k gamma^k φ(s_{t+1+k}) を決定的に回帰する。目標は次の状態と、次の状態で
    方策が選ぶ行動での目標の予測器の出力から TD で作る。状態は価値の TD での報酬の役を、
    予測器は行動価値の役をする。あわせて、状態と次の状態から行動を当てる逆運動モデルの
    損失を足す。次の状態は入力にその行動のトークンを含むので、写すだけで解ける損失でもある。

    遷移ごとの予測の誤差は内発報酬にもする。誤差の指数移動平均の平均と分散で標準化してから
    ``intrinsic_reward_weight`` を掛ける。平均を引くので、ステップを重ねるだけでは報酬が
    たまらず、いつもより予測が外れる遷移ほど報酬が大きい。"""

    # 目標の予測器が予測器を指数移動平均で追う刻み
    TARGET_EMA_RATE = 0.001
    # 内発報酬を標準化する誤差の平均と分散が、遷移ごとの誤差を指数移動平均で追う刻み
    INTRINSIC_EMA_RATE = 0.01

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        detach_predictor: bool,
        gamma: float,
        td_loss_weight: float,
        idm_loss_weight: float,
        intrinsic_reward_weight: float,
    ) -> None:
        super().__init__()
        # 予測の損失の勾配をエンコーダに流さない
        self.detach_predictor = detach_predictor
        self.gamma = gamma
        self.td_loss_weight = td_loss_weight
        self.idm_loss_weight = idm_loss_weight
        self.intrinsic_reward_weight = intrinsic_reward_weight
        # 遷移ごとの予測の誤差の指数移動平均の平均と分散
        self.register_buffer("error_mean", torch.zeros(()))
        self.register_buffer("error_var", torch.ones(()))
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
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """``state`` (B, state_dim) と ``action`` (B, action_dim) から、
        ``next_state`` (B, state_dim) と ``next_action`` (B, action_dim) で
        ブートストラップした状態の割引和を予測する損失と、逆運動モデルの損失、および遷移ごとの
        内発報酬 (B,)。``next_state`` は勾配なしで作ったもの。状態の流れはエピソードをまたいで
        続くものとして、エピソードの終わりでもブートストラップする（区切りで目標の作り方を
        変えると、そこで誤差が跳ねて内発報酬になる）。"""
        if self.detach_predictor:
            state = state.detach()

        self._update_target()
        with torch.no_grad():
            boot = self.target_predictor(torch.cat([next_state, next_action], dim=-1))
            target = (1.0 - self.gamma) * next_state + self.gamma * boot
        pred = self.predictor(torch.cat([state, action], dim=-1))
        error = (pred - target).pow(2).mean(dim=-1)  # (B,)
        td_loss = error.mean()
        intrinsic_reward = self._intrinsic_reward(error.detach())
        # 目標の大半は目標の予測器の出力で、隣り合うステップの状態は似ているので、
        # td_loss は何も学ばなくても小さい。今の状態をそのまま予測にしたときの損失と、
        # 目標のバッチの分散を物差しとして残す
        copy_loss = (state.detach() - target).pow(2).mean()

        # 行動は [-1, 1] なので tanh で範囲を合わせる
        idm_pred = torch.tanh(self.inverse_dynamics(torch.cat([state, next_state], dim=-1)))
        idm_loss = (idm_pred - action).pow(2).mean()

        loss = self.td_loss_weight * td_loss + self.idm_loss_weight * idm_loss
        info = {
            "seq_loss": loss.item(),
            "seq_td_loss": td_loss.item(),
            "seq_idm_loss": idm_loss.item(),
            "seq_td_copy_loss": copy_loss.item(),
            "intrinsic_reward": intrinsic_reward.mean().item(),
            **self._collapse_info(state, target),
        }
        return loss, info, intrinsic_reward

    @torch.no_grad()
    def _intrinsic_reward(self, error: torch.Tensor) -> torch.Tensor:
        """遷移ごとの予測の誤差 (B,) を、それまでの誤差の平均と分散で標準化して重みを掛けた
        内発報酬 (B,)。そのあと平均と分散をこのバッチの誤差へ進める。"""
        reward = (
            self.intrinsic_reward_weight
            * (error - self.error_mean)
            / torch.sqrt(self.error_var + 1e-8)
        )
        batch_mean = error.mean()
        self.error_mean.lerp_(batch_mean, self.INTRINSIC_EMA_RATE)
        self.error_var.lerp_((error - self.error_mean).pow(2).mean(), self.INTRINSIC_EMA_RATE)
        return reward
