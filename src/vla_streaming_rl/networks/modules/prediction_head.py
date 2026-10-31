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

    遷移は行動チャンク単位で数える。予測器 T(φ(s), a) は、チャンク1つごとに割引率 ``gamma``
    で割り引いた将来の状態の正規化した和 (1 - gamma) Σ_k gamma^k φ(s_{t+H(1+k)}) を決定的に
    回帰する（H はチャンクの長さ ``horizon``）。目標は次の状態と、次の状態で方策が選ぶ
    行動チャンクでの目標の予測器の出力から TD で作る。状態は価値の TD での報酬の役を、
    予測器は行動価値の役をする。あわせて、状態と次の状態から行動チャンクを当てる逆運動モデルの
    損失を足す。次の状態は入力にチャンクの最後の行動のトークンを含むので、その分は写すだけで
    解ける損失でもある。"""

    # 目標の予測器が予測器を指数移動平均で追う刻み
    TARGET_EMA_RATE = 0.001

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
        detach_predictor: bool,
        gamma: float,
        td_loss_weight: float,
        idm_loss_weight: float,
    ) -> None:
        super().__init__()
        # 予測の損失の勾配をエンコーダに流さない
        self.detach_predictor = detach_predictor
        self.gamma = gamma
        self.td_loss_weight = td_loss_weight
        self.idm_loss_weight = idm_loss_weight
        chunk_dim = horizon * action_dim
        self.predictor = _MLP(
            state_dim + chunk_dim, predictor_hidden_dim, predictor_block_num, state_dim
        )
        # ブートストラップに使う目標の予測器。勾配では学習せず、予測器を指数移動平均で追う
        self.target_predictor = copy.deepcopy(self.predictor).requires_grad_(False)
        self.inverse_dynamics = _MLP(
            2 * state_dim, predictor_hidden_dim, predictor_block_num, chunk_dim
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
        action_chunk: torch.Tensor,
        next_state: torch.Tensor,
        next_action_chunk: torch.Tensor,
        chunk_dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """``state`` (B, state_dim) と ``action_chunk`` (B, horizon, action_dim) から、
        ``next_state`` (B, state_dim) と ``next_action_chunk`` (B, horizon, action_dim) で
        ブートストラップした状態の割引和を予測する損失と、逆運動モデルの損失。
        ``next_state`` は勾配なしで作ったもの。``chunk_dones`` (B, horizon, 1) のどこかで
        エピソードが終わっていれば、ブートストラップしない。"""
        if self.detach_predictor:
            state = state.detach()
        action = action_chunk.flatten(start_dim=1)

        self._update_target()
        with torch.no_grad():
            boot = self.target_predictor(
                torch.cat([next_state, next_action_chunk.flatten(start_dim=1)], dim=-1)
            )
            # エピソードが終わった遷移には続く状態がないので、次の状態だけを目標にする
            boot_weight = self.gamma * (1.0 - chunk_dones.amax(dim=1))
            target = (1.0 - boot_weight) * next_state + boot_weight * boot
        pred = self.predictor(torch.cat([state, action], dim=-1))
        td_loss = (pred - target).pow(2).mean()
        # 目標の大半は目標の予測器の出力で、隣り合うチャンクの状態は似ているので、
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
            **self._collapse_info(state, target),
        }
        return loss, info
