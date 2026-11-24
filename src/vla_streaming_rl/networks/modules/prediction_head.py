# SPDX-License-Identifier: MIT
import copy
import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

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


def _weights_as_buffers(module: nn.Module) -> nn.Module:
    """``module`` の重みをすべて buffer に付け替えて返す。勾配では学習しないが学習中に更新される
    重み（指数移動平均の目標）を、パラメータではなく状態として持たせて、チェックポイントに載せる。"""
    for submodule in module.modules():
        for name, param in list(submodule.named_parameters(recurse=False)):
            delattr(submodule, name)
            submodule.register_buffer(name, param.detach().clone())
    return module


class LatentTDPredictionHead(nn.Module):
    """ヘッドが読む状態そのものを予測先にした TD-JEPA (Bagatella et al., 2025) の対称版。

    予測器 T(φ(s), a) は、割引率 ``gammas`` のそれぞれについて、ステップごとに割り引いた将来の
    状態の正規化した和 (1 - γ) Σ_k γ^k φ(s_{t+1+k}) を決定的に回帰する。割引率ごとに予測する
    地平が違い、短い割引率は数ステップ先を、長い割引率は数十ステップ先までを読む。目標は割引率
    ごとに、次の状態と、次の状態で方策が選ぶ行動での目標の予測器の出力から TD で作る。状態は
    価値の TD での報酬の役を、予測器は行動価値の役をする。あわせて、状態と次の状態から行動を
    当てる逆運動モデルの損失を足す。次の状態は入力にその行動のトークンを含むので、写すだけで解ける損失でもある。

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
        gammas: list[float],
        td_loss_weight: float,
        idm_loss_weight: float,
        intrinsic_reward_weight: float,
    ) -> None:
        super().__init__()
        # 予測の損失の勾配をエンコーダに流さない
        self.detach_predictor = detach_predictor
        assert len(gammas) >= 1 and all(0.0 <= g < 1.0 for g in gammas), gammas
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float32), persistent=False)
        self.state_dim = state_dim
        self.td_loss_weight = td_loss_weight
        self.idm_loss_weight = idm_loss_weight
        self.intrinsic_reward_weight = intrinsic_reward_weight
        # 遷移ごとの予測の誤差の指数移動平均の平均と分散
        self.register_buffer("error_mean", torch.zeros(()))
        self.register_buffer("error_var", torch.ones(()))
        # 割引率ごとの予測を並べて出す
        self.predictor = _MLP(
            state_dim + action_dim,
            predictor_hidden_dim,
            predictor_block_num,
            len(gammas) * state_dim,
        )
        # ブートストラップに使う目標の予測器。勾配では学習せず、予測器を指数移動平均で追う。
        # 重みは buffer なので、チェックポイントに載る
        self.target_predictor = _weights_as_buffers(copy.deepcopy(self.predictor))
        self.inverse_dynamics = _MLP(
            2 * state_dim, predictor_hidden_dim, predictor_block_num, action_dim
        )

    @torch.no_grad()
    def _update_target(self) -> None:
        for name, online in self.predictor.named_parameters():
            self.target_predictor.get_buffer(name).lerp_(online, self.TARGET_EMA_RATE)

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
        batch_size, num_gammas = state.shape[0], self.gammas.shape[0]
        gammas = self.gammas.view(1, num_gammas, 1)
        with torch.no_grad():
            boot = self.target_predictor(torch.cat([next_state, next_action], dim=-1))
            boot = boot.view(batch_size, num_gammas, self.state_dim)
            target = (1.0 - gammas) * next_state.unsqueeze(1) + gammas * boot
        pred = self.predictor(torch.cat([state, action], dim=-1))
        pred = pred.view(batch_size, num_gammas, self.state_dim)
        error_per_gamma = (pred - target).pow(2).mean(dim=-1)  # (B, num_gammas)
        # 遷移ごとの誤差は割引率で平均したもの
        error = error_per_gamma.mean(dim=-1)  # (B,)
        td_loss = error.mean()
        intrinsic_reward = self._intrinsic_reward(error.detach())
        # 目標の大半は目標の予測器の出力で、隣り合うステップの状態は似ているので、
        # td_loss は何も学ばなくても小さい。今の状態をそのまま予測にしたときの損失と、
        # 目標のバッチの分散を物差しとして残す
        copy_loss = (state.detach().unsqueeze(1) - target).pow(2).mean()

        # 行動は [-1, 1] なので tanh で範囲を合わせる
        idm_pred = torch.tanh(self.inverse_dynamics(torch.cat([state, next_state], dim=-1)))
        idm_loss = (idm_pred - action).pow(2).mean()

        loss = self.td_loss_weight * td_loss + self.idm_loss_weight * idm_loss
        info = {
            "seq_loss": loss.item(),
            "seq_td_loss": td_loss.item(),
            "seq_idm_loss": idm_loss.item(),
            "seq_td_copy_loss": copy_loss.item(),
            **{
                f"seq_td_loss_g{gamma:g}": value.item()
                for gamma, value in zip(
                    self.gammas.tolist(), error_per_gamma.mean(dim=0), strict=True
                )
            },
            "intrinsic_reward": intrinsic_reward.mean().item(),
            **self._collapse_info(state, target.flatten(start_dim=1)),
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


class MeanFlowPredictionHead(nn.Module):
    """次の状態を、今の状態と行動を条件に MeanFlow (Geng et al., 2025) で生成する予測ヘッド。

    予測先は次の 1 ステップの状態そのもので、割引和もブートストラップも使わない。状態は
    ``token_dim`` 幅のトークンの並びとして扱い、次元ごとの指数移動平均の平均と分散で標準化
    してから予測先にする。Transformer が、ノイズを混ぜた予測先 z と 2 つの時刻 t >= r から、
    区間 [r, t] の平均の速度 u と、時刻 t の瞬間の速度 v を出す。

    損失は i-MeanFlow の形。z = (1 - t) x + t e、本当の速度は e - x。v はそのまま回帰し、u は
    u + (t - r) du/dt が本当の速度に合うようにする。du/dt は (dz/dt, dt, dr) = (予測した v, 1, 0)
    の向きの方向微分で、勾配は流さない。どちらの項も、誤差の大きさで割る重みを掛ける。
    生成は 1 ステップで、z を標準正規分布から引いて x = z - u(z, t = 1, r = 0)。

    遷移ごとの内発報酬は、1 ステップで生成した次の状態と本当の次の状態の二乗誤差を、その指数
    移動平均の平均と分散で標準化して ``intrinsic_reward_weight`` を掛けたもの。"""

    # 予測先と内発報酬を標準化する統計が、バッチの値を指数移動平均で追う刻み
    EMA_RATE = 0.01

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        token_dim: int,
        width: int,
        block_num: int,
        heads: int,
        time_mu: float,
        time_sigma: float,
        adaptive_c: float,
        loss_weight: float,
        intrinsic_reward_weight: float,
    ) -> None:
        super().__init__()
        assert state_dim % token_dim == 0, (state_dim, token_dim)
        self.token_dim = token_dim
        self.tokens_num = state_dim // token_dim
        self.width = width
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.adaptive_c = adaptive_c
        self.loss_weight = loss_weight
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.register_buffer("target_mean", torch.zeros(state_dim))
        self.register_buffer("target_var", torch.ones(state_dim))
        self.register_buffer("error_mean", torch.zeros(()))
        self.register_buffer("error_var", torch.ones(()))
        self.state_in = nn.Linear(token_dim, width)
        self.target_in = nn.Linear(token_dim, width)
        self.state_position = nn.Parameter(torch.zeros(1, self.tokens_num, width))
        self.target_position = nn.Parameter(torch.zeros(1, self.tokens_num, width))
        # 2 つの時刻（t と t - r）の特徴と行動を、予測先のトークンに足す条件にする
        self.condition = nn.Sequential(
            nn.Linear(2 * width + action_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList([Block(dim=width, num_heads=heads) for _ in range(block_num)])
        for block in self.blocks:
            # 組み込みの注意（scaled_dot_product_attention）は方向微分（JVP）を通せない
            block.attn.fused_attn = False
        self.norm = nn.LayerNorm(width)
        self.out_u = nn.Linear(width, token_dim)
        self.out_v = nn.Linear(width, token_dim)
        for out in (self.out_u, self.out_v):
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)

    def _time_features(self, time: torch.Tensor) -> torch.Tensor:
        half = self.width // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=time.device) / half)
        angles = time[:, None] * 1000.0 * freqs[None]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def _velocities(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        noisy: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """状態のトークン (B, tokens_num, token_dim)、行動、ノイズを混ぜた予測先（同じ形）、
        時刻 t, r (B,) から、平均の速度 u と瞬間の速度 v（予測先と同じ形）。"""
        condition = self.condition(
            torch.cat([self._time_features(t), self._time_features(t - r), action], dim=-1)
        )
        x = torch.cat(
            [
                self.state_in(state) + self.state_position,
                self.target_in(noisy) + self.target_position + condition[:, None],
            ],
            dim=1,
        )
        for block in self.blocks:
            x = block(x)
        hidden = self.norm(x[:, self.tokens_num :])
        return self.out_u(hidden), self.out_v(hidden)

    def _adaptive_l2(self, error: torch.Tensor) -> torch.Tensor:
        """サンプルごとの二乗誤差に、その大きさで割る重み（勾配なし）を掛けたもの (B,)。"""
        squared = error.pow(2).mean(dim=(1, 2))
        return squared / (squared.detach() + self.adaptive_c)

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """``state`` (B, state_dim) と ``action`` (B, action_dim) から ``next_state``
        (B, state_dim、勾配なしで作ったもの) を生成する MeanFlow の損失と、遷移ごとの内発報酬
        (B,)。``next_action`` は使わない。"""
        del next_action
        batch = state.shape[0]
        with torch.no_grad():
            if batch > 1:
                self.target_mean.lerp_(next_state.mean(dim=0), self.EMA_RATE)
                self.target_var.lerp_(next_state.var(dim=0), self.EMA_RATE)
            target = (next_state - self.target_mean) / torch.sqrt(self.target_var + 1e-4)
            target = target.view(batch, self.tokens_num, self.token_dim)
            samples = torch.sigmoid(
                torch.randn(batch, 2, device=state.device) * self.time_sigma + self.time_mu
            )
            t, r = samples.max(dim=1).values, samples.min(dim=1).values
            noise = torch.randn_like(target)
            level = t.view(-1, 1, 1)
            noisy = (1.0 - level) * target + level * noise
            velocity = noise - target
        tokens = state.view(batch, self.tokens_num, self.token_dim)
        frozen = tokens.detach()
        with torch.no_grad():
            _, v_now = self._velocities(frozen, action, noisy, t, r)
        _, dudt = torch.func.jvp(
            lambda z_, t_, r_: self._velocities(frozen, action, z_, t_, r_)[0],
            (noisy, t, r),
            (v_now, torch.ones_like(t), torch.zeros_like(r)),
        )
        u, v = self._velocities(tokens, action, noisy, t, r)
        mean_loss = self._adaptive_l2(u + (t - r).view(-1, 1, 1) * dudt.detach() - velocity)
        flow_loss = self._adaptive_l2(v - velocity)
        loss = self.loss_weight * (mean_loss + flow_loss).mean()
        with torch.no_grad():
            # 1 ステップで生成した次の状態の、本当の次の状態からの外れ方
            start = torch.randn_like(target)
            ones, zeros = torch.ones_like(t), torch.zeros_like(t)
            generated = start - self._velocities(frozen, action, start, ones, zeros)[0]
            error = (generated - target).pow(2).mean(dim=(1, 2))
            intrinsic_reward = (
                self.intrinsic_reward_weight
                * (error - self.error_mean)
                / torch.sqrt(self.error_var + 1e-8)
            )
            self.error_mean.lerp_(error.mean(), self.EMA_RATE)
            self.error_var.lerp_((error - self.error_mean).pow(2).mean(), self.EMA_RATE)
        info = {
            "seq_loss": loss.item(),
            "seq_mean_flow_loss": mean_loss.mean().item(),
            "seq_flow_loss": flow_loss.mean().item(),
            "seq_velocity_error": (v.detach() - velocity).pow(2).mean().item(),
            "seq_sample_error": error.mean().item(),
            "intrinsic_reward": intrinsic_reward.mean().item(),
            **LatentTDPredictionHead._collapse_info(state, next_state),
        }
        return loss, info, intrinsic_reward
