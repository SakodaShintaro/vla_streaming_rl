# SPDX-License-Identifier: MIT
"""ステップをまたぐ記憶を持つ、Qwen3.5 の GDN の層1つ。

事前学習の重みは凍結し、LoRA だけを学習する。RTRL は層1つの構造化 RTRL で、LoRA の B に
対する再帰状態 S の感度を、1ステップ分のトークンをまとめて進める。

層の計算（transformers の Qwen3_5DecoderLayer と同じ）::

    h      = zc_rms(x, w_in)                                  rms(x) * (1 + w)
    qkv    = SiLU(causal_depthwise_conv(W_qkv h))             直近 conv_size トークンの畳み込み
    q, k   = l2norm(q), l2norm(k);  q = q / sqrt(d_k)
    beta   = sigmoid(W_b h);  alpha = exp(-exp(A_log) * softplus(W_a h + dt_bias))
    S_t    = alpha (I - beta k k^T) S_{t-1} + beta k v^T,   o = S_t^T q
    y      = W_out( rms(o) * w_norm * SiLU(W_z h) )
    x      = x + y
    x      = x + W_down( SiLU(W_gate zc_rms(x, w_post)) * W_up zc_rms(x, w_post) )
"""

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch import nn

# LoRA を付ける射影（層のすべての線形写像）
LORA_TARGETS = ("q", "k", "v", "z", "a", "b", "out", "gate", "up", "down")
# 再帰状態 S に直接入る射影。A を凍結し（LoRA-FA）、RTRL の感度を B についてだけ持つ。
# k の感度は対角だけを残す近似。ほかの射影は S に入らないので、
# 1ステップ分の逆伝播だけで勾配が決まる
RECURRENT_TARGETS = ("k", "v", "a", "b")


def zc_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5 の層の、0 を中心にした RMSNorm: rms(x) * (1 + weight)。"""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight)


def l2norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def silu_grad(c: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(c)
    return sigmoid * (1.0 + c * (1.0 - sigmoid))


class GDNMemoryLayer(nn.Module):
    """凍結した GDN の層に LoRA を付け、再帰状態をステップをまたいで持ち越す層。

    ``forward`` は1ステップ分のトークンを、前のステップから持ち越した再帰状態 S と、畳み込みが
    読む直前のトークンの正規化した入力から読む。前の状態の勾配は常に切る。``use_trace`` なら
    RTRL で、LoRA の B に対する S の感度（トレース）を前向きに運び、今のステップの損失の勾配に
    過去のステップを通る分を足す。層の入力は凍結した下の層の出力で学習するパラメータに
    よらないので、k の対角近似を除けば勾配は厳密（トレースを運ぶ間にパラメータが変わる分を除く）。
    トレースは1本の流れを順に進めるので、バッチは 1 で、1ステップに1回だけ進める。
    """

    def __init__(
        self, hf_layer: nn.Module, lora_targets: tuple[str, ...], lora_rank: int, lora_alpha: float
    ) -> None:
        super().__init__()
        attention = hf_layer.linear_attn
        assert attention.num_k_heads == attention.num_v_heads
        self.num_heads = attention.num_v_heads
        self.key_dim = attention.head_k_dim
        self.value_dim = attention.head_v_dim
        self.conv_size = attention.conv_kernel_size
        self.eps = attention.layer_norm_epsilon
        self.hidden_size = attention.hidden_size
        # 事前学習の重みは float32 の凍結したバッファ。VLM から毎回写すので保存しない
        frozen = {
            "input_norm": hf_layer.input_layernorm.weight,
            "qkv": attention.in_proj_qkv.weight,
            "conv": attention.conv1d.weight,  # (C, 1, conv_size)。最後の列が今のトークンに掛かる
            "z": attention.in_proj_z.weight,
            "a": attention.in_proj_a.weight,
            "b": attention.in_proj_b.weight,
            "A_log": attention.A_log,
            "dt_bias": attention.dt_bias,
            "out_norm": attention.norm.weight,
            "out": attention.out_proj.weight,
            "post_norm": hf_layer.post_attention_layernorm.weight,
            "gate": hf_layer.mlp.gate_proj.weight,
            "up": hf_layer.mlp.up_proj.weight,
            "down": hf_layer.mlp.down_proj.weight,
        }
        for name, weight in frozen.items():
            self.register_buffer(f"w_{name}", weight.detach().float().clone(), persistent=False)
        H, dk, dv = self.num_heads, self.key_dim, self.value_dim
        self.qkv_split = [H * dk, H * dk, H * dv]
        # LoRA: delta W = (alpha / rank) B A。A は一様乱数、B は 0 で始め、学習の始めは凍結した層と同じ
        assert set(lora_targets) <= set(LORA_TARGETS), lora_targets
        self.lora_targets = lora_targets
        self.lora_rank = lora_rank
        self.lora_scale = lora_alpha / lora_rank
        q_weight, k_weight, v_weight = torch.split(self.w_qkv, self.qkv_split, dim=0)
        target_weights = {
            "q": q_weight,
            "k": k_weight,
            "v": v_weight,
            "z": self.w_z,
            "a": self.w_a,
            "b": self.w_b,
            "out": self.w_out,
            "gate": self.w_gate,
            "up": self.w_up,
            "down": self.w_down,
        }
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for target in lora_targets:
            out_features, in_features = target_weights[target].shape
            bound = in_features**-0.5
            A = torch.empty(lora_rank, in_features, device=self.w_qkv.device).uniform_(
                -bound, bound
            )
            if target in RECURRENT_TARGETS:
                self.register_buffer(f"lora_A_frozen_{target}", A)
            else:
                self.lora_A[target] = nn.Parameter(A)
            self.lora_B[target] = nn.Parameter(
                torch.zeros(out_features, lora_rank, device=self.w_qkv.device)
            )
        # B_v, B_a, B_b, B_k に対する S の感度 (1, H, d_k, d_v, rank)。None はまだ 0
        self.trace: dict[str, torch.Tensor] | None = None

    def memory_size(self) -> int:
        """持ち越す状態の大きさ。S と、畳み込みが読む直前の conv_size - 1 トークンの正規化した入力。"""
        return (
            self.num_heads * self.key_dim * self.value_dim + (self.conv_size - 1) * self.hidden_size
        )

    def reset_trace(self) -> None:
        self.trace = None

    def lora_A_weight(self, target: str) -> torch.Tensor:
        if target in RECURRENT_TARGETS:
            return self.get_buffer(f"lora_A_frozen_{target}")
        return self.lora_A[target].clone()

    def lora_B_weight(self, target: str) -> torch.Tensor:
        """学習する重みは複製を読む。計算グラフは複製を保存するので、次のティックで backward
        する前に optimizer が重みをその場で更新しても使える（勾配は1回前の重みで評価したもの）。"""
        return self.lora_B[target].clone()

    def _proj(self, target: str, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """凍結した線形写像に、LoRA があればその差分を足したもの。"""
        y = F.linear(x, weight)
        if target in self.lora_targets:
            y = y + self.lora_scale * F.linear(
                F.linear(x, self.lora_A_weight(target)), self.lora_B_weight(target)
            )
        return y

    def _lift(self, target: str, h: torch.Tensor) -> torch.Tensor:
        """B に対する感度で、入力 h が B の行に掛かる形 s A h (..., rank)。"""
        return self.lora_scale * h @ self.lora_A_weight(target).detach().T

    def _correction(self) -> torch.Tensor:
        """値は 0 で、パラメータに対する勾配が、前の状態を通る分（運んだ感度との縮約）になる項
        (1, H, d_k, d_v)。"""
        H, dk, dv, r = self.num_heads, self.key_dim, self.value_dim, self.lora_rank
        trace = self.trace
        correction = (
            torch.einsum("bhijp,hjp->bhij", trace["v"], self.lora_B_weight("v").view(H, dv, r))
            + torch.einsum("bhijp,hp->bhij", trace["a"], self.lora_B_weight("a"))
            + torch.einsum("bhijp,hp->bhij", trace["b"], self.lora_B_weight("b"))
            + torch.einsum("bhijp,hip->bhij", trace["k"], self.lora_B_weight("k").view(H, dk, r))
        )
        return correction - correction.detach()

    def _project(
        self, x: torch.Tensor, valid: torch.Tensor, h_prev: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """1ステップ分の入力 x (B, L, D) から、再帰に入る量と出力のゲートを作る。``valid`` (B, L)
        の詰め物のトークンでは状態を変えない（alpha = 1、beta = 0）。h_prev は直前の
        conv_size - 1 トークンの正規化した入力 (B, conv_size - 1, D)。"""
        B, L, _ = x.shape
        H, dk, dv = self.num_heads, self.key_dim, self.value_dim
        h = zc_rms_norm(x, self.w_input_norm, self.eps)
        # 直前のステップの入力に続けて、このステップの入力を畳み込む
        h_all = torch.cat([h_prev.detach(), h], dim=1)  # (B, cs - 1 + L, D)
        q_weight, k_weight, v_weight = torch.split(self.w_qkv, self.qkv_split, dim=0)
        qkv_pre = torch.cat(
            [
                self._proj("q", q_weight, h_all),
                self._proj("k", k_weight, h_all),
                self._proj("v", v_weight, h_all),
            ],
            dim=-1,
        )
        conv = F.conv1d(qkv_pre.transpose(1, 2), self.w_conv, groups=qkv_pre.shape[-1])
        conv = conv.transpose(1, 2)  # (B, L, C)、SiLU の前
        q, k, v = torch.split(F.silu(conv), self.qkv_split, dim=-1)
        beta = torch.sigmoid(self._proj("b", self.w_b, h))
        a_pre = self._proj("a", self.w_a, h) + self.w_dt_bias
        log_alpha = -torch.exp(self.w_A_log) * F.softplus(a_pre)
        return {
            "h_all": h_all,
            "conv": conv,
            "q": l2norm(q.view(B, L, H, dk), 1e-6) * dk**-0.5,
            "k": l2norm(k.view(B, L, H, dk), 1e-6),
            "v": v.view(B, L, H, dv),
            "log_alpha": torch.where(valid[..., None], log_alpha, torch.zeros_like(log_alpha)),
            "beta": torch.where(valid[..., None], beta, torch.zeros_like(beta)),
            "a_pre": a_pre,
            "z": self._proj("z", self.w_z, h).view(B, L, H, dv),
        }

    def _output(self, x: torch.Tensor, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """再帰の出力 o (B, L, H, d_v) をゲートして射影し、残差と MLP を通した層の出力 (B, L, D)。"""
        o = o * torch.rsqrt(o.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.w_out_norm
        o = o * F.silu(z)
        y = x + self._proj("out", self.w_out, o.flatten(2))
        post = zc_rms_norm(y, self.w_post_norm, self.eps)
        mlp = F.silu(self._proj("gate", self.w_gate, post)) * self._proj("up", self.w_up, post)
        return y + self._proj("down", self.w_down, mlp)

    def _last_inputs(self, h_all: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """次のステップの畳み込みが読む、最後の conv_size - 1 トークンの正規化した入力。"""
        last = lengths[:, None] + torch.arange(self.conv_size - 1, device=h_all.device)
        return torch.gather(h_all, 1, last[..., None].expand(-1, -1, h_all.shape[-1])).detach()

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        S_prev: torch.Tensor,
        h_prev: torch.Tensor,
        use_trace: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """1ステップ分のトークンの層の入力 x (B, L, D) を、fla の chunk カーネルでまとめて読む。
        ``valid`` (B, L) は詰め物でないトークンで、各行の先頭に詰めてある。前の状態は
        S_prev (B, H, d_k, d_v) と、直前の conv_size - 1 トークンの正規化した入力
        h_prev (B, conv_size - 1, D)。

        層の出力 (B, L, D) と、次の S と h を返す。"""
        B, L, _ = x.shape
        lengths = valid.sum(dim=1)
        S = S_prev.detach()
        if use_trace:
            assert B == 1, B
            assert set(self.lora_targets) == set(LORA_TARGETS), self.lora_targets
            if self.trace is not None:
                S = S + self._correction()

        p = self._project(x, valid, h_prev)
        o, S_next = chunk_gated_delta_rule(
            p["q"],
            p["k"],
            p["v"],
            p["log_alpha"],
            p["beta"],
            scale=1.0,
            initial_state=S,
            output_final_state=True,
        )
        y = self._output(x, o, p["z"])

        if use_trace:
            self._advance_trace(
                p["h_all"].detach(),
                p["conv"].detach(),
                p["k"].detach(),
                p["v"].detach(),
                torch.exp(p["log_alpha"]).detach(),
                p["beta"].detach(),
                p["a_pre"].detach(),
                S_prev.detach(),
                int(lengths[0]),
            )
        return y, S_next.detach(), self._last_inputs(p["h_all"], lengths)

    @torch.no_grad()
    def _advance_trace(
        self,
        h_all: torch.Tensor,
        conv: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        a_pre: torch.Tensor,
        S_start: torch.Tensor,
        length: int,
    ) -> None:
        """感度を、このステップの詰め物でないトークン t = 0..L-1 の分だけ進める。1トークンでは
        M <- J_t M + I_t、J_t = alpha_t (I - beta_t k_t k_t^T)（キーの添字に掛かる）、I_t は前の状態を
        止めたときの、そのトークンでの S の変化（畳み込みの窓のトークンを通る分を含む）。
        L トークン分をまとめて

            M <- G_{-1} M + sum_t G_t I_t,   G_t = J_{L-1} ... J_{t+1}（G_{L-1} = I）

        とする。v, a, b の I_t はキーの添字に k_t か (d_k, d_v) の行列が掛かる形なので、G_t を
        掛けてから t で縮約する。k の感度は対角の近似で、J_t の代わりに対角
        alpha_t (1 - beta_t k_t^2) で運ぶ。"""
        H, dk, dv, cs, r = (
            self.num_heads,
            self.key_dim,
            self.value_dim,
            self.conv_size,
            self.lora_rank,
        )
        B = h_all.shape[0]
        L = length
        if self.trace is None:
            self.trace = {
                target: torch.zeros(B, H, dk, dv, r, device=h_all.device, dtype=h_all.dtype)
                for target in RECURRENT_TARGETS
            }
        trace = self.trace
        k, v, alpha, beta = k[:, :L], v[:, :L], alpha[:, :L], beta[:, :L]
        neg_A = -torch.exp(self.w_A_log)
        c_k = conv[:, :L, H * dk : 2 * H * dk].view(B, L, H, dk)
        c_v = conv[:, :L, 2 * H * dk :].view(B, L, H, dv)
        conv_k = self.w_conv[H * dk : 2 * H * dk, 0].view(H, dk, cs)
        conv_v = self.w_conv[2 * H * dk :, 0].view(H, dv, cs)

        def conv_lift(target: str, conv_weight: torch.Tensor) -> torch.Tensor:
            """畳み込みの窓のトークンの s A h を、チャネルごとの畳み込みの重みで足したもの
            (B, L, H, channels, rank)。"""
            windows = self._lift(target, h_all[:, : cs - 1 + L]).unfold(1, cs, 1)  # (B, L, r, cs)
            return torch.einsum("hcw,blrw->blhcr", conv_weight, windows)

        # 各トークンを読む前の状態 S_{t-1} (B, L, H, d_k, d_v)
        S = S_start
        states = []
        for t in range(L):
            states.append(S)
            decayed = alpha[:, t, :, None, None] * S
            old_value = torch.einsum("bhkv,bhk->bhv", decayed, k[:, t])
            S = decayed + torch.einsum(
                "bhk,bhv->bhkv", k[:, t], beta[:, t, :, None] * (v[:, t] - old_value)
            )
        states = torch.stack(states, dim=1)
        # 後ろのトークンのヤコビアンの積 G_t (B, L, H, d_k, d_k) と、全トークンの積 G_{-1}
        G = torch.eye(dk, device=k.device, dtype=k.dtype).expand(B, H, dk, dk)
        suffix = [G] * L
        for t in reversed(range(L)):
            suffix[t] = G
            Gk = torch.einsum("bhij,bhj->bhi", G, k[:, t])
            G = alpha[:, t, :, None, None] * (
                G - beta[:, t, :, None, None] * Gk[..., None] * k[:, t, :, None, :]
            )
        G_all = G
        suffix = torch.stack(suffix, dim=1)
        Gk = torch.einsum("blhij,blhj->blhi", suffix, k)

        sk = torch.einsum("blhkv,blhk->blhv", states, k)  # S^T k
        residual = v - alpha[..., None] * sk
        # v: I[i, j, p] = beta k_i silu'(c_v[j]) (畳み込んだ s A_v h)[j, p]
        e_v = silu_grad(c_v)[..., None] * conv_lift("v", conv_v)
        immediate_v = torch.einsum("blhi,blhjp->bhijp", beta[..., None] * Gk, e_v)
        # a: I = (I - beta k k^T) S_{t-1} * d alpha / d pre * s A_a h_t
        d_alpha = alpha * neg_A * torch.sigmoid(a_pre[:, :L])
        P = states - beta[..., None, None] * k[..., None] * sk[:, :, :, None, :]
        phi_a = d_alpha[..., None] * self._lift("a", h_all[:, cs - 1 : cs - 1 + L])[:, :, None, :]
        immediate_a = torch.einsum(
            "blhij,blhp->bhijp", torch.einsum("blhik,blhkj->blhij", suffix, P), phi_a
        )
        # b: I = k r^T * beta (1 - beta) * s A_b h_t
        phi_b = (beta * (1.0 - beta))[..., None] * self._lift("b", h_all[:, cs - 1 : cs - 1 + L])[
            :, :, None, :
        ]
        immediate_b = torch.einsum(
            "blhi,blhjp->bhijp", Gk, residual[..., None] * phi_b[:, :, :, None, :]
        )
        for target, immediate in (("v", immediate_v), ("a", immediate_a), ("b", immediate_b)):
            trace[target] = torch.einsum("bhik,bhkjp->bhijp", G_all, trace[target]) + immediate

        # k（対角の近似）: S の行 i は B_k の行 (h, i) にだけよるとし、alpha (1 - beta k_i^2) で運ぶ。
        # dk = Q dx、Q = n (I - k k^T)、n = 1 / |x|、x = silu(c_k)
        x_k = F.silu(c_k)
        n = torch.rsqrt((x_k * x_k).sum(dim=-1, keepdim=True) + 1e-6)  # (B, L, H, 1)
        Q_diag = n * (1.0 - k * k)  # (B, L, H, d_k)
        # (S^T Q)[j, i] = n (S[i, j] - k_i (S^T k)[j])
        SQ = n[..., None] * (states - k[..., None] * sk[:, :, :, None, :])
        coef = (
            beta[..., None, None] * residual[:, :, :, None, :] * Q_diag[..., None]
            - (alpha * beta)[..., None, None] * k[..., None] * SQ
        )
        f = silu_grad(c_k)[..., None] * conv_lift("k", conv_k)  # (B, L, H, d_k, r)
        # 後ろのトークンの対角の積（そのトークンを含まない）と、全トークンの積
        d = alpha[..., None] * (1.0 - beta[..., None] * k * k)  # (B, L, H, d_k)
        inclusive = torch.flip(torch.cumprod(torch.flip(d, [1]), dim=1), [1])
        exclusive = torch.cat([inclusive[:, 1:], torch.ones_like(d[:, :1])], dim=1)
        trace["k"] = inclusive[:, 0, :, :, None, None] * trace["k"] + torch.einsum(
            "blhij,blhip->bhijp", exclusive[..., None] * coef, f
        )
