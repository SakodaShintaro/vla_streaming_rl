# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from transformers import SiglipVisionModel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from vla_streaming_rl.networks.modules.lora_gdn import LORA_TARGETS, LoRAGDNLayer
from vla_streaming_rl.networks.modules.reward_embed import embed_reward
from vla_streaming_rl.networks.modules.self_attention import (
    SpatialTransformerBlock,
    get_fourier_embeds_from_coordinates,
)
from vla_streaming_rl.networks.modules.vlm_encoder import STATE_SCALAR_INDICES

# SigLIP2 の事前学習の画像の正規化（[-1, 1] にする）
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class SiglipEncoder(nn.Module):
    """観測の画像を1枚ずつ凍結した SigLIP2 の画像の塔に読ませ、最後の層のパッチの特徴をそのまま
    1ステップ分の状態にするエンコーダ。

    ``represent`` は画像を ``resolution`` 四方に縮めて SigLIP2 に通し、最後の層（正規化の後）の
    パッチのトークンの格子を ``token_pool`` x ``token_pool`` ずつ平均してバッファに保存する。
    SigLIP2 は学習しないので、バッファの表現は古くならない。

    ``to_state`` は保存した表現を位置ごとに ``token_dim`` へ射影し、同じステップの前の行動・
    報酬・スカラーの観測をそれぞれ ``token_dim`` 幅のトークンにしたものを並べ、
    ``token_mixer_layers`` 層の Transformer でトークンどうしを混ぜて、ヘッドが読む状態にする
    （VLM のエンコーダの ``to_state`` と同じ）。``num_gdn_layers`` が 0 なら時間方向には何もしない。

    ``num_gdn_layers`` が 0 より大きければ、ステップのトークンを学習したクエリの cross-attention
    （Perceiver / Set Transformer のプーリング、``PerceiverPooling``）で ``gdn_memory_tokens`` 個の
    潜在のトークンにまとめ、乱数で初期化した GDN の層（Qwen3.5 の linear_attention の層と同じ形、
    幅は ``token_dim``）にその列を読ませて、再帰の状態をステップをまたいで持ち越す。GDN の層の
    出力を記憶のトークンとしてステップのトークンの後ろに付け足し、混ぜる層に通す。元の重みと
    プーリングはリプレイの窓の BPTT で、LoRA は ``stream_step`` の出力の損失で RTRL で学ぶ
    （``LoRAGDNLayer``）。
    """

    def __init__(
        self,
        image_shapes: list[tuple[int, ...]],
        action_dim: int,
        siglip_config: DictConfig,
        device: str,
    ) -> None:
        super().__init__()
        self.device = device
        self.resolution = siglip_config.resolution
        self.model = SiglipVisionModel.from_pretrained(siglip_config.model_id).to(device)
        # SigLIP2 の位置の埋め込みは事前学習の解像度の格子のものなので、解像度はそれに合わせる
        assert self.resolution == self.model.config.image_size, (
            self.resolution,
            self.model.config.image_size,
        )
        self.model.eval().requires_grad_(False)
        self.register_buffer(
            "pixel_mean", torch.tensor(SIGLIP_MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer("pixel_std", torch.tensor(SIGLIP_STD, device=device).view(1, 3, 1, 1))
        # パッチの格子は resolution / patch_size 四方で、それを token_pool 四方ずつ平均する
        self.token_pool = siglip_config.token_pool
        patch_size = self.model.config.patch_size
        assert self.resolution % (patch_size * self.token_pool) == 0, self.resolution
        self.grid = self.resolution // patch_size
        self.images_num = len(image_shapes)
        self.num_positions = self.images_num * (self.grid // self.token_pool) ** 2
        self.feature_dim = self.model.config.hidden_size

        self.token_dim = siglip_config.token_dim
        self.state_out_proj = nn.Linear(self.feature_dim, self.token_dim).to(device)
        # スカラーをトークンにする sin/cos の角周波数。周期を scalar_max_period から
        # scalar_min_period まで対数等間隔にとり、sin と cos で token_dim 幅にする
        assert self.token_dim % 2 == 0, self.token_dim
        periods = torch.logspace(
            math.log10(siglip_config.scalar_max_period),
            math.log10(siglip_config.scalar_min_period),
            self.token_dim // 2,
        )
        self.register_buffer("scalar_frequencies", (2.0 * math.pi / periods).to(device))
        # 表現の位置、前の行動の成分ごと、報酬、スカラーの観測ごとに1トークン
        tokens_num = self.num_positions + action_dim + 1 + len(STATE_SCALAR_INDICES)
        self.num_gdn_layers = siglip_config.num_gdn_layers
        assert self.num_gdn_layers >= 0, self.num_gdn_layers
        # GDN の層があれば、その出力の記憶のトークンを混ぜる層の入力に付け足す
        self.memory_tokens = siglip_config.gdn_memory_tokens if self.num_gdn_layers > 0 else 0
        self.output_dim = (tokens_num + self.memory_tokens) * self.token_dim
        # 位置の埋め込みは 0、残差は layer scale で小さく始める
        self.token_position = nn.Parameter(
            torch.zeros(1, tokens_num, self.token_dim, device=device)
        )
        self.token_mixer = nn.Sequential(
            *[
                SpatialTransformerBlock(
                    self.token_dim,
                    siglip_config.token_mixer_heads,
                    tokens_num + self.memory_tokens,
                    siglip_config.token_mixer_layer_scale_init,
                )
                for _ in range(siglip_config.token_mixer_layers)
            ]
        ).to(device)

        if self.num_gdn_layers > 0:
            self.pooling = PerceiverPooling(
                self.token_dim, siglip_config.token_mixer_heads, self.memory_tokens
            ).to(device)
            # 1ステップ（記憶のトークンの数だけのトークン）を読む間の減衰を、ヘッドごとに
            # gdn_step_decay_min から gdn_step_decay_max まで散らす
            step_decays = torch.logspace(
                math.log10(-math.log(siglip_config.gdn_step_decay_min)),
                math.log10(-math.log(siglip_config.gdn_step_decay_max)),
                siglip_config.gdn_heads,
            )
            gdn_config = Qwen3_5TextConfig(
                hidden_size=self.token_dim,
                intermediate_size=siglip_config.gdn_mlp_dim,
                num_hidden_layers=self.num_gdn_layers,
                layer_types=["linear_attention"] * self.num_gdn_layers,
                linear_num_key_heads=siglip_config.gdn_heads,
                linear_num_value_heads=siglip_config.gdn_heads,
                linear_key_head_dim=siglip_config.gdn_head_dim,
                linear_value_head_dim=siglip_config.gdn_head_dim,
                linear_conv_kernel_dim=4,
                hidden_act="silu",
                rms_norm_eps=1e-6,
            )
            self.gdn_layers = nn.ModuleList(
                [
                    LoRAGDNLayer(
                        self._init_layer(
                            Qwen3_5DecoderLayer(gdn_config, index),
                            step_decays,
                            self.memory_tokens,
                        ),
                        LORA_TARGETS,
                        siglip_config.gdn_lora_rank,
                        siglip_config.gdn_lora_alpha,
                        True,
                    )
                    for index in range(self.num_gdn_layers)
                ]
            ).to(device)
            # 流れの再帰の状態（最新のティックを読んだ後）。その感度は各層が持つ
            self.gdn_state = self.initial_gdn_state(1)

    @staticmethod
    def _init_layer(
        layer: Qwen3_5DecoderLayer, step_decays: torch.Tensor, tokens_per_step: int
    ) -> Qwen3_5DecoderLayer:
        """乱数で作った層の初期化を2つ変える。

        残差に足す2つの射影（再帰の出力の out と MLP の down）を 0 にし、学習の始めは入力を
        そのまま通して、乱数の重みと過去のステップで決まる成分をトークンに足さない。

        Qwen3.5 の初期化の減衰（A が最大 16）は言語のトークン1つずつに合わせた強さで、
        トークン1つで状態がほぼ消える。a の射影が 0 のときの1ステップ（``tokens_per_step``
        トークン）分の減衰の log が、ヘッドごとに ``-step_decays`` (heads,) になるよう A を小さく
        する。"""
        attention = layer.linear_attn
        with torch.no_grad():
            attention.out_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
            per_token = step_decays.to(attention.A_log) / tokens_per_step
            attention.A_log.copy_((per_token / F.softplus(attention.dt_bias)).log())
        return layer

    def gdn_state_size(self) -> int:
        """GDN の層の再帰の状態を並べた大きさ（層ごとの S と、畳み込みが読む直前の入力）。"""
        return sum(layer.recurrent_size() for layer in self.gdn_layers)

    def initial_gdn_state(self, batch: int) -> torch.Tensor:
        """流れの始めの、空の再帰の状態 (batch, gdn_state_size)。"""
        return torch.zeros(batch, self.gdn_state_size(), device=self.device)

    def lora_parameters(self) -> list[nn.Parameter]:
        """streaming の更新で RTRL で学ぶ、GDN の層の LoRA のパラメータ。"""
        return [
            parameter
            for layer in self.gdn_layers
            for parameter in [*layer.lora_A.parameters(), *layer.lora_B.parameters()]
        ]

    def representation_shape(self) -> tuple[int, ...]:
        """``represent`` が返し、バッファに保存する表現の形。"""
        return (self.num_positions, self.feature_dim)

    @torch.no_grad()
    def represent(self, images: list[torch.Tensor]) -> torch.Tensor:
        """このティックの [0, 1] の画像 (3, H, W) の並びの表現 (num_positions, feature_dim)。
        画像ごとに、パッチの格子を平均でまとめたものを、画像の順に、それぞれラスター順に並べる。"""
        assert len(images) == self.images_num, (len(images), self.images_num)
        pixels = torch.stack(images).to(self.device, torch.float32)
        pixels = F.interpolate(
            pixels,
            size=(self.resolution, self.resolution),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.pixel_mean) / self.pixel_std
        # SigLIP2 は CLS を持たず、全てがラスター順のパッチ
        patches = self.model(pixel_values=pixels).last_hidden_state
        grid = patches.view(self.images_num, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        grid = F.avg_pool2d(grid, self.token_pool)
        return grid.flatten(2).transpose(1, 2).reshape(self.num_positions, self.feature_dim)

    def to_state(
        self,
        image: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """保存した表現 (B, num_positions, feature_dim) を射影したものに、同じステップの前の
        行動 (B, action_dim)、報酬 (B, 1)、スカラーの観測 (B, SCALAR_OBS_DIM) をそれぞれ
        トークンにしたものを並べ、ヘッドが読む状態 (B, output_dim) にする。スカラーは周期の違う
        sin/cos にし、小さい違いも大きい値の位置も別のベクトルにする。"""
        return self.mix(self.step_tokens(image, actions, rewards, scalar_obs))

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        """ステップのトークン (B, tokens_num, token_dim)（GDN の層があれば記憶のトークンを
        付け足したもの）を混ぜる層に通し、ヘッドが読む状態 (B, output_dim) にする。"""
        return self.token_mixer(tokens).flatten(start_dim=1)

    def step_tokens(
        self,
        image: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        scalar_obs: torch.Tensor,
    ) -> torch.Tensor:
        """``to_state`` の、混ぜる層に入る前のトークン (B, tokens_num, token_dim)。位置の埋め込みを
        足したもの。"""
        projected = self.state_out_proj(image.to(self.device, torch.float32))
        device = projected.device
        action_tokens = get_fourier_embeds_from_coordinates(
            self.token_dim, actions.to(device, torch.float32).unsqueeze(1)
        )[:, 0]  # (B, action_dim, token_dim)
        reward_token = embed_reward(rewards.to(device, torch.float32), self.token_dim)
        scalars = scalar_obs.to(device, torch.float32)[:, STATE_SCALAR_INDICES]
        angles = scalars.unsqueeze(-1) * self.scalar_frequencies
        scalar_tokens = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        tokens = torch.cat([projected, action_tokens, reward_token, scalar_tokens], dim=1)
        return tokens + self.token_position

    def gdn_step(
        self, latents: torch.Tensor, state: torch.Tensor, use_trace: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """1ステップ分の潜在のトークン (B, memory_tokens, token_dim)（``pool`` の出力）を、前の
        再帰の状態 (B, gdn_state_size) から GDN の層に下から順に読ませ、最も上の層の出力
        (B, memory_tokens, token_dim) と次の状態を返す。``use_trace`` は streaming で読むか
        （``LoRAGDNLayer.forward`` を参照）。"""
        x = latents
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        layer_states = state.split([layer.recurrent_size() for layer in self.gdn_layers], dim=1)
        next_states = []
        for layer, layer_state in zip(self.gdn_layers, layer_states, strict=True):
            state_size = layer.num_heads * layer.key_dim * layer.value_dim
            S, h = layer_state.split([state_size, layer_state.shape[1] - state_size], dim=1)
            x, S, h = layer(
                x,
                valid,
                S.view(-1, layer.num_heads, layer.key_dim, layer.value_dim),
                h.view(-1, layer.conv_size - 1, layer.hidden_size),
                use_trace,
            )
            next_states += [S.flatten(start_dim=1), h.flatten(start_dim=1)]
        return x, torch.cat(next_states, dim=1)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """ステップのトークン (B, tokens_num, token_dim) を潜在のトークン
        (B, memory_tokens, token_dim) にまとめる。"""
        return self.pooling(tokens)

    def read_step(
        self, tokens: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """リプレイの窓の1ステップ分のトークン (B, tokens_num, token_dim) を、前の再帰の状態
        から読み、記憶のトークンを付け足した混ぜる層の入力 (B, tokens_num + memory_tokens,
        token_dim) と次の状態を返す。"""
        memory, next_state = self.gdn_step(self.pool(tokens), state, False)
        return torch.cat([tokens, memory], dim=1), next_state

    def stream_step(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        scalar_obs: torch.Tensor,
        reset_state: bool,
        use_trace: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """このティックの特徴 (num_positions, feature_dim) と前の行動・報酬・スカラーの観測を流れに
        積む。``reset_state`` なら先に再帰の状態と感度を空にする。読む前の再帰の状態
        (gdn_state_size,) と、記憶のトークンを付け足した混ぜる層の入力
        (tokens_num + memory_tokens, token_dim) を返す。``use_trace`` なら RTRL の感度を進め、
        記憶のトークンは次のティックの損失が勾配を LoRA まで流せるよう計算グラフ付き（呼ぶ側が
        勾配を有効にする）。トークンと潜在のトークンは LoRA だけを学ぶように勾配なしで作る。"""
        if reset_state:
            # 空の状態はパラメータによらないので、感度も 0 に戻す
            self.gdn_state = self.initial_gdn_state(1)
            for layer in self.gdn_layers:
                layer.reset_trace()
        with torch.no_grad():
            tokens = self.step_tokens(
                features[None], action[None], reward.view(1, 1), scalar_obs[None]
            )
            latents = self.pool(tokens)
        state_before = self.gdn_state[0]
        memory, self.gdn_state = self.gdn_step(latents, self.gdn_state, use_trace)
        return state_before, torch.cat([tokens, memory], dim=1)[0]


class PerceiverPooling(nn.Module):
    """学習したクエリ ``num_queries`` 個の cross-attention でトークンの集合を同じ数の潜在の
    トークンにまとめる（Perceiver の cross-attention、Set Transformer の PMA）。pre-norm の
    注意と MLP を残差でつなぐ。"""

    def __init__(self, dim: int, heads: int, num_queries: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """トークン (B, N, dim) を潜在のトークン (B, num_queries, dim) にまとめる。"""
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        inputs = self.input_norm(tokens)
        x = (
            queries
            + self.attention(self.query_norm(queries), inputs, inputs, need_weights=False)[0]
        )
        return x + self.mlp(self.mlp_norm(x))
