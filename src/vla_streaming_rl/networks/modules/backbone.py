# SPDX-License-Identifier: MIT
import torch
from torch import nn

from .reward_embed import embed_reward
from .self_attention import get_fourier_embeds_from_coordinates
from .spatial_temporal_transformer import SpatialTemporalTransformer


class SpatialTemporalEncoder(nn.Module):
    """The observation laid flat on one space axis, mixed across space and time.

    Every input the network has becomes tokens on a single axis -- the patch
    grid of every image, one after another, the previous action a token per
    component, the reward, the interoceptive scalars and the registers -- and
    :class:`SpatialTemporalTransformer` mixes them across that axis and, for
    the registers alone, across the window. The last step's whole axis is the
    state the heads read.
    """

    def __init__(
        self,
        image_features_shapes: list[tuple[int, int, int]],
        image_latent_dim: int,
        seq_len: int,
        spatial_block_num: int,
        temporal_block_num: int,
        action_dim: int,
        scalar_obs_dim: int,
        register_num: int,
        temporal_model_type: str,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.temporal_block_num = temporal_block_num
        # 時間方向の記憶を持つのは register だけ。他のトークンは各ステップの中で混ざるだけで、
        # 過去のステップは読まない
        self.register_num = register_num

        # The frozen encoder's (C, H, W) grid of each image, projected here to the
        # token width and read as [B, H * W, C'] tokens. 画像はどれも同じエンコーダの
        # 出力なのでチャネル数が揃い、射影を共有する。どの画像のトークンかは位置の埋め込みが
        # 見分ける
        self.image_features_shapes = [tuple(shape) for shape in image_features_shapes]
        features_dims = {shape[0] for shape in self.image_features_shapes}
        assert len(features_dims) == 1, self.image_features_shapes
        self.hidden_image_dim = image_latent_dim
        self.image_projection = nn.Conv2d(features_dims.pop(), image_latent_dim, kernel_size=1)
        # 保存した形では、画像ごとの (C, H, W) を平らにして並べてある
        self.image_features_sizes = [c * h * w for c, h, w in self.image_features_shapes]
        self.image_tokens_num = sum(h * w for _, h, w in self.image_features_shapes)

        self.space_len = (
            self.image_tokens_num  # patch grids
            + action_dim  # previous action, one token per component
            + 1  # reward
            + scalar_obs_dim  # interoceptive scalars
            + register_num  # registers, the only tokens mixed across time
        )

        self.spatial_temporal = SpatialTemporalTransformer(
            spatial_block_num=spatial_block_num,
            temporal_block_num=temporal_block_num,
            space_len=self.space_len,
            tempo_len=seq_len,
            hidden_dim=self.hidden_image_dim,
            n_head=1,
            temporal_model_type=temporal_model_type,
            register_num=register_num,
            layer_scale_init=layer_scale_init,
        )
        self.spatial_temporal.compile(mode="reduce-overhead")

        # 残差の大きさはトークンの種類ごとに大きく違うので、ヘッドに渡す前にトークンごとに揃える
        self.output_norm = nn.LayerNorm(self.hidden_image_dim)
        self.output_dim = self.hidden_image_dim * self.space_len

    def init_state(self) -> torch.Tensor:
        state_size = self.spatial_temporal.temporal_blocks[0].get_rnn_state_size()
        # [1, register_num, state_size, temporal_block_num] (batch size 1)
        return torch.zeros(1, self.register_num, state_size, self.temporal_block_num)

    def image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """保存した形の画像 (..., sum(image_features_sizes)) を射影したトークン
        (..., image_tokens_num, C')。画像の順に並べる。"""
        leading = images.shape[:-1]
        flat = images.reshape(-1, images.shape[-1])
        tokens = []
        for (c, h, w), features in zip(
            self.image_features_shapes,
            flat.split(self.image_features_sizes, dim=-1),
            strict=True,
        ):
            latents = self.image_projection(features.reshape(-1, c, h, w))  # [N, C', H, W]
            tokens.append(latents.flatten(2).transpose(1, 2))  # [N, H * W, C']
        return torch.cat(tokens, dim=1).reshape(*leading, self.image_tokens_num, -1)

    def forward(
        self,
        images: torch.Tensor,  # (B, T, sum(image_features_sizes)): the frozen encoder's output
        actions: torch.Tensor,  # (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (B, register_num, state_size, temporal_block_num)
        scalar_tokens: torch.Tensor,  # (B, T, scalar_obs_dim, C'): the scalars already as tokens
        token_keep: torch.Tensor,  # (B, T, space_len) bool: which tokens are kept
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            encoded features: (B, output_dim)
            rnn_state: (B, register_num, state_size, temporal_block_num)
        """
        B, T = images.shape[:2]

        # External [B, register_num, state_size, temporal_block_num]
        # -> internal [1, B*register_num, ...]
        rnn_state_internal = rnn_state.reshape(
            1, B * self.register_num, -1, self.temporal_block_num
        )

        image_embed = self.image_tokens(images)  # [B, T, S, C']

        action_embed = get_fourier_embeds_from_coordinates(self.hidden_image_dim, actions)
        reward_embed = embed_reward(rewards, self.hidden_image_dim)  # [B, T, 1, C']
        register_token = torch.zeros(
            (B, T, self.register_num, self.hidden_image_dim),
            device=images.device,
            dtype=images.dtype,
        )

        all_embed = torch.cat(
            [
                image_embed,
                action_embed,
                reward_embed,
                scalar_tokens.to(image_embed.dtype),
                register_token,
            ],
            dim=2,
        )
        all_embed = all_embed * token_keep.unsqueeze(-1).to(all_embed.dtype)

        spatial_temporal_output, rnn_state_internal = self.spatial_temporal(
            all_embed, rnn_state_internal
        )
        # CUDA グラフの出力は次の再生で上書きされるので、持ち越す状態は複製しておく
        rnn_state = rnn_state_internal.reshape(
            B, self.register_num, -1, self.temporal_block_num
        ).clone()

        # The last step's whole space axis is the state.
        state = self.output_norm(spatial_temporal_output[:, -1, :, :])
        return state.flatten(start_dim=1), rnn_state
