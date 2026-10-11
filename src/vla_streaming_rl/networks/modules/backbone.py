# SPDX-License-Identifier: MIT
import torch
from torch import nn

from .reward_processor import RewardProcessor
from .self_attention import get_fourier_embeds_from_coordinates
from .spatial_temporal_transformer import SpatialTemporalTransformer


class SpatialTemporalEncoder(nn.Module):
    """The observation laid flat on one space axis, mixed across space and time.

    Every input the network has becomes tokens on a single axis -- the patch
    grid, the previous action a token per component, the reward, the
    interoceptive scalars and a register -- and
    :class:`SpatialTemporalTransformer` mixes them across that axis and across
    the window. The last step's whole axis is the state the heads read.
    """

    def __init__(
        self,
        image_features_shape: tuple[int, int, int],
        image_latent_dim: int,
        reward_processor: RewardProcessor,
        seq_len: int,
        n_layer: int,
        action_dim: int,
        scalar_obs_dim: int,
        temporal_model_type: str,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.reward_processor = reward_processor

        # The frozen encoder's (C, H, W) grid, projected here to the token width
        # and read as [B, H * W, C'] tokens.
        features_dim, self.hidden_h, self.hidden_w = image_features_shape
        self.hidden_image_dim = image_latent_dim
        self.image_projection = nn.Conv2d(features_dim, image_latent_dim, kernel_size=1)
        self.image_tokens_num = self.hidden_h * self.hidden_w

        self.space_len = (
            self.image_tokens_num  # patch grid
            + action_dim  # previous action, one token per component
            + 1  # reward
            + scalar_obs_dim  # interoceptive scalars
            + 1  # register
        )

        self.spatial_temporal = SpatialTemporalTransformer(
            n_layer=n_layer,
            space_len=self.space_len,
            tempo_len=seq_len,
            hidden_dim=self.hidden_image_dim,
            n_head=1,
            temporal_model_type=temporal_model_type,
            layer_scale_init=layer_scale_init,
        )

        self.output_dim = self.hidden_image_dim * self.space_len

    def init_state(self) -> torch.Tensor:
        state_size = self.spatial_temporal.spatial_temporal_blocks[
            0
        ].tempo_block.get_rnn_state_size()
        # [1, space_len, state_size, n_layer] (batch size 1)
        return torch.zeros(1, self.space_len, state_size, self.n_layer)

    def forward(
        self,
        images: torch.Tensor,  # (B, T, *image_features_shape): the frozen encoder's output per frame
        actions: torch.Tensor,  # (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (B, space_len, state_size, n_layer)
        scalar_obs: torch.Tensor,  # (B, T, scalar_obs_dim)
        token_keep: torch.Tensor,  # (B, T, space_len) bool: which tokens are kept
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            encoded features: (B, output_dim)
            rnn_state: (B, space_len, state_size, n_layer)
        """
        B, T = images.shape[:2]

        # External [B, space_len, state_size, n_layer] -> internal [1, B*space_len, ...]
        rnn_state_internal = rnn_state.reshape(1, B * self.space_len, -1, self.n_layer)

        all_frames = images.reshape(-1, *images.shape[2:])
        all_latents = self.image_projection(all_frames)  # [B*T, C', H', W']
        all_latents = all_latents.reshape(B, T, *all_latents.shape[1:])
        image_embed = all_latents.flatten(3).transpose(2, 3)  # [B, T, S, C']

        action_embed = get_fourier_embeds_from_coordinates(self.hidden_image_dim, actions)
        reward_embed = self.reward_processor.encode(rewards)  # [B, T, 1, C']
        scalar_obs_embed = get_fourier_embeds_from_coordinates(self.hidden_image_dim, scalar_obs)
        register_token = torch.zeros(
            (B, T, 1, self.hidden_image_dim), device=images.device, dtype=images.dtype
        )

        all_embed = torch.cat(
            [
                image_embed,
                action_embed,
                reward_embed,
                scalar_obs_embed,
                register_token,
            ],
            dim=2,
        )
        all_embed = all_embed * token_keep.unsqueeze(-1).to(all_embed.dtype)

        spatial_temporal_output, rnn_state_internal = self.spatial_temporal(
            all_embed, rnn_state_internal
        )
        rnn_state = rnn_state_internal.reshape(B, self.space_len, -1, self.n_layer)

        # The last step's whole space axis is the state.
        return spatial_temporal_output[:, -1, :, :].flatten(start_dim=1), rnn_state
