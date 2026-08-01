# SPDX-License-Identifier: MIT
import torch
from torch import nn

from .image_processor import ImageProcessor
from .reward_processor import RewardProcessor
from .self_attention import get_fourier_embeds_from_coordinates
from .spatial_temporal_transformer import SpatialTemporalTransformer


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SpatialTemporalEncoder(nn.Module):
    """Sequence encoder using SpatialTemporalTransformer"""

    def __init__(
        self,
        image_processor: ImageProcessor,
        reward_processor: RewardProcessor,
        seq_len: int,
        n_layer: int,
        action_dim: int,
        scalar_obs_dim: int,
        temporal_model_type: str,
        use_image_only: bool,
    ) -> None:
        super().__init__()

        self.use_image_only = use_image_only
        self.n_layer = n_layer
        self.temporal_model_type = temporal_model_type

        self.image_processor = image_processor
        self.reward_processor = reward_processor

        # image_processor outputs [B, C, H, W] -> treat as [B, H * W, C] (H * W tokens, C channels each)
        self.hidden_image_dim = self.image_processor.output_shape[0]
        self.hidden_h = self.image_processor.output_shape[1]
        self.hidden_w = self.image_processor.output_shape[2]
        self.image_tokens_num = self.hidden_h * self.hidden_w
        action_tokens_num = action_dim
        scalar_obs_tokens_num = scalar_obs_dim
        reward_tokens_num = 1
        register_tokens_num = 1

        self.space_len = (
            self.image_tokens_num
            + action_tokens_num
            + reward_tokens_num
            + scalar_obs_tokens_num
            + register_tokens_num
        )

        self.spatial_temporal = SpatialTemporalTransformer(
            n_layer=n_layer,
            space_len=self.space_len,
            tempo_len=seq_len,
            hidden_dim=self.hidden_image_dim,
            n_head=1,
            temporal_model_type=temporal_model_type,
        )

        token_num = (
            self.image_tokens_num
            if self.use_image_only
            else (
                self.image_tokens_num
                + action_tokens_num
                + reward_tokens_num
                + scalar_obs_tokens_num
                + register_tokens_num
            )
        )

        self.output_dim = self.hidden_image_dim * token_num

    def init_state(self) -> torch.Tensor:
        # Get state_size using block's get_rnn_state_size()
        state_size = self.spatial_temporal.spatial_temporal_blocks[
            0
        ].tempo_block.get_rnn_state_size()
        # [1, space_len, state_size, n_layer] (batch size 1)
        return torch.zeros(1, self.space_len, state_size, self.n_layer)

    def forward(
        self,
        images: torch.Tensor,  # (B, T, 3, H, W)
        actions: torch.Tensor,  #  (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (B, space_len, state_size, n_layer)
        scalar_obs: torch.Tensor,  # (B, T, scalar_obs_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            encoded features: (B, output_dim)
            rnn_state: (B, space_len, state_size, n_layer)
            action_text: str (always empty string for non-VLM encoders)
        """
        # Encode all frames with AE but preserve spatial structure
        # images: (B, T, C, H, W) -> encode all frames
        B, T = images.shape[:2]

        # External format [B, space_len, state_size, n_layer] -> Internal format [1, B*space_len, state_size, n_layer]
        rnn_state_internal = rnn_state.reshape(1, B * self.space_len, -1, self.n_layer)

        # Reshape to process all frames: (B*T, C, H, W)
        all_frames = images.reshape(-1, *images.shape[2:])
        # Encode all frames at once
        all_latents = self.image_processor.encode(all_frames)  # [B*T, C', H', W']
        all_latents = all_latents.reshape(B, T, *all_latents.shape[1:])  # [B, T, C', H', W']
        image_embed = all_latents.flatten(3).transpose(2, 3)  # [B, T, S(=H'*W'), C']

        # [B, T, action_dim] -> [B, T, action_dim, C']
        action_embed = get_fourier_embeds_from_coordinates(self.hidden_image_dim, actions)

        # [B, T, 1] -> [B, T, 1, C']
        reward_embed = self.reward_processor.encode(rewards)  # (B, T, 1, C')

        # [B, T, scalar_obs_dim] -> [B, T, scalar_obs_dim, C']
        scalar_obs_embed = get_fourier_embeds_from_coordinates(self.hidden_image_dim, scalar_obs)

        # [B, T, 1, C']
        register_token = torch.zeros(
            (B, T, 1, self.hidden_image_dim), device=images.device, dtype=images.dtype
        )

        # [B, T, S+action_dim+1+scalar_obs_dim+1, C']
        all_embed = torch.cat(
            [image_embed, action_embed, reward_embed, scalar_obs_embed, register_token], dim=2
        )

        # Apply STT to all frames
        spatial_temporal_output, rnn_state_internal = self.spatial_temporal(
            all_embed, rnn_state_internal
        )  # [B, T, S+action_dim+1, C'], [1, B*space_len, state_size, n_layer]

        # Internal format [1, B*space_len, state_size, n_layer] -> External format [B, space_len, state_size, n_layer]
        rnn_state = rnn_state_internal.reshape(B, self.space_len, -1, self.n_layer)

        # Use last timestep's image tokens for final representation
        last_frame_emb = spatial_temporal_output[:, -1, :, :]  # [B, S+action_dim+1, C']

        if self.use_image_only:
            last_frame_emb = last_frame_emb[:, : self.image_tokens_num, :]  # [B, S, C']

        output = last_frame_emb.flatten(start_dim=1)  # [B, token_num * C']

        return output, rnn_state
