# SPDX-License-Identifier: MIT
"""The spatio-temporal encoder of the chain-of-thought network.

The same body as :class:`SpatialTemporalEncoder` -- the patch grid, the previous
action, the reward, the interoceptive scalars and a register token laid flat on
the space axis and run through :class:`SpatialTemporalTransformer` -- with the
activations the chain of thought issued this step appended to that axis. The
per-step token budget is fixed, so ``space_len`` is constant and the recurrent
state keeps its shape however the chain restarts.

The state the heads read is the whole space axis at the last step, chain tokens
included. Dropping those tokens is exactly ``actor_critic_with_action_value``,
which is therefore the ablation this network is measured against.
"""

import torch
from torch import nn

from .image_processor import ImageProcessor
from .reward_processor import RewardProcessor
from .self_attention import get_fourier_embeds_from_coordinates
from .spatial_temporal_transformer import SpatialTemporalTransformer


class CoTEncoder(nn.Module):
    def __init__(
        self,
        image_processor: ImageProcessor,
        reward_processor: RewardProcessor,
        seq_len: int,
        n_layer: int,
        action_dim: int,
        scalar_obs_dim: int,
        temporal_model_type: str,
        cot_tokens_num: int,
        cot_dim: int,
    ) -> None:
        super().__init__()
        assert cot_tokens_num >= 1, (
            f"cot_tokens_num {cot_tokens_num} leaves no chain-of-thought tokens; "
            "the no-chain case is network_class=actor_critic_with_action_value"
        )
        self.n_layer = n_layer
        self.image_processor = image_processor
        self.reward_processor = reward_processor
        self.cot_tokens_num = cot_tokens_num

        # image_processor outputs [B, C, H, W] -> treat as [B, H * W, C]
        self.hidden_image_dim = self.image_processor.output_shape[0]
        self.hidden_h = self.image_processor.output_shape[1]
        self.hidden_w = self.image_processor.output_shape[2]
        self.image_tokens_num = self.hidden_h * self.hidden_w

        self.space_len = (
            self.image_tokens_num  # patch grid
            + action_dim  # previous action, one token per component
            + 1  # reward
            + scalar_obs_dim  # interoceptive scalars
            + 1  # register
            + cot_tokens_num  # this step's chain-of-thought activations
        )

        # The VLM's hidden width down to the encoder's: the only trainable thing
        # on the chain's path, since the VLM itself never learns.
        self.cot_proj = nn.Linear(cot_dim, self.hidden_image_dim)

        self.spatial_temporal = SpatialTemporalTransformer(
            n_layer=n_layer,
            space_len=self.space_len,
            tempo_len=seq_len,
            hidden_dim=self.hidden_image_dim,
            n_head=1,
            temporal_model_type=temporal_model_type,
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
        images: torch.Tensor,  # (B, T, 3, H, W)
        actions: torch.Tensor,  # (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (B, space_len, state_size, n_layer)
        scalar_obs: torch.Tensor,  # (B, T, scalar_obs_dim)
        cot_activations: torch.Tensor,  # (B, T, cot_tokens_num, cot_dim)
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
        all_latents = self.image_processor.encode(all_frames)  # [B*T, C', H', W']
        all_latents = all_latents.reshape(B, T, *all_latents.shape[1:])
        image_embed = all_latents.flatten(3).transpose(2, 3)  # [B, T, S, C']

        action_embed = get_fourier_embeds_from_coordinates(self.hidden_image_dim, actions)
        reward_embed = self.reward_processor.encode(rewards)  # [B, T, 1, C']
        scalar_obs_embed = get_fourier_embeds_from_coordinates(self.hidden_image_dim, scalar_obs)
        register_token = torch.zeros(
            (B, T, 1, self.hidden_image_dim), device=images.device, dtype=images.dtype
        )
        cot_embed = self.cot_proj(cot_activations.to(image_embed.dtype))  # [B, T, L, C']

        all_embed = torch.cat(
            [image_embed, action_embed, reward_embed, scalar_obs_embed, register_token, cot_embed],
            dim=2,
        )

        spatial_temporal_output, rnn_state_internal = self.spatial_temporal(
            all_embed, rnn_state_internal
        )
        rnn_state = rnn_state_internal.reshape(B, self.space_len, -1, self.n_layer)

        # The last step's whole space axis is the state.
        return spatial_temporal_output[:, -1, :, :].flatten(start_dim=1), rnn_state
