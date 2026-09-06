# SPDX-License-Identifier: MIT
import torch
from torch import nn
from torch.nn import functional as F

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
    """The observation laid flat on one space axis, mixed across space and time.

    Every input the network has becomes tokens on a single axis -- the patch
    grid, the previous action a token per component, the reward, the
    interoceptive scalars, a register, and the activations a chain of thought
    issued this step -- and :class:`SpatialTemporalTransformer` mixes them across
    that axis and across the window. The last step's whole axis is the state the
    heads read.

    The chain's budget is fixed per step, so ``space_len`` is constant and the
    recurrent state keeps its shape however the chain restarts.
    ``cot_tokens_num = 0`` leaves the axis without chain tokens, which is the
    ablation a chain-carrying run is measured against.

    The chain arrives with a depth axis -- one hidden state, or every one behind
    it -- and a learned softmax over that axis picks which depth to read, the way
    the VLM network weights its own. ``cot_pool`` then decides how the step's
    tokens reach the space axis. The temporal block runs one recurrence per space
    position, so slot i of an unpooled chain holds tokens i, i+L, i+2L ... of one
    continuous chain: a stride-L phase whose alignment shifts every restart,
    which is not the correspondence the other tokens have. ``mean`` collapses the
    step to a single slot whose recurrence follows the chain itself, and also
    frees ``space_len`` -- and with it the recurrent state's shape -- from L.
    """

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
        cot_layers: int,
        cot_dim: int,
        cot_pool: str,
        cot_steps_per_chain: int,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        assert cot_tokens_num >= 0, f"cot_tokens_num must not be negative; got {cot_tokens_num}"
        assert cot_pool in ("mean", "none"), f"Unknown cot_pool: {cot_pool}"
        assert cot_steps_per_chain >= 1, cot_steps_per_chain
        # Pooling leaves one slot per step; without it every token gets its own.
        # With no chain at all there is nothing to pool, and averaging an empty
        # axis would invent a slot full of NaN.
        self.pool_cot = cot_pool == "mean" and cot_tokens_num > 0
        cot_slots = 1 if self.pool_cot else cot_tokens_num
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
            + cot_slots  # this step's chain of thought
        )

        # The VLM's hidden width down to the encoder's: the only trainable thing
        # on the chain's path, since the VLM itself never learns.
        # Input-independent logits over the chain's depth axis; the softmax over
        # them is the weighting that picks which depth the encoder reads.
        self.cot_layer_logits = nn.Parameter(torch.zeros(cot_layers))
        self.cot_proj = nn.Linear(cot_dim, self.hidden_image_dim)
        # How old the chain is, added onto the chain's own tokens rather than
        # given a token of its own: the same activations mean one thing on the
        # frame they were written about and another fifteen steps later, and
        # what has to know that is the slot carrying them. Zero-initialized, so
        # a run starts from exactly the chain embedding it had before and moves
        # off it only if the age turns out to matter.
        self.cot_age_embed = nn.Embedding(cot_steps_per_chain, self.hidden_image_dim)
        nn.init.zeros_(self.cot_age_embed.weight)

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
        images: torch.Tensor,  # (B, T, 3, H, W)
        actions: torch.Tensor,  # (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (B, space_len, state_size, n_layer)
        scalar_obs: torch.Tensor,  # (B, T, scalar_obs_dim)
        cot_activations: torch.Tensor,  # (B, T, cot_tokens_num, cot_layers, cot_dim)
        cot_age: torch.Tensor,  # (B, T, 1)
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
        weights = F.softmax(self.cot_layer_logits, dim=0)
        cot = (cot_activations.to(image_embed.dtype) * weights.view(1, 1, 1, -1, 1)).sum(dim=3)
        cot = cot.mean(dim=2, keepdim=True) if self.pool_cot else cot
        # [B, T, 1, C'], broadcast over the slots: every token of one step's
        # chain is that step's chain, so they share its age.
        age_embed = self.cot_age_embed(cot_age.squeeze(-1).long()).unsqueeze(2)
        cot_embed = self.cot_proj(cot) + age_embed  # [B, T, cot_slots, C']

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
