# SPDX-License-Identifier: MIT
import torch
from torch import nn

from vla_streaming_rl.self_forcing.utils.wan_wrapper import WanVAEWrapper


class ImageProcessor(nn.Module):
    def __init__(self, observation_space_shape: tuple[int]) -> None:
        super().__init__()
        self.observation_space_shape = observation_space_shape
        self.processor = WanVAEWrapper().eval().requires_grad_(False)
        x = torch.zeros(1, *observation_space_shape)
        self.output_shape = list(self.encode(x).size())[1:]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1] -> Wan expects (B, 3, T=1, H, W) in [-1, 1]
        z = self.processor.encode_to_latent((x * 2.0 - 1.0).unsqueeze(2))
        self.processor.model.clear_cache()
        return z.squeeze(1)  # (B, 16, H/8, W/8)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 16, H_lat, W_lat) -> (B, 3, H, W) in [0, 1]
        x = x.view(x.size(0), *self.output_shape)
        pix = self.processor.decode_to_pixel(x.unsqueeze(1)).squeeze(1)
        self.processor.model.clear_cache()
        return (pix * 0.5 + 0.5).clamp(0, 1)
