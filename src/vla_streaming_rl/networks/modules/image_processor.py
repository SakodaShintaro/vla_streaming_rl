# SPDX-License-Identifier: MIT
import torch
from diffusers import AutoencoderTiny
from torch import nn

from vla_streaming_rl.wan import WanVAEWrapper

# "taesd" (madebyollin/taesd, 4 latent channels) or "wan" (Wan2.1 VAE, 16 latent channels).
VAE_TYPE = "taesd"


class ImageProcessor(nn.Module):
    def __init__(self, observation_space_shape: tuple[int]) -> None:
        super().__init__()
        self.observation_space_shape = observation_space_shape
        self.processor = _build_vae().eval().requires_grad_(False)
        x = torch.zeros(1, *observation_space_shape)
        self.output_shape = list(self.encode(x).size())[1:]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1] -> VAE expects [-1, 1]
        x = x * 2.0 - 1.0
        if VAE_TYPE == "wan":
            # Wan expects (B, 3, T=1, H, W)
            z = self.processor.encode_to_latent(x.unsqueeze(2))
            self.processor.model.clear_cache()
            return z.squeeze(1)  # (B, 16, H/8, W/8)
        return self.processor.encode(x).latents  # (B, 4, H/8, W/8)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_lat, H_lat, W_lat) -> (B, 3, H, W) in [0, 1]
        x = x.view(x.size(0), *self.output_shape)
        if VAE_TYPE == "wan":
            pix = self.processor.decode_to_pixel(x.unsqueeze(1)).squeeze(1)
            self.processor.model.clear_cache()
        else:
            pix = self.processor.decode(x).sample
        return (pix * 0.5 + 0.5).clamp(0, 1)


def _build_vae() -> nn.Module:
    if VAE_TYPE == "wan":
        return WanVAEWrapper()
    return AutoencoderTiny.from_pretrained("madebyollin/taesd")
