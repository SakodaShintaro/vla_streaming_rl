# SPDX-License-Identifier: MIT
"""Wan 2.1 VAE (vendored).

Image latent encoder used by ``networks.modules.image_processor.ImageProcessor``.
The VAE model lives in ``vae.py`` (Alibaba Wan Team); ``WanVAEWrapper`` below is
the thin encode/decode glue.
"""

import torch
from huggingface_hub import hf_hub_download

from .vae import _video_vae

WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        vae_path = hf_hub_download(WAN_REPO_ID, "Wan2.1_VAE.pth")
        self.model = (
            _video_vae(
                pretrained_path=vae_path,
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

    def _scale(self, device, dtype):
        return [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        scale = self._scale(pixel.device, pixel.dtype)
        output = torch.stack(
            [
                self.model.encode(u.unsqueeze(0), scale, cache=self.model.make_encoder_cache())
                .float()
                .squeeze(0)
                for u in pixel
            ],
            dim=0,
        )
        # [B, C, T, H, W] -> [B, T, C, H, W]
        return output.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        # [B, T, C, H, W] -> [B, C, T, H, W]
        zs = latent.permute(0, 2, 1, 3, 4)
        scale = self._scale(latent.device, latent.dtype)
        output = torch.stack(
            [self.model.decode(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0) for u in zs],
            dim=0,
        )
        # [B, C, T, H, W] -> [B, T, C, H, W]
        return output.permute(0, 2, 1, 3, 4)
