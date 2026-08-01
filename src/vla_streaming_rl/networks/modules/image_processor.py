# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F
from diffusers import AutoencoderTiny
from torch import nn
from transformers import AutoModel

from vla_streaming_rl.wan import WanVAEWrapper

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class TaesdEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return self.vae.encode(x).latents  # (B, 4, H/8, W/8)


class WanEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vae = WanVAEWrapper()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        z = self.vae.encode_to_latent(x.unsqueeze(2))
        self.vae.model.clear_cache()
        return z.squeeze(1)  # (B, 16, H/8, W/8)


class Dinov2Encoder(nn.Module):
    resolution = 224
    patch_size = 14
    grid_size = resolution // patch_size

    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear")
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        tokens = self.model(pixel_values=x).last_hidden_state[:, 1:, :]  # drop CLS token
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size)


class Siglip2Encoder(nn.Module):
    resolution = 224
    patch_size = 16
    grid_size = resolution // patch_size

    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").vision_model

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear")
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        tokens = self.model(pixel_values=x).last_hidden_state  # no CLS token
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size)


class Vjepa2Encoder(nn.Module):
    resolution = 256
    patch_size = 16
    tubelet_size = 2
    grid_size = resolution // patch_size

    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear")
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        clip = x.unsqueeze(2).repeat(1, 1, self.tubelet_size, 1, 1)  # (B, 3, T=tubelet_size, H, W)
        tokens = self.model(pixel_values_videos=clip).last_hidden_state  # (B, grid*grid, C)
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size)


IMAGE_ENCODERS = {
    "taesd": TaesdEncoder,
    "wan": WanEncoder,
    "dinov2": Dinov2Encoder,
    "siglip2": Siglip2Encoder,
    "vjepa2": Vjepa2Encoder,
}


class ImageProcessor(nn.Module):
    def __init__(
        self,
        observation_space_shape: tuple[int],
        image_encoder_type: str,
        image_encoder_output_dim: int,
    ) -> None:
        super().__init__()
        self.observation_space_shape = observation_space_shape
        self.backbone = IMAGE_ENCODERS[image_encoder_type]().eval().requires_grad_(False)
        x = torch.zeros(1, *observation_space_shape)
        backbone_dim = self.backbone.encode(x).size(1)
        if backbone_dim > image_encoder_output_dim:
            self.projection = nn.Conv2d(backbone_dim, image_encoder_output_dim, kernel_size=1)
        else:
            self.projection = nn.Identity()
        self.output_shape = list(self.encode(x).size())[1:]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone.encode(x))  # (B, C, H, W)
