# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F
from diffusers import AutoencoderTiny
from torch import nn
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

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


class QwenImageEncoder(nn.Module):
    """Runs Qwen3.5's ViT on a single image per batch element (sequence length 1).

    No temporal attention (there is only one frame), so this is just the
    original ViT forward pass: patch_embed + pos_embed, ViT blocks, PatchMerger.
    c.f. VideoEncoder in video_encoder.py, which handles multiple frames.
    """

    resolution = 224
    model_id = "Qwen/Qwen3.5-0.8B"

    def __init__(self) -> None:
        super().__init__()
        self.visual = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.float32
        ).model.visual
        self.image_processor = AutoProcessor.from_pretrained(self.model_id).image_processor
        dummy = torch.zeros(3, self.resolution, self.resolution, dtype=torch.float32)
        grid = self.image_processor(images=[dummy], return_tensors="pt", do_rescale=False)[
            "image_grid_thw"
        ][0]
        self.grid_h = int(grid[1].item()) // self.image_processor.merge_size
        self.grid_w = int(grid[2].item()) // self.image_processor.merge_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear")
        batch_size = x.size(0)
        cpu_images = [x[i].detach().cpu().float() for i in range(batch_size)]
        img_out = self.image_processor(images=cpu_images, return_tensors="pt", do_rescale=False)
        pixel_values = img_out["pixel_values"].to(x.device).type(self.visual.dtype)
        image_grid_thw = img_out["image_grid_thw"].to(x.device)

        hidden_states = self.visual.patch_embed(pixel_values)
        pos_embeds = self.visual.fast_pos_embed_interpolate(image_grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.visual.rot_pos_emb(image_grid_thw)
        total_tokens, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(total_tokens, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(total_tokens, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        patches_per_image = (image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=hidden_states.device)
        for i, n in enumerate(patches_per_image):
            cu_seqlens[i + 1] = cu_seqlens[i] + n

        for blk in self.visual.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        merged = self.visual.merger(hidden_states)  # (B*grid_h*grid_w, hidden_dim)
        hidden_dim = merged.size(-1)
        merged = merged.view(batch_size, self.grid_h, self.grid_w, hidden_dim)
        return merged.permute(0, 3, 1, 2)  # (B, hidden_dim, grid_h, grid_w)


IMAGE_ENCODERS = {
    "taesd": TaesdEncoder,
    "wan": WanEncoder,
    "dinov2": Dinov2Encoder,
    "siglip2": Siglip2Encoder,
    "vjepa2": Vjepa2Encoder,
    "qwen": QwenImageEncoder,
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
