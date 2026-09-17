# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F
from diffusers import AutoencoderTiny
from torch import nn
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

from vla_streaming_rl.networks.modules.qwen_vision import (
    interpolated_pos_embed,
    rotary_pos_embed,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def resize_and_normalize(x: torch.Tensor, resolution: int) -> torch.Tensor:
    x = F.interpolate(x, size=(resolution, resolution), mode="bilinear")
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def as_token(vector: torch.Tensor) -> torch.Tensor:
    """(B, C) -> (B, C, 1, 1): a pooled vector kept in the (B, C, H, W) contract
    every encoder here returns, so the single-token path is a 1x1 grid rather
    than a separate shape the consumers would have to branch on."""
    return vector[:, :, None, None]


def fold_grid_into_channels(latent: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C * H * W, 1, 1). The lossless way to reach one token
    for encoders whose grid carries no semantic summary (the VAEs): every cell
    is kept and ``ImageProcessor``'s 1x1 convolution becomes the learned linear
    layer that mixes them, exactly as ``AnimalBackbone`` flattens its tower
    output into a single dense layer."""
    return latent.flatten(1)[:, :, None, None]


class TaesdEncoder(nn.Module):
    def __init__(self, observation_space_shape: tuple[int]) -> None:
        super().__init__()
        assert observation_space_shape[0] == 3
        self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return self.vae.encode(x).latents  # (B, 4, H/8, W/8)

    def encode_token(self, x: torch.Tensor) -> torch.Tensor:
        return fold_grid_into_channels(self.encode(x))  # (B, 4 * H/8 * W/8, 1, 1)


class Dinov2Encoder(nn.Module):
    resolution = 224
    patch_size = 14
    grid_size = resolution // patch_size

    def __init__(self, observation_space_shape: tuple[int]) -> None:
        super().__init__()
        assert observation_space_shape[0] == 3
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_and_normalize(x, self.resolution)
        tokens = self.model(pixel_values=x).last_hidden_state[:, 1:, :]  # drop CLS token
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size)

    def encode_token(self, x: torch.Tensor) -> torch.Tensor:
        """The CLS token: DINOv2's image-level distillation loss acts on it, and
        it is what the official linear probes read as the image summary."""
        x = resize_and_normalize(x, self.resolution)
        return as_token(self.model(pixel_values=x).pooler_output)  # pooler_output is the CLS token


class Siglip2Encoder(nn.Module):
    resolution = 224
    patch_size = 16
    grid_size = resolution // patch_size

    def __init__(self, observation_space_shape: tuple[int]) -> None:
        super().__init__()
        assert observation_space_shape[0] == 3
        self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").vision_model

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_and_normalize(x, self.resolution)
        tokens = self.model(pixel_values=x).last_hidden_state  # no CLS token
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size)

    def encode_token(self, x: torch.Tensor) -> torch.Tensor:
        """SigLIP has no CLS token; it is pretrained with a MAP head -- one
        learned query attending over the patches -- and that pooled vector is the
        embedding its contrastive loss aligns with text, so it is the single
        token this checkpoint was actually trained to produce."""
        x = resize_and_normalize(x, self.resolution)
        return as_token(self.model(pixel_values=x).pooler_output)  # attention pooling head


class Vjepa2Encoder(nn.Module):
    resolution = 256
    patch_size = 16
    tubelet_size = 2
    grid_size = resolution // patch_size

    def __init__(self, observation_space_shape: tuple[int]) -> None:
        super().__init__()
        assert observation_space_shape[0] == 3
        self.model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_and_normalize(x, self.resolution)
        clip = x.unsqueeze(1).repeat(1, self.tubelet_size, 1, 1, 1)  # (B, T=tubelet_size, 3, H, W)
        tokens = self.model(pixel_values_videos=clip).last_hidden_state  # (B, grid*grid, C)
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size)

    def encode_token(self, x: torch.Tensor) -> torch.Tensor:
        """V-JEPA 2 has neither a CLS token nor a pretrained pooling head -- the
        attentive probe ships only with the classification checkpoints -- so the
        mean over the patch tokens is the summary its frozen-encoder evaluations
        fall back on."""
        return self.encode(x).mean(dim=(2, 3), keepdim=True)


class QwenImageEncoder(nn.Module):
    """Runs Qwen3.5's ViT on a single image per batch element (sequence length 1).

    No temporal attention (there is only one frame), so this is just the
    original ViT forward pass: patch_embed + pos_embed, ViT blocks, PatchMerger.
    """

    resolution = 224
    model_id = "Qwen/Qwen3.5-0.8B"

    def __init__(self, observation_space_shape: tuple[int]) -> None:
        super().__init__()
        assert observation_space_shape[0] == 3
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
        img_out = self.image_processor(
            images=x, return_tensors="pt", do_rescale=False, device=x.device
        )
        pixel_values = img_out["pixel_values"].type(self.visual.dtype)
        image_grid_thw = img_out["image_grid_thw"].to(x.device)

        hidden_states = self.visual.patch_embed(pixel_values)
        pos_embeds = interpolated_pos_embed(self.visual, image_grid_thw)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = rotary_pos_embed(self.visual, image_grid_thw)
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

    def encode_token(self, x: torch.Tensor) -> torch.Tensor:
        """The ViT hands the LLM a grid of merged patches and has no summary
        token of its own, so the tokens are averaged after the PatchMerger --
        i.e. after the 2x2 merge the checkpoint was trained to perform."""
        return self.encode(x).mean(dim=(2, 3), keepdim=True)


IMAGE_ENCODERS = {
    "taesd": TaesdEncoder,
    "dinov2": Dinov2Encoder,
    "siglip2": Siglip2Encoder,
    "vjepa2": Vjepa2Encoder,
    "qwen": QwenImageEncoder,
}


ENCODE_MODES = {
    # the patch grid, one token per cell
    "grid": lambda backbone, x: backbone.encode(x),
    # the whole image as one token, pooled the way each backbone was trained
    "single_token": lambda backbone, x: backbone.encode_token(x),
}


class ImageProcessor(nn.Module):
    """A frozen pretrained image encoder.

    It never trains, so a replay buffer stores what it produces for a frame
    rather than the frame: the agents run ``encode`` once when a frame is
    collected, and a learning step reads ``output_shape`` tensors back. The
    trainable layer on top, projecting to the network's token width, belongs to
    the network.
    """

    def __init__(
        self,
        observation_space_shape: tuple[int],
        image_encoder_type: str,
        image_encode_mode: str,
    ) -> None:
        super().__init__()
        assert image_encode_mode in ENCODE_MODES
        self.observation_space_shape = observation_space_shape
        self.image_encode_mode = image_encode_mode
        backbone = IMAGE_ENCODERS[image_encoder_type](observation_space_shape)
        self.backbone = backbone.train(False).requires_grad_(False)
        self.output_shape = list(self.encode(torch.zeros(1, *observation_space_shape)).size())[1:]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, C, H', W'); H' = W' = 1 in the single-token mode
        return ENCODE_MODES[self.image_encode_mode](self.backbone, x)


if __name__ == "__main__":
    import time

    device = "cuda"
    measure_speed = True
    speed_iters = 10
    x = torch.zeros(1, 3, 96, 96, device=device)

    for name, encoder_class in IMAGE_ENCODERS.items():
        encoder = encoder_class(tuple(x.shape[1:])).eval().requires_grad_(False).to(device)
        param_num = sum(p.numel() for p in encoder.parameters())
        with torch.inference_mode():
            output = encoder.encode(x)
            token = encoder.encode_token(x)
        print(
            f"{name}: params={param_num:,} output_shape={tuple(output.shape)} "
            f"token_shape={tuple(token.shape)}"
        )

        if measure_speed:
            with torch.inference_mode():
                for _ in range(3):
                    encoder.encode(x)
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                for _ in range(speed_iters):
                    encoder.encode(x)
                torch.cuda.synchronize()
                elapsed_time = time.perf_counter() - start_time
            print(f"{name}: {elapsed_time / speed_iters * 1000:.2f} ms/iter")
