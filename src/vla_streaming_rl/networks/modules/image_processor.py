# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F
from diffusers import AutoencoderTiny
from torch import nn
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

from vla_streaming_rl.wan import WanVAEWrapper

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


class WanEncoder(nn.Module):
    def __init__(self, observation_space_shape: tuple[int]) -> None:
        super().__init__()
        assert observation_space_shape[0] == 3
        self.vae = WanVAEWrapper()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        z = self.vae.encode_to_latent(x.unsqueeze(2))
        self.vae.model.clear_cache()
        return z.squeeze(1)  # (B, 16, H/8, W/8)

    def encode_token(self, x: torch.Tensor) -> torch.Tensor:
        return fold_grid_into_channels(self.encode(x))  # (B, 16 * H/8 * W/8, 1, 1)


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
    c.f. VideoEncoder in video_encoder.py, which handles multiple frames.
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

    def encode_token(self, x: torch.Tensor) -> torch.Tensor:
        """The ViT hands the LLM a grid of merged patches and has no summary
        token of its own, so the tokens are averaged after the PatchMerger --
        i.e. after the 2x2 merge the checkpoint was trained to perform."""
        return self.encode(x).mean(dim=(2, 3), keepdim=True)


class ChannelAttention(nn.Module):
    """The channel means through two bias-free 1x1 convolutions, squashed to a
    per-channel gate."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(depth, depth // 4, 1, bias=False)
        self.expand = nn.Conv2d(depth // 4, depth, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(dim=(2, 3), keepdim=True)
        out = self.expand(F.elu(self.reduce(out)))
        return torch.sigmoid(out)


class FixupAttentionBlock(nn.Module):
    """A residual block with no normalization, four scalar biases and a scalar
    multiplier, and the channel gate applied between the two convolutions."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.res1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.res2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.attention = ChannelAttention(depth)
        self.bias0 = nn.Parameter(torch.zeros(()))
        self.bias1 = nn.Parameter(torch.zeros(()))
        self.bias2 = nn.Parameter(torch.zeros(()))
        self.bias3 = nn.Parameter(torch.zeros(()))
        self.multiplier = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.elu(x) + self.bias0
        out = self.res1(out) + self.bias1
        out = out * self.attention(out)
        out = F.elu(out) + self.bias2
        out = self.res2(out) * self.multiplier + self.bias3
        return out + x


class FixupEncoder(nn.Module):
    """The Animal-AI Olympics winning network's visual trunk -- a Fixup residual
    tower with channel attention, one stride-2 max pool per stage -- as an
    encoder like the others, so ``networks/animal_ppo.py`` reaches it and the
    pretrained backbones through the same :class:`ImageProcessor`.

    Nothing here is pretrained, so this is the encoder that wants
    ``image_encoder_trainable`` set: frozen, it stays a random projection.
    """

    depths = (16, 32, 64, 128)

    def __init__(self, observation_space_shape: tuple[int]) -> None:
        super().__init__()
        in_channels = observation_space_shape[0]
        self.stages = nn.ModuleList()
        for depth in self.depths:
            self.stages.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(in_channels, depth, 3, padding=1),
                        "block1": FixupAttentionBlock(depth),
                        "block2": FixupAttentionBlock(depth),
                    }
                )
            )
            in_channels = depth

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for stage in self.stages:
            out = stage["conv"](out)
            out = F.max_pool2d(out, 3, 2, padding=1)
            out = stage["block1"](out)
            out = stage["block2"](out)
        return F.elu(out)  # (B, depths[-1], H/16, W/16), rounding up

    def encode_token(self, x: torch.Tensor) -> torch.Tensor:
        """Trained from scratch, the trunk has no pooling anyone pretrained for
        it, so the grid is folded whole into the channel axis -- exactly the
        flattening the dense layer above it already does."""
        return fold_grid_into_channels(self.encode(x))


IMAGE_ENCODERS = {
    "fixup": FixupEncoder,
    "taesd": TaesdEncoder,
    "wan": WanEncoder,
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
    def __init__(
        self,
        observation_space_shape: tuple[int],
        image_encoder_type: str,
        image_encoder_output_dim: int,
        image_encode_mode: str,
        image_encoder_trainable: bool,
    ) -> None:
        super().__init__()
        assert image_encode_mode in ENCODE_MODES
        self.observation_space_shape = observation_space_shape
        self.image_encode_mode = image_encode_mode
        backbone = IMAGE_ENCODERS[image_encoder_type](observation_space_shape)
        # the pretrained backbones are normally frozen feature extractors and the
        # fixup trunk is normally trained from scratch, but both are the config's call
        self.backbone = backbone.train(image_encoder_trainable).requires_grad_(
            image_encoder_trainable
        )
        x = torch.zeros(1, *observation_space_shape)
        with torch.no_grad():
            backbone_dim = ENCODE_MODES[image_encode_mode](self.backbone, x).size(1)
        if backbone_dim > image_encoder_output_dim:
            self.projection = nn.Conv2d(backbone_dim, image_encoder_output_dim, kernel_size=1)
        else:
            self.projection = nn.Identity()
        with torch.no_grad():
            self.output_shape = list(self.encode(x).size())[1:]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W); H = W = 1 in the single-token mode
        return self.projection(ENCODE_MODES[self.image_encode_mode](self.backbone, x))


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
