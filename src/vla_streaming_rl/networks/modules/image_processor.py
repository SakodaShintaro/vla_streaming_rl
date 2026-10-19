# SPDX-License-Identifier: MIT
import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


class QwenImageEncoder(nn.Module):
    """Runs Qwen3.5's ViT on a single image per batch element (sequence length 1).

    No temporal attention (there is only one frame), so this is just the
    original ViT forward pass: patch_embed + pos_embed, ViT blocks, PatchMerger.
    画像は VLA と同じく観測の大きさのまま入れ、大きさの調整はプロセッサに任せる。
    """

    model_id = "Qwen/Qwen3.5-0.8B"

    def __init__(self) -> None:
        super().__init__()
        self.visual = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.float32
        ).model.visual
        self.image_processor = AutoProcessor.from_pretrained(self.model_id).image_processor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        img_out = self.image_processor(
            images=x, return_tensors="pt", do_rescale=False, device=x.device
        )
        pixel_values = img_out["pixel_values"].type(self.visual.dtype)
        image_grid_thw = img_out["image_grid_thw"].to(x.device)

        # タワーの forward を使わずブロックごとに回すので、学習済みの位置埋め込みをこの
        # グリッドへ補間したものと、パッチ位置の回転埋め込みをここで作る
        hidden_states = self.visual.patch_embed(pixel_values)
        indices, weights = get_vision_interpolation_indices_and_weights(
            image_grid_thw,
            num_grid_per_side=self.visual.num_grid_per_side,
            mode=self.visual.interpolation_mode,
            align_corners=self.visual.interpolation_align_corners,
            spatial_merge_size=self.visual.config.spatial_merge_size,
        )
        pos_embeds = (self.visual.pos_embed(indices) * weights[:, :, None]).sum(1)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = self.visual.rotary_pos_emb(
            get_vision_position_ids(image_grid_thw, self.visual.spatial_merge_size)
        )
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
        # 同じ大きさの画像のバッチなので、格子はどの画像も同じ
        merge_size = self.image_processor.merge_size
        grid_h = int(image_grid_thw[0, 1].item()) // merge_size
        grid_w = int(image_grid_thw[0, 2].item()) // merge_size
        merged = merged.view(batch_size, grid_h, grid_w, hidden_dim)
        return merged.permute(0, 3, 1, 2)  # (B, hidden_dim, grid_h, grid_w)


class ImageProcessor(nn.Module):
    """A frozen pretrained image encoder (Qwen3.5's ViT, :class:`QwenImageEncoder`).

    It never trains, so a replay buffer stores what it produces for a frame
    rather than the frame: the agents run ``encode`` once when a frame is
    collected, and a learning step reads ``output_shapes`` tensors back. The
    trainable layer on top, projecting to the network's token width, belongs to
    the network.
    """

    def __init__(self, image_shapes: list[tuple[int, ...]]) -> None:
        super().__init__()
        # 1ステップの観測の画像ごとの (3, H, W)。どれも同じエンコーダに通す
        assert all(shape[0] == 3 for shape in image_shapes), image_shapes
        backbone = QwenImageEncoder()
        self.backbone = backbone.train(False).requires_grad_(False)
        # 画像ごとのエンコーダの出力の (C, H', W')
        self.output_shapes = [
            tuple(self.encode(torch.zeros(1, *shape)).size()[1:]) for shape in image_shapes
        ]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, C, H', W'): one cell per patch of the encoder's grid
        return self.backbone.encode(x)
