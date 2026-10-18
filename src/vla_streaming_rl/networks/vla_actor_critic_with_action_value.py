# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from ..replay_buffer import ReplayBufferData
from ..scalar_obs import SCALAR_OBS_DIM, SCALAR_OBS_NAMES
from .interface import (
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from .modules.running_normalizer import RunningNormalizer
from .modules.value_head import DistributionalValueHead
from .modules.vlm_backbone import load_model

# VLM のチャットテンプレートの書式
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

# ヘッドに数値のまま渡すスカラーの観測の、SCALAR_OBS_NAMES での位置。global_step は
# エピソードの中の状況を表さず、学習の間ずっと増え続けるので渡さない
HEAD_SCALAR_INDICES = [
    index for index, name in enumerate(SCALAR_OBS_NAMES) if name != "global_step"
]


@dataclass
class LossTerms:
    """リプレイバッチ1つ分の損失と、それを計算する途中で得た状態。"""

    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    info: dict
    state: torch.Tensor
    action_chunk: torch.Tensor
    td_error: torch.Tensor


class VLAActorCriticWithActionValue(NetworkInterface):
    """1枚の画像と指示の文から行動を出す VLA。

    毎ステップ、一人称の視点の画像1枚と指示の文を、1つの user ターンとして VLM に読ませる。過去の観測は読まない。全位置の活性を
    全層（埋め込みと各層の出力）で平均し、画像のトークンの格子を ``image_pool_size`` 四方に平均
    プーリングしたものと、末尾の ``num_last_tokens`` 個の位置を並べたものを表現とする。末尾は
    チャットテンプレートの固定の部分で、画像と指示を読んだ後の活性を持つ。それを位置ごとに
    ``state_out_dim`` へ射影し、そのステップのスカラーの観測（global_step を除く）を正規化して
    数値のまま連結したものを状態にする。方策ヘッドと行動価値ヘッドはその状態を読む。

    VLM は凍結して使うので、表現は毎ステップ一度だけ計算してバッファに保存する（行動は
    ``horizon`` ステップのチャンクごとに決めるが、表現はチャンクの途中のステップでも保存する）。
    学習はそれを読み戻し、VLM を通さない。
    """

    def __init__(
        self,
        *,
        action_space_shape: tuple[int],
        value_head_factory: Callable[[int, int], DistributionalValueHead],
        policy_head_factory,
        horizon: int,
        critic_loss_weight: float,
        vla_config: DictConfig,
    ) -> None:
        super().__init__()
        # 表現はバッファに保存するので、バッファから読む窓は今のステップだけでよい
        self.seq_len = 1
        self.horizon = horizon
        self.critic_loss_weight = critic_loss_weight
        device = "cuda"
        self.device = device
        self.vlm, self.processor = load_model(
            model_id=vla_config.model_id,
            load_in_4bit=vla_config.load_in_4bit,
            device=torch.device(device),
        )
        self.vlm.eval().requires_grad_(False)
        self.vlm_hidden_size = self.vlm.config.text_config.hidden_size
        self.image_pool_size = vla_config.image_pool_size
        self.num_last_tokens = vla_config.num_last_tokens
        self.image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.merge_size = self.processor.image_processor.merge_size
        self.num_positions = self.image_pool_size**2 + self.num_last_tokens
        self.state_out_proj = nn.Linear(self.vlm_hidden_size, vla_config.state_out_dim).to(device)
        self.scalar_obs_normalizer = RunningNormalizer(SCALAR_OBS_DIM)
        state_dim = self.num_positions * vla_config.state_out_dim + len(HEAD_SCALAR_INDICES)
        action_dim = action_space_shape[0]
        self.policy_head = policy_head_factory(state_dim=state_dim, action_dim=action_dim).to(
            device
        )
        self.value_head = value_head_factory(in_channels=state_dim, action_dim=action_dim).to(
            device
        )
        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def stored_image_shape(self) -> tuple[int, ...]:
        """何も保存しない。画像は推論のときに ``InferInput.frames`` で受け取る。"""
        return (0,)

    def to_stored_image(self, images: list[torch.Tensor]) -> torch.Tensor:
        del images
        return torch.zeros(0)

    def stored_state_shape(self) -> tuple[int, ...]:
        """VLM の表現の、画像をプーリングした格子と末尾の位置。射影は学習するので、その手前を
        保存する。"""
        return (self.num_positions, self.vlm_hidden_size)

    def observe_scalar_obs(self, scalar_obs: np.ndarray) -> None:
        self.scalar_obs_normalizer.update(scalar_obs)

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        return self._act(data.frames, data.instruction, data.scalar_obs_seq, data.warmup)

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        terms = self._loss_terms(data, data.states[:, -1])
        return LossResult(loss=terms.total_loss, info=terms.info, td_error=terms.td_error)

    def infer_and_compute_loss(
        self,
        data: ReplayBufferData,
        frames: list[torch.Tensor],
        instruction: str,
    ) -> InferLossResult:
        # 最後のステップはまだ行動を決めていないので、ここで表現を作る
        with torch.inference_mode():
            infer_result = self._act(frames, instruction, data.scalar_obs, False)
        terms = self._loss_terms(data, infer_result.stored_state.unsqueeze(0).clone())
        # -Q(s,a) for eligibility trace backward (detached from encoder)
        neg_value_detached = -self.value_head.scalar_value(
            terms.state.detach(), terms.action_chunk.detach()
        ).mean()
        return InferLossResult(
            infer_result=infer_result,
            loss_result=LossResult(loss=terms.total_loss, info=terms.info, td_error=terms.td_error),
            et_info=EligibilityTraceInfo(
                actor_entropy_loss=terms.actor_loss,
                neg_value=neg_value_detached,
                delta=terms.info["losses/delta"],
            ),
        )

    ####################
    # Internal methods #
    ####################

    def _act(
        self,
        frames: list[torch.Tensor],
        instruction: str,
        scalar_obs: torch.Tensor,
        warmup: bool,
    ) -> InferResult:
        """一人称の視点（IMAGE_OBS_NAMES の先頭）の1枚と指示から表現を作り、最後のステップの
        スカラーの観測（(1, T, SCALAR_OBS_DIM) の最後）と合わせて行動を決める。"""
        assert scalar_obs.shape[0] == 1, "Batch size must be 1 for inference"
        representation = self._represent(frames[0], instruction)
        state = self._state(representation.unsqueeze(0), scalar_obs[:, -1])
        action = self.policy_head.get_action(state)
        if warmup:
            # バッファを貯めている間は一様乱数の行動を返す
            action = torch.rand_like(action) * 2.0 - 1.0
        return InferResult(
            action=action,
            value_report=self.value_head.value_report(self.value_head(state, action).output),
            rnn_state=self._dummy_state.clone(),
            features=state,
            stored_state=representation,
        )

    def _represent(self, image: torch.Tensor, text: str) -> torch.Tensor:
        """[0, 1] の画像 (3, H, W) と文を VLM に読ませた、全位置の活性の全層平均から、画像の
        トークンの格子を ``image_pool_size`` 四方にプーリングしたものと末尾の
        ``num_last_tokens`` 個の位置を並べた表現 (num_positions, hidden_size)。"""
        prompt = f"{IM_START}user\n{IMAGE_PLACEHOLDER}{text}{IM_END}\n{IM_START}assistant\n"
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", do_rescale=False
        ).to(self.device)
        outputs = self.vlm(**inputs, output_hidden_states=True, logits_to_keep=1)
        # 全層を一度に積まず、足し込みながら平均する
        layer_mean = sum(depth[0].to(torch.float32) for depth in outputs.hidden_states) / len(
            outputs.hidden_states
        )
        # 画像のトークンは、統合後の格子のラスター順に並ぶ
        image_tokens = layer_mean[inputs["input_ids"][0] == self.image_pad_id]
        height, width = (inputs["image_grid_thw"][0, 1:] // self.merge_size).tolist()
        grid = image_tokens.T.reshape(1, self.vlm_hidden_size, height, width)
        pooled = F.adaptive_avg_pool2d(grid, self.image_pool_size).flatten(start_dim=2)[0].T
        return torch.cat([pooled, layer_mean[-self.num_last_tokens :]]).to(torch.bfloat16)

    def _loss_terms(self, data: ReplayBufferData, next_representation: torch.Tensor) -> LossTerms:
        """リプレイバッチの損失。現在の状態はチャンクの手前のステップに保存した表現から、
        次の状態は ``next_representation``（最後のステップのもの）から作る。"""
        with torch.no_grad():
            next_state = self._state(next_representation, data.scalar_obs[:, -1])
            next_action = self.policy_head.get_action(next_state)
            next_output = self.value_head(next_state, next_action).output
        target_value = self.value_head.compute_target_value(
            next_output, data.rewards[:, -self.horizon :], data.dones[:, -self.horizon :]
        )
        state = self._state(
            data.states[:, -self.horizon - 1], data.scalar_obs[:, -self.horizon - 1]
        )
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)
        critic_loss, critic_info, td_error = self.value_head.compute_critic_loss(
            state, action_chunk, target_value, data.weights
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state, action_chunk, value_head=self.value_head
        )
        info = {f"losses/{key}": value for key, value in {**critic_info, **actor_info}.items()}
        return LossTerms(
            total_loss=self.critic_loss_weight * critic_loss + actor_loss,
            actor_loss=actor_loss,
            info=info,
            state=state,
            action_chunk=action_chunk,
            td_error=td_error,
        )

    def _state(self, representation: torch.Tensor, scalar_obs: torch.Tensor) -> torch.Tensor:
        """保存した表現 (B, num_positions, hidden_size) を射影したものに、同じステップの
        スカラーの観測 (B, SCALAR_OBS_DIM) を正規化して連結し、ヘッドが読む状態にする。"""
        projected = self.state_out_proj(representation.to(torch.float32)).flatten(start_dim=1)
        normalized = self.scalar_obs_normalizer.normalize(
            scalar_obs.to(projected.device, torch.float32)
        )
        return torch.cat([projected, normalized[:, HEAD_SCALAR_INDICES]], dim=1)
