# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.networks.interface import InferResult
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_scaler import FixedRewardScaler, RewardScaler
from vla_streaming_rl.scalar_obs import read_scalar_obs


@dataclass
class StepResult:
    """Structured return value of an agent's ``select_action`` / ``step``.

    Replaces the old free-form ``info`` dict, which mixed three unrelated
    concerns and forced the trainer to know each agent's internal keys.
    The concerns are now split:

    - ``action``: the env action to execute this tick.
    - ``metrics``: flat scalar telemetry, logged verbatim to wandb.
    - ``panels``: named RGB image panels (HxWx3) for the render strip. The
      trainer concatenates whatever panels are present, in insertion order,
      so an agent can contribute extra panels (e.g. the next-frame prediction
      or a bird's-eye trajectory view) without the trainer knowing about them.
      Stable-panel contract: an agent must emit the SAME set of panel keys with
      the SAME shapes on every step of a run (use a blank placeholder when the
      content does not exist yet, e.g. before the first prediction). The render
      frames are encoded into a single video, which requires a constant size.
    - ``texts``: named free-form text, written to the episode's ``texts.tsv``.
      What a panel shows as pixels this keeps as characters, so a chain of
      thought can be read back and searched after the run. Every agent puts the
      instruction its policy read this tick under ``"prompt"``, which is also
      what the trainer captions the render with.
    """

    action: np.ndarray
    metrics: dict[str, float]
    panels: dict[str, np.ndarray]
    texts: dict[str, str]


def split_actor_critic_params(
    network: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """ネットワークのパラメータを actor 側と critic 側（value head）に分けたもの。"""
    critic_params = list(network.value_head.parameters())
    critic_param_ids = {id(p) for p in critic_params}
    actor_params = [p for p in network.parameters() if id(p) not in critic_param_ids]
    return actor_params, critic_params


class Agent(ABC):
    """The contract the trainer drives: ``step`` on every tick while learning,
    ``select_action`` when acting without learning.

    The agent grid is orthogonal: the learning rule (off-policy / streaming) is
    the class, which ``agent_type`` names, and the network it optimizes is a
    constructor argument. ``instruction`` is the text the policy reads beside
    the image.

    どの学習モードも、毎ティックの観測をバッファに1行積み、ネットワークが出した行動
    チャンクを1ステップずつ実行する。その共通の流れ（:meth:`_observe`、
    :meth:`_continue_chunk`、:meth:`_start_chunk`）と状態はここが持ち、サブクラスは
    いつ・何から学習するかと、推論にどの窓と再帰状態を渡すかだけを書く。"""

    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        image_names: tuple[str, ...],
        network: nn.Module,
        instruction: str,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        buffer_size: int,
        buffer_device: str,
        priority_alpha: float,
        priority_beta: float,
        priority_epsilon: float,
        seq_len: int,
        horizon: int,
        normalizing_by_return: bool,
        reward_scaler: RewardScaler | FixedRewardScaler,
        use_done: bool,
        reset_on_episode_end: bool,
    ) -> None:
        self.device = torch.device("cuda")
        # 観測から読んでネットワークに渡す画像の名前（IMAGE_OBS_NAMES の並び）
        self.image_names = image_names
        self.network = network
        self.instruction = instruction
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.seq_len = seq_len
        self.horizon = horizon
        self.normalizing_by_return = normalizing_by_return
        self.use_done = use_done
        self.reset_on_episode_end = reset_on_episode_end
        self.reward_scaler = reward_scaler

        # 方策の行動は [-1, 1] に正規化した空間にあり、環境の行動空間へはこれで写す
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0

        self.rnn_state = network.init_state().to(self.device)
        self.rb = ReplayBuffer(
            size=buffer_size,
            seq_len=seq_len + horizon,
            horizon=horizon,
            obs_shape=network.stored_image_shape(),
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(buffer_device),
            priority_alpha=priority_alpha,
            priority_beta=priority_beta,
            priority_epsilon=priority_epsilon,
        )

        # 実行中の行動チャンク (horizon, action_dim) と、そのうち実行した数
        self.action_chunk: np.ndarray | None = None
        self.chunk_step = 0
        # 最後に環境へ渡した行動。エピソードをまたいで持ち越す
        self.prev_action = np.zeros(action_space.shape, dtype=np.float32)
        # the first observation of a run starts an episode
        self._previous_done = True
        # Shared representation fed to policy/value/prediction heads on the most
        # recent inference (used by scripts/collect_probe_data.py).
        self.last_features: torch.Tensor | None = None

    # --- agent surface -----------------------------------------------------

    @abstractmethod
    def select_action(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult: ...

    @abstractmethod
    def step(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        """Act on this tick and learn from it, the way this learning mode learns."""

    def on_episode_end(self, score: float) -> None:
        """エピソードが ``score`` で終わったときに呼ぶ。学習でも評価でも呼ぶ。"""
        if self.normalizing_by_return:
            self.reward_scaler.update(score)

    def optimizer_state_dict(self) -> dict:
        """Optimizer states to checkpoint. The trainer stores and restores this
        verbatim."""
        return {
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state: dict) -> None:
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic"])

    # --- per-tick machinery ------------------------------------------------

    def _observe(
        self, obs: dict[str, Any], reward: float, terminated: bool, truncated: bool, info: dict
    ) -> dict[str, float]:
        """このティックの観測を受け取り、エピソードの境目を処理して、バッファに1行積む。
        metrics を返す。"""
        episode_done = terminated or truncated
        # A terminal observation still belongs to the episode that ended, so the
        # recurrent state is reset on the tick after it rather than on it.
        reset_state = self._previous_done and self.reset_on_episode_end
        if self._previous_done:
            if reset_state:
                self.rnn_state = self.network.init_state().to(self.device)
            # A chunk begun on the terminal frame would otherwise run into the
            # new episode; the first action of an episode comes from its own frame.
            self.action_chunk = None
            self.chunk_step = 0
        self._previous_done = episode_done

        # What the agent trains on, against what the env reported as its score.
        shaped_reward = info["shaped_reward"]
        if not self.normalizing_by_return:
            self.reward_scaler.update(shaped_reward)
        metrics = {
            "shaped_reward": shaped_reward,
            "action_norm": float(np.linalg.norm(self.prev_action)),
            # 前のステップに実行した行動の成分ごとの値。偏りを見る
            **{f"action_{i}": float(value) for i, value in enumerate(self.prev_action)},
            "processed_reward": self.reward_scaler.normalize(torch.tensor(shaped_reward)).item(),
        }

        images = [torch.from_numpy(obs[name]).to(self.device) for name in self.image_names]
        scalar_obs = read_scalar_obs(obs)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        self.rb.add(
            self.network.to_stored_image(images, scalar_obs, self.instruction, reset_state),
            shaped_reward,
            episode_done if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            scalar_obs,
        )
        return metrics

    def _has_chunk_left(self) -> bool:
        """実行中のチャンクに、まだ実行していない行動があるか。"""
        return self.action_chunk is not None and self.chunk_step < self.horizon

    def _start_chunk(self, infer_result: InferResult, metrics: dict) -> StepResult:
        """ネットワークが出したチャンクを始め、その最初の行動を実行する。"""
        self.rnn_state = infer_result.rnn_state
        self.last_features = infer_result.features
        metrics.update(infer_result.value_report)
        self.action_chunk = infer_result.action[0].cpu().numpy()
        self.chunk_step = 0
        return self._continue_chunk(metrics)

    def _continue_chunk(self, metrics: dict) -> StepResult:
        """実行中のチャンクの次の行動を、環境の行動空間に写して実行する。"""
        action = np.clip(
            self.action_chunk[self.chunk_step] * self.action_scale + self.action_bias,
            self.action_low,
            self.action_high,
        )
        self.chunk_step += 1
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step
        return StepResult(
            action=action,
            metrics=metrics,
            panels={},
            texts={"prompt": self.instruction},
        )
