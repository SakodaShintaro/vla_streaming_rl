# SPDX-License-Identifier: MIT
"""Off-policy learning mode: act now, learn later from a large replay buffer.

Every tick is stored; once ``learning_starts`` ticks have gone by, one gradient
step on a ``batch_size`` sample of the buffer fires every ``horizon`` ticks,
before the action of that tick is chosen. Below ``learning_starts`` the env is
driven by uniform random actions, so the buffer fills with something other than
an untrained policy's output while the network's recurrent state still follows
the episode.

The learning mode is the class and the network is a constructor argument, so
this file is one half of the (learning mode) x (network) grid; the streaming
half is ``streaming.py``. The two share no base beyond :class:`Agent`, which
costs some repetition in the per-tick path and buys each mode being readable
end to end in one file.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.agents.base import Agent, StepResult, render_high_level
from vla_streaming_rl.agents.prompt import PromptBuilder
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.networks.modules.high_level_policy import HighLevelPolicy
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor

# サブタスクに向けた行動の点数の移動平均を更新する割合の下限。判定の回数がこの逆数に届くまでは
# 単純平均（回数で割る）で取るので、決め打ちの初期値に引きずられない
SCORE_MEAN_RATE = 0.01


class OffPolicyAgent(Agent):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        network: nn.Module,
        normalizing_by_return: bool,
        learning_starts: int,
        batch_size: int,
        max_grad_norm: float,
        use_done: bool,
        seq_len: int,
        horizon: int,
        actor_lr: float,
        critic_lr: float,
        weight_decay: float,
        buffer_size: int,
        buffer_device: str,
        max_prompt_tokens: int,
        pad_token_id: int,
        reset_on_episode_end: bool,
        prompt_builder: PromptBuilder,
        score_reward_weight: float,
        high_level_policy: HighLevelPolicy | None,
    ) -> None:
        super().__init__(
            horizon=horizon,
            reset_on_episode_end=reset_on_episode_end,
            prompt_builder=prompt_builder,
        )
        # 凍結した高レベル方策。持たない学習では None
        self.high_level_policy = high_level_policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # サブタスクに向けた行動の点数を内発的な報酬にする重みと、点数の移動平均。報酬は
        # 平均との差にして、採点の甘さ・厳しさの偏りを打ち消す
        self.score_reward_weight = score_reward_weight
        self.score_mean = 0.0
        self.score_count = 0

        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0
        self.reward_processor = RewardProcessor("scaling", 1.0)
        self.normalizing_by_return = normalizing_by_return

        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.use_done = use_done

        # Sequence observation management
        self.seq_len = seq_len

        # Action chunking state
        self.action_chunk = None  # (horizon, action_dim) - current action chunk
        self.chunk_step = 0  # current step within chunk

        self.network = network
        self.rnn_state = self.network.init_state().to(self.device)

        # Actor / critic optimizer split (critic == value head); both AdamW,
        # the replayed update has no trace to carry.
        critic_params = list(self.network.value_head.parameters())
        critic_param_ids = {id(p) for p in critic_params}
        actor_params = [p for p in self.network.parameters() if id(p) not in critic_param_ids]
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay)
        self.critic_optimizer = optim.AdamW(critic_params, lr=critic_lr, weight_decay=weight_decay)

        self.rb = ReplayBuffer(
            size=buffer_size,
            seq_len=self.seq_len + self.horizon,
            horizon=self.horizon,
            obs_shape=self.network.stored_image_shape(),
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            subtask_shape=self.network.subtask_shape,
            output_device=self.device,
            storage_device=torch.device(buffer_device),
            max_prompt_tokens=max_prompt_tokens,
            pad_token_id=pad_token_id,
        )

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        # the first observation of a run starts an episode
        self._previous_done = True
        # Shared representation fed to policy/value/prediction heads on the
        # most recent select_action inference (used by scripts/probe.py).
        self.last_features: torch.Tensor | None = None
        self.is_learning = False

    # --- agent surface -----------------------------------------------------

    def step(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        train_metrics = {}
        if (
            global_step >= self.learning_starts
            and global_step % self.horizon == 0
            and self.rb.num_stored() >= self.batch_size + self.rb.seq_len
        ):
            if not self.is_learning:
                print(f"Start learning at global step {global_step}.")
                self.is_learning = True
            data = self.rb.sample(self.batch_size)
            data.rewards = self.reward_processor.normalize(data.rewards)
            result = self.network.compute_loss(data)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            train_metrics = result.info
        step_result = self.select_action(global_step, obs, reward, terminated, truncated, info)
        step_result.metrics.update(train_metrics)
        return step_result

    def on_episode_end(self, score: float) -> dict:
        del score
        return {}

    def optimizer_state_dict(self) -> dict:
        return {
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state: dict) -> None:
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic"])

    # --- per-tick machinery ------------------------------------------------

    def _reset_rnn_state_if_fresh(self, episode_done: bool) -> None:
        if self._previous_done and self.reset_on_episode_end:
            self.rnn_state = self.network.init_state().to(self.device)
        self._previous_done = episode_done

    @torch.no_grad()
    def select_action(
        self,
        global_step: int,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepResult:
        metrics = {}
        episode_done = terminated or truncated
        # A terminal observation still belongs to the episode that ended, so the
        # conversation and the chain are dropped on the tick after it rather than
        # on it -- the same boundary the rnn state resets on. Dropped on the
        # terminal tick instead, the turn that tick then writes would be the one
        # the next episode opens on.
        episode_started = self._previous_done
        # What the agent trains on, against what the env reported as its score.
        shaped_reward = info["shaped_reward"]
        metrics["shaped_reward"] = shaped_reward
        self._reset_rnn_state_if_fresh(episode_done)
        if episode_started:
            self.prompt_builder.reset()
            # A chunk begun on the terminal frame would otherwise run into the
            # new episode; the first action of an episode comes from its own frame.
            self.action_chunk = None
            self.chunk_step = 0
        metrics["action_norm"] = np.linalg.norm(self.prev_action)
        if not self.normalizing_by_return:
            self.reward_processor.update(shaped_reward)
        metrics["processed_reward"] = self.reward_processor.normalize(
            torch.tensor(shaped_reward)
        ).item()
        (
            image,
            velocity_x,
            velocity_y,
            velocity_z,
            episode_return,
            pass_mark,
            remaining_return,
            global_step_obs,
            episode_step_obs,
            health_obs,
        ) = self._preprocess(obs)
        # The language this tick, composed from the env's state and never read off
        # the observation. The chain reads the same conversation on the steps it
        # writes, and writes its own turn back into it.
        self.prompt_builder.observe(obs, reward, info, image)
        prompt = self.prompt_builder.task_text()
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        self.rb.add(
            self.network.to_stored_image(image),
            shaped_reward,
            episode_done if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            self.network.tokenize(prompt),
            self.network.tokenize(self.prompt_builder.turn_text()),
            velocity_x,
            velocity_y,
            velocity_z,
            episode_return,
            pass_mark,
            remaining_return,
            global_step_obs,
            episode_step_obs,
            health_obs,
        )
        # 高レベル方策を進め、その返答でこのステップの行を埋める。エピソードの最初の
        # ステップでは、前のエピソードの返答を捨ててそのフレームで書き直す
        panels = {}
        texts = {"prompt": prompt}
        if self.high_level_policy is None:
            self.rb.amend_latest(torch.zeros(self.network.subtask_shape), 0, [])
        else:
            if episode_started:
                self.high_level_policy.reset()
            reply = self.high_level_policy.advance()
            self.rb.amend_latest(
                self.network.to_stored_subtask(reply.activations),
                reply.age,
                self.network.tokenize(reply.text),
            )
            panels, high_level_texts = render_high_level(reply)
            texts.update(high_level_texts)
            if reply.age == 0 and reply.score is not None:
                self._reward_score(reply.score, metrics)
            elif reply.age > 0 and episode_done:
                # エピソードが終わると実行中のサブタスクには次の返答が来ないので、終端の
                # フレームで判定する
                self._reward_score(self.high_level_policy.score_current(), metrics)

        warmup = global_step < self.learning_starts

        if not warmup and self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self._to_env_action(self.action_chunk[self.chunk_step])
            self.prev_action = action
            self.chunk_step += 1
            metrics["chunk_step"] = self.chunk_step
            return StepResult(
                action=action,
                metrics=metrics,
                panels=panels,
                texts=texts,
            )

        latest_data = self.rb.get_latest(self.seq_len)
        infer_result = self.network.infer(
            InferInput(
                s_seq=latest_data.observations,
                a_seq=latest_data.actions,
                r_seq=latest_data.rewards,
                rnn_state=self.rnn_state,
                system_token_ids_seq=latest_data.system_token_ids,
                turn_token_ids_seq=latest_data.turn_token_ids,
                reply_token_ids_seq=latest_data.reply_token_ids,
                velocity_x_seq=latest_data.velocity_x,
                velocity_y_seq=latest_data.velocity_y,
                velocity_z_seq=latest_data.velocity_z,
                episode_return_seq=latest_data.episode_return,
                pass_mark_seq=latest_data.pass_mark,
                remaining_return_seq=latest_data.remaining_return,
                global_step_seq=latest_data.global_step,
                episode_step_seq=latest_data.episode_step,
                health_seq=latest_data.health,
                subtask_activations_seq=latest_data.subtask_activations,
                subtask_age_seq=latest_data.subtask_age,
            )
        )
        self.rnn_state = infer_result.rnn_state
        self.last_features = infer_result.features
        metrics.update(infer_result.value_report)
        action_chunk = infer_result.action[0].cpu().numpy()
        if warmup:
            # The network was queried anyway so its recurrent state keeps
            # following the episode; only the action it chose is dropped.
            action_chunk = np.repeat(
                self._to_net_action(self.action_space.sample())[None], self.horizon, axis=0
            )
        self.action_chunk = action_chunk
        self.chunk_step = 1
        action = self._to_env_action(action_chunk[0])
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step
        return StepResult(
            action=action,
            metrics=metrics,
            panels=panels,
            texts=texts,
        )

    def _reward_score(self, score: float, metrics: dict) -> None:
        """サブタスクの区間はこの行に入る遷移で終わったので、点数を移動平均との差に
        して、その遷移の報酬に足す。"""
        # 平均はこの採点を含めて更新してから差を取るので、最初の採点の報酬は 0 になる
        self.score_count += 1
        rate = max(1.0 / self.score_count, SCORE_MEAN_RATE)
        self.score_mean += rate * (score - self.score_mean)
        bonus = self.score_reward_weight * (score - self.score_mean)
        self.rb.add_latest_reward(bonus)
        metrics["text/score"] = score
        metrics["text/score_bonus"] = bonus

    def _preprocess(self, obs: dict[str, Any]) -> tuple:
        """Turn the raw observation into what the replay buffer stores this tick:
        the image tensor, the raw scalar observations (velocity_x, velocity_y,
        velocity_z, episode_return, pass_mark, remaining_return, global_step,
        episode_step,
        health; the network updates its running normalizer stats here)."""
        image = torch.from_numpy(obs["image"]).to(self.device)
        velocity_x, velocity_y, velocity_z = obs["velocity"].astype(np.float32)
        episode_return = np.float32(obs["episode_return"][0])
        pass_mark = np.float32(obs["pass_mark"][0])
        remaining_return = np.float32(obs["remaining_return"][0])
        global_step_obs = np.float32(obs["global_step"][0])
        episode_step_obs = np.float32(obs["episode_step"][0])
        health_obs = np.float32(obs["health"][0])
        self.network.observe_scalar_obs(
            velocity_x,
            velocity_y,
            velocity_z,
            episode_return,
            pass_mark,
            remaining_return,
            global_step_obs,
            episode_step_obs,
            health_obs,
        )
        return (
            image,
            velocity_x,
            velocity_y,
            velocity_z,
            episode_return,
            pass_mark,
            remaining_return,
            global_step_obs,
            episode_step_obs,
            health_obs,
        )

    def _to_net_action(self, env_action: np.ndarray) -> np.ndarray:
        """Map a single env action into the policy's normalized action space."""
        return ((env_action - self.action_bias) / self.action_scale).astype(np.float32)

    def _to_env_action(self, net_action: np.ndarray) -> np.ndarray:
        """Map a single normalized policy action into the env's action space."""
        return np.clip(
            net_action * self.action_scale + self.action_bias, self.action_low, self.action_high
        )
