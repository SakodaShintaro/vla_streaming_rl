# SPDX-License-Identifier: MIT
"""Streaming learning mode: inference and the update share one forward pass.

There is no replay: the buffer holds only the window the update reads (the
latest ``seq_len + horizon`` transitions), and every time a new action chunk is
needed the grad-enabled forward that produces it also produces the loss. With
``use_eligibility_trace`` the critic is optimized by AdamET, which carries the
trace across ticks and is reset on episode boundaries.

The learning mode is the class and the network is a constructor argument, so
this file is one half of the (learning mode) x (network) grid; the off-policy
half is ``off_policy.py``. The two share no base beyond :class:`Agent`, which
costs some repetition in the per-tick path and buys each mode being readable
end to end in one file.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.agents.prompt import PromptBuilder
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.optimizers.adam_et import AdamET
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor


class StreamingAgent(Agent):
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        network: nn.Module,
        normalizing_by_return: bool,
        max_grad_norm: float,
        use_done: bool,
        seq_len: int,
        horizon: int,
        use_eligibility_trace: bool,
        actor_lr: float,
        critic_lr: float,
        weight_decay: float,
        gamma: float,
        et_lambda: float,
        buffer_device: str,
        max_prompt_tokens: int,
        pad_token_id: int,
        reset_on_episode_end: bool,
        prompt_builder: PromptBuilder,
    ) -> None:
        super().__init__(
            horizon=horizon,
            reset_on_episode_end=reset_on_episode_end,
            prompt_builder=prompt_builder,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observation_space = observation_space

        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0
        self.reward_processor = RewardProcessor("scaling", 1.0)
        self.normalizing_by_return = normalizing_by_return

        self.max_grad_norm = max_grad_norm
        self.use_done = use_done

        # Sequence observation management
        self.seq_len = seq_len

        # Action chunking state
        self.action_chunk = None  # (horizon, action_dim) - current action chunk
        self.chunk_step = 0  # current step within chunk

        self.network = network
        self.rnn_state = self.network.init_state().to(self.device)

        # Actor / critic optimizer split (critic == value head). The critic uses
        # AdamET (eligibility traces) when the trace is on, AdamW otherwise; the
        # actor is always AdamW.
        self.use_eligibility_trace = bool(use_eligibility_trace)
        critic_params = list(self.network.value_head.parameters())
        critic_param_ids = {id(p) for p in critic_params}
        actor_params = [p for p in self.network.parameters() if id(p) not in critic_param_ids]
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay)
        if self.use_eligibility_trace:
            self.critic_optimizer = AdamET(
                critic_params, lr=critic_lr, gamma=gamma, et_lambda=et_lambda
            )
        else:
            self.critic_optimizer = optim.AdamW(
                critic_params, lr=critic_lr, weight_decay=weight_decay
            )

        # Only the one window the update trains on is kept.
        self.rb = ReplayBuffer(
            size=seq_len + horizon,
            seq_len=self.seq_len + self.horizon,
            obs_shape=self.network.observation_space_shape,
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            cot_shape=self.network.cot_shape,
            output_device=self.device,
            storage_device=torch.device(buffer_device),
            max_prompt_tokens=max_prompt_tokens,
            pad_token_id=pad_token_id,
        )

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._episode_reset = False
        # the first observation of a run starts an episode
        self._previous_done = True
        # Shared representation fed to policy/value/prediction heads on the
        # most recent select_action inference (used by scripts/probe.py).
        self.last_features: torch.Tensor | None = None

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
        del global_step
        metrics = {}
        # The language this tick: composed here from the env's state, never read
        # off the observation.
        prompt = self.prompt_builder(obs, reward, info)
        episode_done = terminated or truncated
        # What the agent trains on, against what the env reported as its score.
        shaped_reward = info["shaped_reward"]
        metrics["shaped_reward"] = shaped_reward
        self._reset_rnn_state_if_fresh(episode_done)
        if episode_done:
            self.action_chunk = None
            self.chunk_step = 0
            self._episode_reset = self.use_done
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
            task_prompt_token_ids,
        ) = self._preprocess(obs, prompt)
        # The chain of thought advances once per environment step, whether or not
        # this tick needs a new action chunk.
        cot_activation = self.network.advance_cot(image, prompt, episode_done)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        self.rb.add(
            image,
            shaped_reward,
            episode_done if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            task_prompt_token_ids,
            velocity_x,
            velocity_y,
            velocity_z,
            episode_return,
            pass_mark,
            remaining_return,
            global_step_obs,
            episode_step_obs,
            health_obs,
            cot_activation,
        )
        if self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self._to_env_action(self.action_chunk[self.chunk_step])
            self.prev_action = action
            self.chunk_step += 1
            metrics["chunk_step"] = self.chunk_step
            return StepResult(
                action=action,
                metrics=metrics,
                panels=self.network.render_panels(),
                texts={"prompt": prompt, **self.network.render_texts()},
            )

        # new chunk: a single grad-enabled forward yields both the action chunk
        # and the training loss (fused inference + training).
        data = self.rb.get_latest(self.seq_len + self.horizon)
        data.rewards = self.reward_processor.normalize(data.rewards)
        result = self.network.infer_and_compute_loss(data)

        infer_result = result.infer_result
        self.rnn_state = infer_result.rnn_state
        metrics.update(infer_result.value_report)
        action_chunk = infer_result.action[0].cpu().numpy()
        self.action_chunk = action_chunk
        self.chunk_step = 1
        action = self._to_env_action(action_chunk[0])
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step
        metrics.update(result.loss_result.info)

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_eligibility_trace:
            # Actor: backward actor-only loss → encoder + actor grads.
            result.et_info.actor_entropy_loss.backward(retain_graph=True)
            # Critic: backward -V(s) → value_head grads only (detached from encoder).
            result.et_info.neg_value.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step(delta=result.et_info.delta, reset=self._episode_reset)
            self._episode_reset = False
        else:
            result.loss_result.loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return StepResult(
            action=action,
            metrics=metrics,
            panels=self.network.render_panels(),
            texts={"prompt": prompt, **self.network.render_texts()},
        )

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
        """Act without learning: the trainer calls this once per episode after
        the reset, and the testbed calls it on every tick."""
        del global_step
        metrics = {}
        # The language this tick: composed here from the env's state, never read
        # off the observation.
        prompt = self.prompt_builder(obs, reward, info)
        episode_done = terminated or truncated
        # What the agent trains on, against what the env reported as its score.
        shaped_reward = info["shaped_reward"]
        metrics["shaped_reward"] = shaped_reward
        self._reset_rnn_state_if_fresh(episode_done)
        if episode_done:
            self.action_chunk = None
            self.chunk_step = 0
            self._episode_reset = self.use_done
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
            task_prompt_token_ids,
        ) = self._preprocess(obs, prompt)
        # The chain of thought advances once per environment step, whether or not
        # this tick needs a new action chunk.
        cot_activation = self.network.advance_cot(image, prompt, episode_done)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        self.rb.add(
            image,
            shaped_reward,
            episode_done if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            task_prompt_token_ids,
            velocity_x,
            velocity_y,
            velocity_z,
            episode_return,
            pass_mark,
            remaining_return,
            global_step_obs,
            episode_step_obs,
            health_obs,
            cot_activation,
        )

        if self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self._to_env_action(self.action_chunk[self.chunk_step])
            self.prev_action = action
            self.chunk_step += 1
            metrics["chunk_step"] = self.chunk_step
            return StepResult(
                action=action,
                metrics=metrics,
                panels=self.network.render_panels(),
                texts={"prompt": prompt, **self.network.render_texts()},
            )

        latest_data = self.rb.get_latest(self.seq_len)
        infer_result = self.network.infer(
            InferInput(
                s_seq=latest_data.observations,
                a_seq=latest_data.actions,
                r_seq=latest_data.rewards,
                rnn_state=self.rnn_state,
                task_prompts=[prompt],
                velocity_x_seq=latest_data.velocity_x,
                velocity_y_seq=latest_data.velocity_y,
                velocity_z_seq=latest_data.velocity_z,
                episode_return_seq=latest_data.episode_return,
                pass_mark_seq=latest_data.pass_mark,
                remaining_return_seq=latest_data.remaining_return,
                global_step_seq=latest_data.global_step,
                episode_step_seq=latest_data.episode_step,
                health_seq=latest_data.health,
                cot_activations_seq=latest_data.cot_activations,
            )
        )
        self.rnn_state = infer_result.rnn_state
        self.last_features = infer_result.features
        metrics.update(infer_result.value_report)
        action_chunk = infer_result.action[0].cpu().numpy()
        self.action_chunk = action_chunk
        self.chunk_step = 1
        action = self._to_env_action(action_chunk[0])
        self.prev_action = action
        metrics["chunk_step"] = self.chunk_step
        return StepResult(
            action=action,
            metrics=metrics,
            panels=self.network.render_panels(),
            texts={"prompt": prompt, **self.network.render_texts()},
        )

    def _preprocess(self, obs: dict[str, Any], prompt: str) -> tuple:
        """Turn the raw observation into what the replay buffer stores this tick:
        the image tensor, the raw scalar observations (velocity_x, velocity_y,
        velocity_z, episode_return, pass_mark, remaining_return, global_step,
        episode_step,
        health; the network updates its running normalizer stats here) and the
        tokenized form of the prompt the agent composed for this tick."""
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
        task_prompt_token_ids = self.network.tokenize_task_prompt(prompt)
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
            task_prompt_token_ids,
        )

    def _to_env_action(self, net_action: np.ndarray) -> np.ndarray:
        """Map a single normalized policy action into the env's action space."""
        return np.clip(
            net_action * self.action_scale + self.action_bias, self.action_low, self.action_high
        )
