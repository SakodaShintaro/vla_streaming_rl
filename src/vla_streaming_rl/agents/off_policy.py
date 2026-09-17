# SPDX-License-Identifier: MIT
"""Off-policy learning mode: act now, learn later from a large replay buffer.

Every tick is stored; once ``learning_starts`` ticks have gone by, one gradient
step on a ``batch_size`` sample of the buffer fires every ``horizon`` ticks,
before the action of that tick is chosen. Below ``learning_starts`` the env is
driven by uniform random actions, so the buffer fills with something other than
an untrained policy's output while the network's recurrent state still follows
the episode. With ``text_action`` on, the action the chain of thought names in
its ``<answer>`` is a second candidate, held for the chain's whole cadence: it
drives the env alone below ``learning_starts``, and from then on every tick
runs whichever of it and the head's action the critic values higher.

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

from vla_streaming_rl.agents.base import Agent, StepResult
from vla_streaming_rl.agents.prompt import ANSWER_RE, PromptBuilder
from vla_streaming_rl.networks.interface import InferInput
from vla_streaming_rl.networks.modules.cot_batch import CoTBatch
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.utils import render_selection_panel


def _format_action(action: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:+.2f}" for value in action) + "]"


class OffPolicyAgent(Agent):
    SELECTION_PANEL_WIDTH = 320
    SELECTION_PANEL_HEIGHT = 560

    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
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
        text_action: bool,
        parse_action_text,
    ) -> None:
        super().__init__(
            horizon=horizon,
            reset_on_episode_end=reset_on_episode_end,
            prompt_builder=prompt_builder,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_action = text_action
        if text_action:
            assert isinstance(network.cot_module, CoTBatch), (
                "text_action reads the action off a finished chain, which only "
                "cot_mode=batch writes; set cot_tokens_num > 0 and cot_mode=batch"
            )
        self.parse_action_text = parse_action_text
        self.vlm_action = np.zeros(int(np.prod(action_space.shape)), dtype=np.float32)
        self.vlm_answer_text = ""
        self.text_parse_failed = 0.0
        self.selection_status = ""
        self.selection_rows = []
        self.decisions_num = 0
        self.vlm_chosen_num = 0

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
            self.decisions_num = 0
            self.vlm_chosen_num = 0
        if episode_done:
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
        # The chain reads its prompt off the rows just stored, and what it
        # writes completes this tick's row.
        cot_activation, cot_age = self.network.advance_cot(
            episode_started, self.rb.get_latest(self.seq_len)
        )
        self.rb.amend_latest(
            cot_activation, cot_age, self.network.tokenize(self.network.thought_text())
        )
        if self.text_action and cot_age == 0:
            self._read_vlm_action()
        if self.text_action:
            metrics["text/parse_failed"] = self.text_parse_failed

        warmup = global_step < self.learning_starts

        if not warmup and self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self._to_env_action(self.action_chunk[self.chunk_step])
            self.prev_action = action
            self.chunk_step += 1
            metrics["chunk_step"] = self.chunk_step
            return StepResult(
                action=action,
                metrics=metrics,
                panels=self._panels(),
                texts={"prompt": prompt, **self.network.render_texts()},
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
                cot_activations_seq=latest_data.cot_activations,
                cot_age_seq=latest_data.cot_age,
            )
        )
        self.rnn_state = infer_result.rnn_state
        self.last_features = infer_result.features
        metrics.update(infer_result.value_report)
        action_chunk = infer_result.action[0].cpu().numpy()
        if self.text_action:
            vlm_chunk = np.repeat(self._to_net_action(self.vlm_action)[None], self.horizon, axis=0)
            q_vlm = self.network.action_value(infer_result.features, vlm_chunk)
            q_head = self.network.action_value(infer_result.features, action_chunk)
            vlm_chosen = warmup or q_vlm >= q_head
            metrics["select/q_vlm"] = q_vlm
            metrics["select/q_head"] = q_head
            metrics["select/vlm_chosen"] = float(vlm_chosen)
            self.decisions_num += 1
            self.vlm_chosen_num += int(vlm_chosen)
            self.selection_status = (
                f"step {global_step}, chain age {cot_age}, "
                f"{'warmup: VLM only' if warmup else 'chosen by Q'}. "
                f"VLM chosen {self.vlm_chosen_num}/{self.decisions_num} this episode."
            )
            self.selection_rows = [
                (
                    "VLM",
                    f"{self.vlm_answer_text}  {_format_action(self.vlm_action)}",
                    q_vlm,
                    vlm_chosen,
                ),
                (
                    "head",
                    _format_action(self._to_env_action(action_chunk[0])),
                    q_head,
                    not vlm_chosen,
                ),
            ]
            if vlm_chosen:
                action_chunk = vlm_chunk
        elif warmup:
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
            panels=self._panels(),
            texts={"prompt": prompt, **self.network.render_texts()},
        )

    def _panels(self) -> dict[str, np.ndarray]:
        """The network's panels, plus the two candidates and their action values
        when the chain's answer is one: the same keys on every step of a run."""
        panels = self.network.render_panels()
        if self.text_action:
            panels["selection"] = render_selection_panel(
                self.selection_status,
                self.selection_rows,
                self.SELECTION_PANEL_WIDTH,
                self.SELECTION_PANEL_HEIGHT,
            )
        return panels

    def _read_vlm_action(self) -> None:
        """Read the action the chain just written names, the VLM policy's
        candidate until the next chain. A reply that named no runnable action
        makes the candidate standing still, and is answered by the env in its
        own turn."""
        answer_match = ANSWER_RE.search(self.network.thought_text())
        answer_text = answer_match.group(1).strip() if answer_match is not None else ""
        action_array, parse_ok = self.parse_action_text(answer_text)
        self.vlm_answer_text = answer_text if parse_ok else f"(unparsed: {answer_text})"
        if parse_ok:
            self.vlm_action = np.clip(
                action_array[0].astype(np.float32), self.action_low, self.action_high
            )
        else:
            self.vlm_action = np.zeros(self.action_dim, dtype=np.float32)
            self.prompt_builder.reject(answer_text)
        self.text_parse_failed = float(not parse_ok)

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
