# SPDX-License-Identifier: MIT
"""The Animal-AI Olympics winning agent, ported from ``~/work/rl_animal``.

There it was an epoch loop over 24 parallel environments; here the trainer owns
the loop and hands the agent one transition at a time, so the rollout is
buffered inside the agent and the PPO update fires once ``steps_num``
transitions are in. That is the whole of this on-policy rule.

Two things the original environment did for the network live here instead,
because this repo's Animal-AI env deliberately reports what the arena reports:

- the velocity/clock vector the dense branch reads (scaled velocity plus the
  agent's health, which stands in for the episode clock), and
- the reward the update is trained on. The score the trainer logs stays the
  arena's own reward; only the returns the critic fits see the shaping.
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from vla_streaming_rl.agents.animal_reward import shape_animal_reward
from vla_streaming_rl.agents.base import Agent, StepResult

# Animal-AI's native action is MultiDiscrete([3, 3]): one move (noop / forward /
# back) and one rotation (noop / right / left). The flat Discrete(9) the network
# emits splits as (a // 3, a % 3), and the env takes Box(-1, 1, shape=(2,)) and
# discretizes back with a +/-1/3 dead zone, so each branch value maps onto the
# extreme Box value that survives it.
_BRANCH_TO_BOX = np.array([0.0, 1.0, -1.0], dtype=np.float32)


@dataclass
class Pending:
    """The half of a transition that is known before the environment answers.

    ``done`` and ``rnn_state`` are the ones that held *before* the action was
    chosen: they are what the recurrent unrolling replays.
    """

    visual: torch.Tensor
    vels: torch.Tensor
    rnn_state: torch.Tensor
    done: float
    action: int
    value: float
    neglogpac: float


class RolloutBuffer:
    """One on-policy rollout of a single environment, and the advantage pass
    that turns it into the tensors the update indexes."""

    def __init__(self) -> None:
        self.visual: list[torch.Tensor] = []
        self.vels: list[torch.Tensor] = []
        self.states: list[torch.Tensor] = []
        self.dones: list[float] = []
        self.actions: list[int] = []
        self.values: list[float] = []
        self.neglogpacs: list[float] = []
        self.rewards: list[float] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(self, pending: Pending, reward: float) -> None:
        self.visual.append(pending.visual)
        self.vels.append(pending.vels)
        self.states.append(pending.rnn_state)
        self.dones.append(pending.done)
        self.actions.append(pending.action)
        self.values.append(pending.value)
        self.neglogpacs.append(pending.neglogpac)
        self.rewards.append(reward)

    def finish(
        self, last_value: float, final_done: float, gamma: float, lam: float, seq_len: int
    ) -> "Rollout":
        """Generalized advantage estimation over the collected steps. ``seq_len``
        is the window the LSTM is unrolled over during the update: one recurrent
        state is kept per window, the one recorded at its first step."""
        steps_num = len(self)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(steps_num)):
            if t == steps_num - 1:
                next_non_terminal = 1.0 - final_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            running = delta + gamma * lam * next_non_terminal * running
            advantages[t] = running

        device = self.visual[0].device
        return Rollout(
            visual=torch.stack(self.visual),
            vels=torch.stack(self.vels),
            states=torch.stack(self.states)[::seq_len],
            dones=torch.as_tensor(dones, device=device),
            actions=torch.as_tensor(np.asarray(self.actions), device=device).long(),
            values=torch.as_tensor(values, device=device),
            neglogpacs=torch.as_tensor(
                np.asarray(self.neglogpacs, dtype=np.float32), device=device
            ),
            returns=torch.as_tensor(advantages + values, device=device),
        )


@dataclass
class Rollout:
    """One finished rollout, on the training device."""

    visual: torch.Tensor
    vels: torch.Tensor
    states: torch.Tensor
    dones: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor
    neglogpacs: torch.Tensor
    returns: torch.Tensor


class AnimalPPOAgent(Agent):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        network: nn.Module,
        horizon: int,
        steps_num: int,
        minibatch_size: int,
        mini_epochs: int,
        seq_len: int,
        gamma: float,
        lam: float,
        learning_rate: float,
        e_clip: float,
        entropy_coef: float,
        critic_coef: float,
        clip_value: bool,
        normalize_advantage: bool,
        max_grad_norm: float,
        velocity_scale: list[float],
        health_scale: float,
    ) -> None:
        super().__init__(horizon=horizon)
        assert steps_num % seq_len == 0, f"steps_num {steps_num} is not a multiple of {seq_len}"
        assert minibatch_size % seq_len == 0, (
            f"minibatch_size {minibatch_size} is not a multiple of seq_len {seq_len}"
        )
        assert steps_num >= minibatch_size, (
            f"minibatch_size {minibatch_size} exceeds the rollout of {steps_num} steps"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)

        self.steps_num = steps_num
        self.minibatch_size = minibatch_size
        self.mini_epochs = mini_epochs
        self.seq_len = seq_len
        self.gamma = gamma
        self.lam = lam
        self.e_clip = e_clip
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.clip_value = bool(clip_value)
        self.normalize_advantage = bool(normalize_advantage)
        self.max_grad_norm = max_grad_norm

        self.velocity_scale = np.asarray(velocity_scale, dtype=np.float32)
        self.health_scale = health_scale

        self.rnn_state = self.network.init_state(1, self.device)
        self.buffer = RolloutBuffer()
        self.pending: Pending | None = None
        # the first observation of a run starts an episode
        self._previous_done = True

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
        visual, vels = self._preprocess(obs, info)
        shaped = shape_animal_reward(reward, obs, terminated or truncated)
        # this observation is the outcome of the action chosen on the previous
        # tick, so that transition can only be completed now
        self.buffer.add(self.pending, shaped)

        fresh = self._episode_boundary(terminated, truncated)
        metrics = {"shaped_reward": shaped}
        action = self._act(visual, vels, fresh)

        if len(self.buffer) >= self.steps_num:
            # the value just computed for this observation is the bootstrap the
            # last collected step needs, and a transition that ended the episode
            # cuts it
            rollout = self.buffer.finish(
                self.pending.value,
                max(fresh, float(terminated or truncated)),
                self.gamma,
                self.lam,
                self.seq_len,
            )
            self.buffer = RolloutBuffer()
            metrics.update(self._update(rollout))

        return StepResult(action=action, metrics=metrics, panels={})

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
        """Act without learning. The trainer calls this once per episode after
        the reset; the testbed calls it on every tick. Both are covered by the
        same boundary rule, and the action chosen on a terminal observation is
        dropped simply by never being buffered."""
        del global_step, reward
        visual, vels = self._preprocess(obs, info)
        fresh = self._episode_boundary(terminated, truncated)
        return StepResult(action=self._act(visual, vels, fresh), metrics={}, panels={})

    def _episode_boundary(self, terminated: bool, truncated: bool) -> float:
        """1 when this observation is the first of an episode, which is what
        clears the recurrent state.

        An observation returned by ``step`` is always a continuation -- a
        terminal observation still belongs to its episode -- so the boundary is
        one tick later, when the driver has reset and hands over the fresh
        observation. Only the previous tick's flags can tell the two apart, and
        the very first observation of a run is a boundary by definition.
        """
        fresh = float(self._previous_done)
        self._previous_done = bool(terminated or truncated)
        return fresh

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        del score, feedback_text
        return {}

    def optimizer_state_dict(self) -> dict:
        return {"ppo": self.optimizer.state_dict()}

    def load_optimizer_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["ppo"])

    # --- per-tick machinery ------------------------------------------------

    @torch.no_grad()
    def _act(self, visual: torch.Tensor, vels: torch.Tensor, fresh: float) -> np.ndarray:
        """Sample an action for one observation and stash everything the update
        will need about it. ``fresh`` clears the recurrent state carried in."""
        state = self.rnn_state
        logits, value, self.rnn_state = self.network(
            visual.unsqueeze(0),
            vels.unsqueeze(0),
            state,
            torch.tensor([fresh], device=self.device),
            1,
        )
        action = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        neglogpac = F.cross_entropy(logits, action, reduction="none")
        self.pending = Pending(
            visual=visual,
            vels=vels,
            rnn_state=state.squeeze(0),
            done=fresh,
            action=int(action.item()),
            value=float(value.squeeze().item()),
            neglogpac=float(neglogpac.item()),
        )
        return self._to_env_action(self.pending.action)

    def _preprocess(self, obs: dict[str, Any], info: dict) -> tuple:
        """The image as the wrappers already produce it, and the velocity/clock
        vector the dense branch reads.

        The original counted the arena's own ``t`` down and fed the remainder;
        this env does not expose an episode length, so health takes that slot.
        It drains over an episode the same way, and unlike a step count it is
        bounded and already an observation here.
        """
        del info
        visual = torch.from_numpy(obs["image"]).to(self.device)
        velocity = obs["velocity"].astype(np.float32)
        clock = obs["health"].astype(np.float32) / self.health_scale
        vels = np.concatenate([velocity / self.velocity_scale, clock]).astype(np.float32)
        return visual, torch.from_numpy(vels).to(self.device)

    def _to_env_action(self, net_action: int) -> np.ndarray:
        return np.array(
            [_BRANCH_TO_BOX[net_action // 3], _BRANCH_TO_BOX[net_action % 3]], dtype=np.float32
        )

    # --- update ------------------------------------------------------------

    def _update(self, rollout: Rollout) -> dict:
        advantages = rollout.returns - rollout.values
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_sequences = len(rollout.returns) // self.seq_len
        sequences_per_batch = self.minibatch_size // self.seq_len
        num_minibatches = len(rollout.returns) // self.minibatch_size
        step_indexes = np.arange(total_sequences * self.seq_len).reshape(
            total_sequences, self.seq_len
        )

        losses: dict[str, list[float]] = {}
        for _ in range(self.mini_epochs):
            order = np.random.permutation(total_sequences)
            for minibatch in range(num_minibatches):
                chosen = order[
                    minibatch * sequences_per_batch : (minibatch + 1) * sequences_per_batch
                ]
                flat = torch.as_tensor(step_indexes[chosen].ravel(), device=self.device)
                reported = self._minibatch_step(
                    rollout,
                    advantages,
                    flat,
                    torch.as_tensor(chosen, device=self.device),
                    len(chosen),
                )
                for name, value in reported.items():
                    if name not in losses:
                        losses[name] = []
                    losses[name].append(value)

        return {f"ppo/{name}": float(np.mean(values)) for name, values in losses.items()}

    def _minibatch_step(
        self,
        rollout: Rollout,
        advantages: torch.Tensor,
        flat: torch.Tensor,
        sequences: torch.Tensor,
        sequence_num: int,
    ) -> dict:
        logits, values, auxiliary_loss, auxiliary_reported = self.network.forward_for_update(
            rollout.visual[flat],
            rollout.vels[flat],
            rollout.states[sequences],
            rollout.dones[flat],
            rollout.actions[flat],
            sequence_num,
        )

        neglogpacs = F.cross_entropy(logits, rollout.actions[flat], reduction="none")
        old_neglogpacs = rollout.neglogpacs[flat]
        batch_advantages = advantages[flat]

        ratio = torch.exp(old_neglogpacs - neglogpacs)
        unclipped = -batch_advantages * ratio
        clipped = -batch_advantages * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
        actor_loss = torch.max(unclipped, clipped).mean()

        returns = rollout.returns[flat]
        old_values = rollout.values[flat]
        value_loss = (values - returns) ** 2
        if self.clip_value:
            clipped_values = old_values + torch.clamp(
                values - old_values, -self.e_clip, self.e_clip
            )
            value_loss = torch.max(value_loss, (clipped_values - returns) ** 2)
        critic_loss = value_loss.mean()

        log_probabilities = F.log_softmax(logits, dim=-1)
        entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()

        loss = (
            actor_loss
            + 0.5 * self.critic_coef * critic_loss
            - self.entropy_coef * entropy
            + auxiliary_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            kl = 0.5 * ((old_neglogpacs - neglogpacs) ** 2).mean()

        return {
            "actor": actor_loss.item(),
            "critic": critic_loss.item(),
            "entropy": entropy.item(),
            "kl": kl.item(),
            **auxiliary_reported,
        }
