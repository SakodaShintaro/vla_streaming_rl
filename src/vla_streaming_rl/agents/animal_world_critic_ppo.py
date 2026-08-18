# SPDX-License-Identifier: MIT
"""The Animal-AI PPO agent with WCM's world-critic objective, ported from ``~/work/WCM``.

WCM trains its critic representation on two objectives at once: the value of the
current state from the history, and the next latent state given the action. Here
PPO already supplies the first -- the value head reads the LSTM output, which no
action ever reaches -- so this agent adds the second as an auxiliary loss on the
same minibatch the PPO update already runs.

Per minibatch sequence, and only for the step pairs that stay inside one episode:

- the dynamics head predicts the next step's visual latent from the current
  latent, the recurrent context and the action actually taken, against the
  encoder's own detached latent, and
- SIGReg keeps the context latents isotropic. WCM needs it because a latent that
  is its own prediction target can be satisfied by collapsing to a constant, and
  nothing else in this loss forbids that.

``seq_len`` is what bounds this: a window of ``n`` steps contributes ``n - 1``
prediction pairs, so ``seq_len`` of 1 has nothing to train on and small values
train on little. It has to be raised above what plain PPO runs with.
"""

import gymnasium as gym
import torch
from torch import nn

from vla_streaming_rl.agents.animal_ppo import AnimalPPOAgent, Rollout


class AnimalWorldCriticPPOAgent(AnimalPPOAgent):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Box,
        network: nn.Module,
        learning_mode: str,
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
        reward_bonus: float,
        ramps_coef: float,
        back_move_coef: float,
        next_state_coef: float,
        sigreg_coef: float,
    ) -> None:
        super().__init__(
            action_space=action_space,
            network=network,
            learning_mode=learning_mode,
            horizon=horizon,
            steps_num=steps_num,
            minibatch_size=minibatch_size,
            mini_epochs=mini_epochs,
            seq_len=seq_len,
            gamma=gamma,
            lam=lam,
            learning_rate=learning_rate,
            e_clip=e_clip,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            clip_value=clip_value,
            normalize_advantage=normalize_advantage,
            max_grad_norm=max_grad_norm,
            velocity_scale=velocity_scale,
            health_scale=health_scale,
            reward_bonus=reward_bonus,
            ramps_coef=ramps_coef,
            back_move_coef=back_move_coef,
        )
        assert seq_len >= 2, (
            f"seq_len {seq_len} leaves no next-state pair for the world-critic loss"
        )
        self.next_state_coef = next_state_coef
        self.sigreg_coef = sigreg_coef

    def _forward_minibatch(
        self, rollout: Rollout, flat: torch.Tensor, sequences: torch.Tensor, env_num: int
    ) -> tuple:
        logits, values, _, state_latent, context_latent = self.network.forward_with_latents(
            rollout.visual[flat],
            rollout.vels[flat],
            rollout.states[sequences],
            rollout.dones[flat],
            env_num,
        )
        state_latent = state_latent.reshape(env_num, self.seq_len, -1)
        context_latent = context_latent.reshape(env_num, self.seq_len, -1)
        actions = rollout.actions[flat].reshape(env_num, self.seq_len)
        fresh = rollout.dones[flat].reshape(env_num, self.seq_len)

        predicted = self.network.predict_next_state(
            state_latent[:, :-1], context_latent[:, :-1], actions[:, :-1]
        )
        target = state_latent[:, 1:].detach()
        valid = (fresh[:, 1:] < 0.5).to(predicted.dtype).unsqueeze(-1)
        squared = (predicted - target).square() * valid
        next_state_loss = squared.sum() / valid.sum().clamp_min(1.0) / predicted.shape[-1]

        sigreg_loss = self.network.sigreg_loss(context_latent)

        auxiliary = self.next_state_coef * next_state_loss + self.sigreg_coef * sigreg_loss
        reported = {
            "next_state": float(next_state_loss.item()),
            "sigreg": float(sigreg_loss.item()),
        }
        return logits, values.squeeze(-1), auxiliary, reported
