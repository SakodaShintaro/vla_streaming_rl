# SPDX-License-Identifier: MIT
"""The reward an Animal-AI agent trains on.

What the environment reports is the arena's own reward, which is what a score is taken
from. Everything here is training signal, so it sits on the agent's side of that boundary,
and every learning rule shares it so that they optimize the same thing.
"""

# The first three terms are the winning agent's (``~/work/rl_animal``, ``EnvConfig``); the
# pass-mark term is this project's.
GOAL_BONUS = 0.5
RAMPS_COEF = 0.01
BACK_MOVE_COEF = 0.001
PASS_MARK_BONUS = 1.0


def shape_animal_reward(reward: float, obs: dict, episode_done: bool) -> float:
    """Reaching a goal is worth more than the arena says, climbing is encouraged, walking
    backwards is discouraged, and an episode ends with a bonus for having cleared the
    arena's pass mark or a penalty for not."""
    velocity = obs["velocity"]
    if reward > 0.1:
        reward += GOAL_BONUS
    if velocity[1] > 0.01:
        reward += float(velocity[1]) * RAMPS_COEF
    if velocity[2] < 0:
        reward += float(velocity[2]) * BACK_MOVE_COEF
    if episode_done:
        # obs carries the arena's own return including this tick, on the pass mark's scale
        cleared = float(obs["episode_return"]) >= float(obs["pass_mark"])
        reward += PASS_MARK_BONUS if cleared else -PASS_MARK_BONUS
    return reward
