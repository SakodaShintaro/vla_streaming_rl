# SPDX-License-Identifier: MIT
"""What the run tells itself about an arena it just failed.

An episode that fell short of its pass mark is played back to a hosted model as
video, along with the instruction the agent had been given, and what comes back
is the instruction for the next attempt at that arena. The run then prompts with
that instead: the arena's standing task text is replaced, and the agent reads
what the last attempt at this arena was worth rather than the same sentence it
already failed on.

Only the task sentence is replaced. The framing and the answer protocol are the
run's own and stay as they are, so a rewritten task cannot cost the agent the
format its action is read out of.

The verdict is never asked for -- the run knows it from the return and the pass
mark, and a model asked to judge an episode calls it a success whenever a goal
sphere is in shot. What is asked for is the part the run cannot supply.
"""

from pathlib import Path

import imageio
import numpy as np
from omegaconf import DictConfig

from vla_streaming_rl.agents.vlm_backends import OpenRouterBackend

# The shortest clip the provider accepts. A short episode is played back slower
# rather than padded: every frame of it still goes over, and only how long each
# is on screen changes.
MIN_VIDEO_SECONDS = 4.0

CRITIC_PROMPT = (
    "This is one episode of an agent in Animal-AI, played back from what the "
    "agent itself saw, at {fps:.1f} frames per second -- one second of video is "
    "{fps:.1f} steps of the episode. The agent acts once per step, choosing one "
    "move (stand still / walk forward / walk backward) and one rotation (no turn "
    "/ turn right / turn left), applied together.\n\n"
    "It was told: {task}\n\n"
    "It FAILED: the episode ended on a return of {score:+.3f} against a pass "
    "mark of {pass_mark:+.3f}. Take that as given -- do not dispute it, and do "
    "not describe the agent as having succeeded.\n\n"
    "Write the instruction to give it for its next attempt at this same arena. "
    "Say what to do, including what to do first and what not to do before that, "
    "in at most three sentences of plain imperative English. Write only the "
    "instruction: no preamble, no explanation of the failure, no formatting, and "
    "nothing about <answer> or the reply format."
)


class EpisodeCritic:
    """The advice held per arena, and the model that writes it."""

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        reasoning_max_tokens: int,
        temperature: float,
        api_max_retries: int,
        body_max_retries: int,
        fps: int,
        video_dir: Path,
    ) -> None:
        self.backend = OpenRouterBackend(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            reasoning_max_tokens=reasoning_max_tokens,
            temperature=temperature,
            api_max_retries=api_max_retries,
            body_max_retries=body_max_retries,
        )
        self.fps = fps
        self.video_dir = video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _encode(self, obs_list: list[np.ndarray], path: Path) -> float:
        """The episode's own frames as one mp4, and the rate it plays back at."""
        frames = [(obs.transpose(1, 2, 0) * 255).astype(np.uint8) for obs in obs_list]
        rate = min(float(self.fps), len(frames) / MIN_VIDEO_SECONDS)
        imageio.mimsave(str(path), frames, fps=rate, macro_block_size=1)
        return rate

    def review(
        self,
        *,
        arena_name: str,
        task: str,
        obs_list: list[np.ndarray],
        score: float,
        pass_mark: float,
        episode_id: int,
    ) -> str:
        """The instruction for the next attempt at ``arena_name``."""
        path = self.video_dir / f"{episode_id:08d}_{arena_name}.mp4"
        rate = self._encode(obs_list, path)
        prompt = CRITIC_PROMPT.format(fps=rate, task=task, score=score, pass_mark=pass_mark)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        advice = self.backend.generate(messages).text.strip()
        path.with_suffix(".txt").write_text(
            f"score {score:+.4f}   pass_mark {pass_mark:+.4f}   frames {len(obs_list)}\n"
            f"was told: {task}\n\nadvice:\n{advice}\n"
        )
        return advice


def build_episode_critic(args: DictConfig, result_dir: Path):
    return EpisodeCritic(
        model_id=args.critic_model_id,
        max_new_tokens=args.critic_max_new_tokens,
        reasoning_max_tokens=args.critic_reasoning_max_tokens,
        temperature=args.critic_temperature,
        api_max_retries=args.api_max_retries,
        body_max_retries=args.body_max_retries,
        fps=args.critic_fps,
        video_dir=result_dir / "critic",
    )
