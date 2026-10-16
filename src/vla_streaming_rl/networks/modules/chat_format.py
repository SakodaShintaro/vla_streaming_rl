# SPDX-License-Identifier: MIT
"""VLM に読ませる会話のメッセージの形と、返答から行動を読む決まり。"""

from collections.abc import Callable

import numpy as np

# 思考を閉じるタグ。返答のうち、これより後ろが行動
THINK_CLOSE = "</think>"


def read_action_reply(
    reply_text: str,
    parse_action_text: Callable[[str], tuple[np.ndarray, bool]],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[str, np.ndarray, bool]:
    """The one-step action a reply names, clipped to the action bounds: the
    reply itself, or what follows its thinking if it wrote any. A reply that
    named no runnable action yields standing still."""
    answer_text = reply_text.rsplit(THINK_CLOSE, 1)[-1].strip()
    action, parse_ok = parse_action_text(answer_text)
    if parse_ok:
        action = np.clip(action.astype(np.float32), action_low, action_high)
    else:
        action = np.zeros(action_low.shape, dtype=np.float32)
    return answer_text, action, parse_ok


def assistant_turn(text: str) -> dict:
    """A reply as the message the conversation holds it as."""
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def user_turn(seconds: float, images: list, text: str) -> dict:
    """A turn the env puts to the agent: the frames under their timestamp, then
    the turn's text. The timestamp comes first and is written the way the model's
    video frames carry theirs, which is the order it learned them in.

    1ステップの観測の画像（IMAGE_OBS_NAMES の並び）をこの順に並べる。どれが何の画像かは
    タスクの文が伝える。"""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"<{seconds:.1f} seconds>"},
            *[{"type": "image", "image": image} for image in images],
            {"type": "text", "text": text},
        ],
    }
