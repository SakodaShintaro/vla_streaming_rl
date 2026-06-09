# SPDX-License-Identifier: MIT
"""
STL-10 Panel Game - Gymnasium Environment

4 quadrants each showing a random STL-10 image from 4 distinct classes.
Click the correct class -> reward +1, wrong -> reward -0.01
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image

from vla_streaming_rl.envs.base_gui_env import BaseGUIEnv

LABEL_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


class STL10PanelEnv(BaseGUIEnv):
    def __init__(self, render_mode, data_dir):
        super().__init__(render_mode)
        self._window_title = "STL-10 Panel Game"
        self.task_prompt = ""

        half_w = self.width // 2
        half_h = self.height // 2
        self.quadrants = [
            (0, 0, half_w, half_h),
            (half_w, 0, half_w, half_h),
            (0, half_h, half_w, half_h),
            (half_w, half_h, half_w, half_h),
        ]

        self.data_dir = Path(data_dir)
        self.image_paths = {}
        for label in LABEL_NAMES:
            label_dir = self.data_dir / label
            self.image_paths[label] = sorted(label_dir.glob("*.png"))

        self.selected_labels = []
        self.quadrant_images = []
        self.correct_quadrant_idx = 0

    def _sample_panel(self):
        self.selected_labels = random.sample(LABEL_NAMES, 4)
        self.quadrant_images = []
        for label in self.selected_labels:
            path = random.choice(self.image_paths[label])
            img = np.array(Image.open(path).convert("RGB"))
            half_w = self.width // 2
            half_h = self.height // 2
            img = np.array(Image.fromarray(img).resize((half_w, half_h), Image.BILINEAR))
            self.quadrant_images.append(img)
        self.correct_quadrant_idx = random.randint(0, 3)
        target_label = self.selected_labels[self.correct_quadrant_idx]
        self.task_prompt = f"Click {target_label}"
        if self.render_mode == "human":
            print(f"\n=== {self.task_prompt} ===")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_panel()
        self.step_count = 0
        self.cursor_x = 0.5
        self.cursor_y = 0.5
        return self._render_frame(), {"task_prompt": self.task_prompt}

    def step(self, action):
        self.step_count += 1
        dx, dy, button = action
        self._update_cursor(dx, dy)
        x, y = self._cursor_pixel()
        current_button_state = button > 0.0

        reward = 0.0

        if current_button_state:
            clicked_quadrant = None
            for i, (qx, qy, qw, qh) in enumerate(self.quadrants):
                if qx <= x < qx + qw and qy <= y < qy + qh:
                    clicked_quadrant = i
                    break

            if clicked_quadrant is not None:
                if clicked_quadrant == self.correct_quadrant_idx:
                    reward = 1.0
                else:
                    reward = -0.01
                if self.render_mode == "human":
                    print(
                        f"Clicked quadrant {clicked_quadrant} ({self.selected_labels[clicked_quadrant]}), reward={reward:.4f}"
                    )

            self._sample_panel()

        observation = self._render_frame()
        truncated = self.step_count >= 200 if self.render_mode != "human" else False

        if self.render_mode == "human":
            self._render_human(observation)

        return observation, reward, False, truncated, {"task_prompt": self.task_prompt}

    def _render_frame(self):
        image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        for i, (qx, qy, qw, qh) in enumerate(self.quadrants):
            image[qy : qy + qh, qx : qx + qw] = self.quadrant_images[i]
            image[qy, qx : qx + qw] = (0, 0, 0)
            image[qy + qh - 1, qx : qx + qw] = (0, 0, 0)
            image[qy : qy + qh, qx] = (0, 0, 0)
            image[qy : qy + qh, qx + qw - 1] = (0, 0, 0)

        self._draw_cursor(image)
        return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    STL10PanelEnv(render_mode="human", data_dir=args.data_dir).run()
