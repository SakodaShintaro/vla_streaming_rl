# SPDX-License-Identifier: MIT
"""
Color Panel Game - Gymnasium Environment

4 colored quadrants (red, green, yellow, blue) with text instruction.
Click the correct color -> reward +1, wrong -> reward -0.01
"""

import numpy as np

from vla_streaming_rl.envs.base_gui_env import BaseGUIEnv

COLORS = {
    "RED": (255, 0, 0),
    "GREEN": (0, 200, 0),
    "YELLOW": (255, 255, 0),
    "BLUE": (0, 0, 255),
}
COLOR_NAMES = list(COLORS.keys())


class ColorPanelEnv(BaseGUIEnv):
    def __init__(self, render_mode):
        super().__init__(render_mode)
        self._window_title = "Color Panel Game"
        self.task_prompt = ""

        half_w = self.width // 2
        half_h = self.height // 2
        self.quadrants = [
            (0, 0, half_w, half_h),
            (half_w, 0, half_w, half_h),
            (0, half_h, half_w, half_h),
            (half_w, half_h, half_w, half_h),
        ]

        self.color_assignment = list(range(4))
        self.correct_color_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.color_assignment = self.np_random.permutation(4).tolist()
        self.correct_color_idx = int(self.np_random.integers(0, 4))
        self._update_task_prompt()
        self.step_count = 0
        self.cursor_x = 0.5
        self.cursor_y = 0.5
        if self.render_mode == "human":
            print(f"\n=== {self.task_prompt} ===")
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
                clicked_color_idx = self.color_assignment[clicked_quadrant]
                if clicked_color_idx == self.correct_color_idx:
                    reward = 1.0
                else:
                    reward = -0.01

            self.color_assignment = self.np_random.permutation(4).tolist()
            self.correct_color_idx = int(self.np_random.integers(0, 4))
            self._update_task_prompt()

        observation = self._render_frame()
        truncated = self.step_count >= 200 if self.render_mode != "human" else False

        if self.render_mode == "human":
            self._render_human(observation)
            print(f"[{self.task_prompt}] reward={reward:.4f}")

        return observation, reward, False, truncated, {"task_prompt": self.task_prompt}

    def _render_frame(self):
        image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        for i, (qx, qy, qw, qh) in enumerate(self.quadrants):
            color_idx = self.color_assignment[i]
            color_name = COLOR_NAMES[color_idx]
            color = COLORS[color_name]
            image[qy : qy + qh, qx : qx + qw] = color
            # Draw border (black)
            image[qy, qx : qx + qw] = (0, 0, 0)
            image[qy + qh - 1, qx : qx + qw] = (0, 0, 0)
            image[qy : qy + qh, qx] = (0, 0, 0)
            image[qy : qy + qh, qx + qw - 1] = (0, 0, 0)

        self._draw_cursor(image)
        return image

    def _update_task_prompt(self):
        target_name = COLOR_NAMES[self.correct_color_idx]
        self.task_prompt = f"Click {target_name}"


if __name__ == "__main__":
    ColorPanelEnv(render_mode="human").run()
