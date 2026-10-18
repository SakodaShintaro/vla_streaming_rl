# SPDX-License-Identifier: MIT
"""環境ごとの、方策が画像と並べて読む指示の文。"""

ENV_INSTRUCTIONS = {
    "CarRacing-v3": (
        "You control the red car in CarRacing-v3 (top-down). Stay on the gray road "
        "and avoid going onto the green grass; hug the road center when possible."
    ),
    "AnimalAI-v0": (
        "You control the agent in Animal-AI (first-person view). "
        "Health drains every step and the episode fails when it reaches 0. "
        "Green and yellow spheres are goals that give reward: green ones look "
        "yellow-green and touching one gives its reward and ends the episode; yellow "
        "ones are bright yellow and the episode goes on after touching them. Red "
        "spheres and red zones are penalties that end the episode as a failure, never "
        "touch them. "
        "A sphere's reward equals its size: between goals the bigger one is worth more. "
        "Turn until the target is at the center of the view, then move forward "
        "with keeping it centered."
    ),
    "CARLA-Leaderboard-v0": (
        "Drive a car along a route in CARLA. Follow the planned route, "
        "obey traffic rules, and avoid collisions."
    ),
}
# 鳥瞰図も観測にあるとき（Animal-AI の topdown_observation）に指示に足す、2枚の画像が何かを
# 伝える文。自キャラクターは env の AGENT_SKIN（hedgehog）に固定している
TOPDOWN_INSTRUCTION = (
    "The first image is your first-person view and the second is the whole arena seen "
    "from straight above. In the view from above you are the dark spiky hedgehog, and its "
    "light-colored face points the way you are facing."
)
