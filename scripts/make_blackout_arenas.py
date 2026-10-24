# SPDX-License-Identifier: MIT
"""暗転の課題（レベル 14）のアリーナの YAML を書き出す。

エージェントはアリーナの角から対角線の向きに置く。緑と赤を最大の大きさで 1 個ずつ、エージェントの
すぐ前の左右に置き、14-01-01 では緑が右前で赤が左前、14-02-01 ではその逆にする。最初の
1 ステップだけ止まったまま明るい画面で配置を見て、そこから最後まで暗転し、同時に動けるように
なる。暗転の中で、見えていた緑の側を覚えて取りに行き、赤を避ける。
暗転（``blackouts``）と停止（``frozenAgentDelays``）は Python の環境の側でステップ数として行う
（env.py の ``lights_on`` と ``without_python_side_keys``）。競技のアリーナと同じ
``external/animal-ai/configs/competition`` に書く。表を変えたら、これを実行し直す。

    uv run python scripts/make_blackout_arenas.py
"""

import math

from vla_streaming_rl.envs.animalai.curriculum import COMPETITION_DIR

# 体力が尽きるまでのステップ数。全開で進めば 20 ステップで 27 ほど進むので、対角線の端（約 55）
# まで届く
BLACKOUT_ARENA_T = 100

# 緑と赤の大きさ。競技のアリーナで使われている最大の 5 にして、遠くからでも見え、触れやすくする。
# 報酬は大きさに比例するので、緑は 5、赤は -5 になる
BLACKOUT_GOAL_SIZE = 5

# 最初の 1 ステップだけ止まって明るい画面を見て、そのステップから最後まで暗転する
BLACKOUT_VISIBLE_STEPS = 1

# エージェントの位置と向き（度。0 が +z、時計回り）
BLACKOUT_AGENT_X = 1.0
BLACKOUT_AGENT_Z = 1.0
BLACKOUT_AGENT_ROTATION = 45.0
# 緑と赤を置く位置の、エージェントから見た前方と左右へのずれ。玉の直径が 5 なので、玉どうしが
# 重ならず（中心の間が 8。Unity は重なりの判定に余裕を持たせ、6 では赤を置かなかった）、
# エージェントとも壁とも重ならない近さにする
BLACKOUT_GOAL_FORWARD = 7.0
BLACKOUT_GOAL_SIDE = 4.0

# 課題ごとの、緑を右前に置くか（False なら左前。赤はその反対側）
BLACKOUT_TASKS = {"14-01-01": True, "14-02-01": False}


def _goal_position(side: float) -> tuple[float, float]:
    """エージェントから見て前方に ``BLACKOUT_GOAL_FORWARD``、右に ``side``（負なら左）ずれた
    位置の (x, z)。"""
    yaw = math.radians(BLACKOUT_AGENT_ROTATION)
    forward = (math.sin(yaw), math.cos(yaw))
    right = (math.cos(yaw), -math.sin(yaw))
    x = BLACKOUT_AGENT_X + BLACKOUT_GOAL_FORWARD * forward[0] + side * right[0]
    z = BLACKOUT_AGENT_Z + BLACKOUT_GOAL_FORWARD * forward[1] + side * right[1]
    return round(x, 2), round(z, 2)


def blackout_arena_yaml(green_on_right: bool) -> str:
    side = BLACKOUT_GOAL_SIDE if green_on_right else -BLACKOUT_GOAL_SIDE
    green_x, green_z = _goal_position(side)
    red_x, red_z = _goal_position(-side)
    size = BLACKOUT_GOAL_SIZE
    return f"""!ArenaConfig
arenas:
  0: !Arena
    pass_mark: 0
    t: {BLACKOUT_ARENA_T}
    blackouts: [{BLACKOUT_VISIBLE_STEPS}]
    items:
    - !Item
      name: Agent
      positions:
      - !Vector3 {{x: {BLACKOUT_AGENT_X}, y: 0, z: {BLACKOUT_AGENT_Z}}}
      rotations: [{BLACKOUT_AGENT_ROTATION}]
      frozenAgentDelays: [{BLACKOUT_VISIBLE_STEPS}]
    - !Item
      name: GoodGoal
      positions:
      - !Vector3 {{x: {green_x}, y: 0, z: {green_z}}}
      sizes:
      - !Vector3 {{x: {size}, y: {size}, z: {size}}}
    - !Item
      name: BadGoal
      positions:
      - !Vector3 {{x: {red_x}, y: 0, z: {red_z}}}
      sizes:
      - !Vector3 {{x: {size}, y: {size}, z: {size}}}
"""


if __name__ == "__main__":
    COMPETITION_DIR.mkdir(parents=True, exist_ok=True)
    for name, green_on_right in BLACKOUT_TASKS.items():
        (COMPETITION_DIR / f"{name}.yaml").write_text(blackout_arena_yaml(green_on_right))
    print(f"wrote {len(BLACKOUT_TASKS)} arenas to {COMPETITION_DIR}")
