# SPDX-License-Identifier: MIT
"""左右の橋の課題のアリーナの YAML を書き出す。

レベル 11 は課題の組とバリアントの表（組ごとの道幅、バリアントごとの報酬）から作る 30 個、
レベル 12 は振る舞いを調整するための 1 個（``12-01-01``）。競技のアリーナと同じ
``XX-YY-ZZ.yaml`` の名前で、競技のアリーナと同じ ``external/animal-ai/configs/competition``
に書く。表を変えたら、これを実行し直してアリーナを作り直す。どちらの道を選んだかを終わりの
位置から判定する ``bridge_path`` も、あとの分析のためにここに置く。

    uv run python scripts/make_bridge_arenas.py
"""

import math
from dataclasses import dataclass

from vla_streaming_rl.envs.animalai.curriculum import COMPETITION_DIR

# 橋のアリーナの寸法（アリーナは 40 x 40、x が右、z が前）。エージェントは中央から前を
# 向いて始まる。スタートからマグマの手前までは全幅が安全な床、マグマの区間は左右の橋以外が
# マグマ、その先は安全な床で、中央の壁で左右を分ける。2本の橋はスタートから左右の報酬へ
# まっすぐ向かう斜めの道で、スタートでその角度だけ向きを変えて直進すれば渡れる。回転の刻み
# （1 ステップ 10 度）の倍数の角度ならちょうど向けられ、そうでなければ直進だけでは道から
# ずれていく
BRIDGE_ARENA_SIZE = 40.0
BRIDGE_AGENT_Z = 4.0
# マグマは z 方向にこの幅の帯に切って、帯ごとに橋の外側を箱で埋める
BRIDGE_SLICE_Z = 1.0
# ハイリスク側の橋の縁に並べる赤い球の大きさと、奥行きの間隔
BRIDGE_EDGE_SPHERE_SIZE = 1.0
BRIDGE_EDGE_SPHERE_SPACING_Z = 2.0
# レベル 11 のマグマの区間、報酬の奥行き、橋の角度
BRIDGE_LAVA_START_Z = 10.0
BRIDGE_LAVA_END_Z = 32.0
BRIDGE_GOAL_Z = 36.0
BRIDGE_ANGLE_DEG = 20.0
# レベル 11 の橋の課題。タスク番号 YY の組は (YY + 1) // 2 で、奇数はハイリスク側が左、
# 偶数はその左右反転。組ごとの (ハイリスク側の道幅, ローリスク側の道幅) と、ローリスク側の
# 報酬（組 5 だけ報酬なし）
BRIDGE_PAIR_WIDTHS = {
    1: (1.5, 2.5),  # 両方とも細い
    2: (1.5, 5.0),  # 細い と 中くらい
    3: (1.5, 12.0),  # 細い と 太い
    4: (5.0, 12.0),  # 中くらい と 太い
    5: (1.5, 12.0),  # 細い と 太い（ローリスク側は報酬なし）
}
BRIDGE_PAIR_LOW_REWARD = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0}
# バリアントは報酬の大きさだけを変える。Animal-AI の報酬の球でよく使われる大きさ
# （普通 1、大 2、さらに 3 と 5）に合わせ、ハイリスク側を 5・2・3 にする
BRIDGE_VARIANT_HIGH_REWARD = {"01": 5.0, "02": 2.0, "03": 3.0}


@dataclass(frozen=True)
class BridgeTask:
    """橋のアリーナ1つの中身。ハイリスク側の位置、左右の道の幅と報酬と角度、マグマの区間と
    報酬の奥行き。"""

    high_on_right: bool
    width_high: float
    width_low: float
    reward_high: float
    reward_low: float
    angle_high_deg: float
    angle_low_deg: float
    lava_start_z: float
    lava_end_z: float
    # 左右の報酬の奥行き。それぞれの橋の中心線の延長上に置く。ローリスク側をマグマの手前に
    # 置くと、ローリスク側の橋は渡らずに取れる
    goal_z_high: float
    goal_z_low: float
    # ハイリスク側の橋の両側の縁に、触れると失敗する赤い球（BadGoal）を並べるか。マグマは
    # 低い一人称のカメラでは地平線の細い帯にしか見えないので、道の細さを目に見えるようにする
    mark_high_edges: bool


def _level11_task(task: int, variant: str) -> BridgeTask:
    """レベル 11 のタスク番号とバリアントの課題。"""
    pair = (task + 1) // 2
    width_high, width_low = BRIDGE_PAIR_WIDTHS[pair]
    return BridgeTask(
        high_on_right=task % 2 == 0,
        width_high=width_high,
        width_low=width_low,
        reward_high=BRIDGE_VARIANT_HIGH_REWARD[variant],
        reward_low=BRIDGE_PAIR_LOW_REWARD[pair],
        angle_high_deg=BRIDGE_ANGLE_DEG,
        angle_low_deg=BRIDGE_ANGLE_DEG,
        lava_start_z=BRIDGE_LAVA_START_Z,
        lava_end_z=BRIDGE_LAVA_END_Z,
        goal_z_high=BRIDGE_GOAL_Z,
        goal_z_low=BRIDGE_GOAL_Z,
        mark_high_edges=False,
    )


# アリーナ名ごとの課題。レベル 12 は、1回目は大きい報酬のハイリスク側を選んで落ち、
# 振り返りのあとでローリスク側に切り替える、という振る舞いを引き出すための調整用。
# ハイリスク側は細く、角度を回転の刻みの倍数からずらして直進だけでは渡れなくする。
# ローリスク側の小さい報酬はマグマの手前の床に置き、橋を渡らずに確実に取れるようにする
BRIDGE_TASKS = {
    **{
        f"11-{task:02d}-{variant}": _level11_task(task, variant)
        for task in range(1, 2 * len(BRIDGE_PAIR_WIDTHS) + 1)
        for variant in BRIDGE_VARIANT_HIGH_REWARD
    },
    "12-01-01": BridgeTask(
        high_on_right=True,
        width_high=1.5,
        width_low=0.0,
        reward_high=5.0,
        reward_low=1.0,
        angle_high_deg=15.0,
        angle_low_deg=40.0,
        lava_start_z=12.0,
        lava_end_z=24.0,
        goal_z_high=28.0,
        goal_z_low=10.0,
        mark_high_edges=False,
    ),
}


def bridge_center_x(task: BridgeTask, on_right: bool, z: float) -> float:
    """右（左）の橋の中心線の、奥行き z での x。スタートから報酬へ向かう直線。"""
    high = on_right == task.high_on_right
    angle = task.angle_high_deg if high else task.angle_low_deg
    offset = (z - BRIDGE_AGENT_Z) * math.tan(math.radians(angle))
    return BRIDGE_ARENA_SIZE / 2 + (offset if on_right else -offset)


def bridge_half_span(task: BridgeTask, on_right: bool) -> float:
    """右（左）の橋が x 方向に占める幅の半分。道幅は道に直交する向きの幅。"""
    high = on_right == task.high_on_right
    angle = task.angle_high_deg if high else task.angle_low_deg
    width = task.width_high if high else task.width_low
    return width / 2 / math.cos(math.radians(angle))


def _bridge_item(name: str, boxes: list[tuple[float, float, float, float]], height: float) -> str:
    """``boxes`` は (中心 x, 中心 z, x の幅, z の幅) の並び。"""
    lines = [f"    - !Item\n      name: {name}\n      positions:"]
    lines += [f"      - !Vector3 {{x: {x:g}, y: 0, z: {z:g}}}" for x, z, _, _ in boxes]
    lines.append(f"      rotations: [{', '.join('0' for _ in boxes)}]")
    lines.append("      sizes:")
    lines += [f"      - !Vector3 {{x: {sx:g}, y: {height:g}, z: {sz:g}}}" for _, _, sx, sz in boxes]
    return "\n".join(lines)


def bridge_arena_yaml(task: BridgeTask) -> str:
    """左右の斜めの橋のアリーナ。橋の先に、触れると終わる緑の球を報酬の大きさで置く。報酬が
    0 の側には何も置かない。"""
    # マグマを奥行きの帯ごとに、橋の外側（左の橋の左、2本の橋の間、右の橋の右）の箱で埋める。
    # 帯の中で橋は斜めに動くので、帯の両端での橋の範囲を合わせた外側だけを埋める。2本の橋が
    # 重なる帯には、橋の間のマグマはない
    lava = []
    slices = round((task.lava_end_z - task.lava_start_z) / BRIDGE_SLICE_Z)
    left_half = bridge_half_span(task, False)
    right_half = bridge_half_span(task, True)
    for index in range(slices):
        z0 = task.lava_start_z + index * BRIDGE_SLICE_Z
        z1 = z0 + BRIDGE_SLICE_Z
        left_lo = bridge_center_x(task, False, z1) - left_half
        left_hi = bridge_center_x(task, False, z0) + left_half
        right_lo = bridge_center_x(task, True, z0) - right_half
        right_hi = bridge_center_x(task, True, z1) + right_half
        edges = [(0.0, left_lo), (left_hi, right_lo), (right_hi, BRIDGE_ARENA_SIZE)]
        lava += [
            ((lo + hi) / 2, (z0 + z1) / 2, hi - lo, BRIDGE_SLICE_Z)
            for lo, hi in edges
            if hi - lo > 0.0
        ]
    divider = [
        (
            BRIDGE_ARENA_SIZE / 2,
            (task.lava_end_z + BRIDGE_ARENA_SIZE) / 2,
            0.5,
            BRIDGE_ARENA_SIZE - task.lava_end_z,
        )
    ]
    left_reward = task.reward_low if task.high_on_right else task.reward_high
    right_reward = task.reward_high if task.high_on_right else task.reward_low
    left_goal_z = task.goal_z_low if task.high_on_right else task.goal_z_high
    right_goal_z = task.goal_z_high if task.high_on_right else task.goal_z_low
    goals = [
        (bridge_center_x(task, on_right, z), z, size)
        for on_right, z, size in (
            (False, left_goal_z, left_reward),
            (True, right_goal_z, right_reward),
        )
        if size > 0
    ]
    # ハイリスク側の橋の両縁の外に、赤い球を奥行きの一定の間隔で並べる
    edge_spheres = []
    if task.mark_high_edges:
        high = task.high_on_right
        offset = bridge_half_span(task, high) + BRIDGE_EDGE_SPHERE_SIZE / 2
        count = round((task.lava_end_z - task.lava_start_z) / BRIDGE_EDGE_SPHERE_SPACING_Z) + 1
        for index in range(count):
            z = task.lava_start_z + index * BRIDGE_EDGE_SPHERE_SPACING_Z
            center = bridge_center_x(task, high, z)
            edge_spheres += [(center - offset, z), (center + offset, z)]
    edge_lines = []
    if edge_spheres:
        edge_lines = ["    - !Item\n      name: BadGoal\n      positions:"]
        edge_lines += [f"      - !Vector3 {{x: {x:g}, y: 0, z: {z:g}}}" for x, z in edge_spheres]
        edge_lines.append("      sizes:")
        size = BRIDGE_EDGE_SPHERE_SIZE
        edge_lines += [
            f"      - !Vector3 {{x: {size:g}, y: {size:g}, z: {size:g}}}" for _ in edge_spheres
        ]
    goal_lines = ["    - !Item\n      name: GoodGoal\n      positions:"]
    goal_lines += [f"      - !Vector3 {{x: {x:g}, y: 0, z: {z:g}}}" for x, z, _ in goals]
    goal_lines.append("      sizes:")
    goal_lines += [
        f"      - !Vector3 {{x: {size:g}, y: {size:g}, z: {size:g}}}" for _, _, size in goals
    ]
    return (
        "\n".join(
            [
                "!ArenaConfig",
                "arenas:",
                "  0: !Arena",
                "    pass_mark: 0",
                "    t: 250",
                "    items:",
                "    - !Item",
                "      name: Agent",
                "      positions:",
                f"      - !Vector3 {{x: {BRIDGE_ARENA_SIZE / 2:g}, y: 0, z: {BRIDGE_AGENT_Z:g}}}",
                "      rotations: [0]",
                _bridge_item("DeathZone", lava, 0.0),
                _bridge_item("Wall", divider, 2.0),
                *(["\n".join(edge_lines)] if edge_lines else []),
                "\n".join(goal_lines),
            ]
        )
        + "\n"
    )


# エージェントの半径。マグマの縁で死んだときも、その道に入ったとみなすための余裕
BRIDGE_AGENT_RADIUS = 0.5


def bridge_path(task: BridgeTask, x: float, z: float) -> int:
    """エピソードの終わりの位置 (x, z) が、どちらの道にあるか。ハイリスク側なら 1、
    ローリスク側なら -1、どちらでもない（スタートの床、橋の間や外側のマグマ）なら 0。
    橋の先の床では、中央の壁のどちら側かで決める。2本の橋が重なるところではハイリスク側に
    数える。"""
    if z > task.lava_end_z:
        return 1 if (x > BRIDGE_ARENA_SIZE / 2) == task.high_on_right else -1
    if z < task.lava_start_z - BRIDGE_AGENT_RADIUS:
        return 0
    high = task.high_on_right
    if abs(x - bridge_center_x(task, high, z)) <= (
        bridge_half_span(task, high) + BRIDGE_AGENT_RADIUS
    ):
        return 1
    if abs(x - bridge_center_x(task, not high, z)) <= (
        bridge_half_span(task, not high) + BRIDGE_AGENT_RADIUS
    ):
        return -1
    return 0


if __name__ == "__main__":
    COMPETITION_DIR.mkdir(parents=True, exist_ok=True)
    for name, task in BRIDGE_TASKS.items():
        (COMPETITION_DIR / f"{name}.yaml").write_text(bridge_arena_yaml(task))
    print(f"wrote {len(BRIDGE_TASKS)} arenas to {COMPETITION_DIR}")
