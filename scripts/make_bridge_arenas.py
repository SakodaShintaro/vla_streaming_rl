# SPDX-License-Identifier: MIT
"""レベル 11（左右の橋の課題）のアリーナの YAML を書き出す。

課題の中身（組ごとの道幅、バリアントごとの報酬）はここの表にあり、それを競技のアリーナと
同じ ``11-YY-ZZ.yaml`` の名前で、競技のアリーナと同じ ``external/animal-ai/configs/competition``
に書く。表を変えたら、これを実行し直してアリーナを作り直す。どちらの道を選んだかを終わりの
位置から判定する ``bridge_path`` も、あとの分析のためにここに置く。

    uv run python scripts/make_bridge_arenas.py
"""

from dataclasses import dataclass

from vla_streaming_rl.envs.animalai_curriculum import COMPETITION_DIR

# 橋のアリーナの寸法（アリーナは 40 x 40、x が右、z が前）。z = 0〜10 は全幅が安全な床で、
# エージェントは中央から前を向いて始まる。z = 10〜32 は左右の橋以外がマグマ、z = 32〜40 は
# 橋の先の安全な床で、中央の壁で左右を分ける
BRIDGE_ARENA_SIZE = 40.0
BRIDGE_LEFT_X = 10.0
BRIDGE_RIGHT_X = 30.0
BRIDGE_START_Z = 10.0
BRIDGE_END_Z = 32.0
BRIDGE_GOAL_Z = 36.0
BRIDGE_AGENT_Z = 4.0
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
    """レベル 11 のアリーナ1つの、ハイリスク側の位置と左右の道の幅と報酬。"""

    high_on_right: bool
    width_high: float
    width_low: float
    reward_high: float
    reward_low: float


def bridge_task(name: str) -> BridgeTask:
    """レベル 11 のアリーナ名（"11-YY-ZZ"）から、その課題の中身を引く。"""
    level, task, variant = name.split("-")
    assert level == "11", f"{name} is not a bridge arena"
    pair = (int(task) + 1) // 2
    width_high, width_low = BRIDGE_PAIR_WIDTHS[pair]
    return BridgeTask(
        high_on_right=int(task) % 2 == 0,
        width_high=width_high,
        width_low=width_low,
        reward_high=BRIDGE_VARIANT_HIGH_REWARD[variant],
        reward_low=BRIDGE_PAIR_LOW_REWARD[pair],
    )


def _bridge_item(name: str, boxes: list[tuple[float, float, float, float]], height: float) -> str:
    """``boxes`` は (中心 x, 中心 z, x の幅, z の幅) の並び。"""
    lines = [f"    - !Item\n      name: {name}\n      positions:"]
    lines += [f"      - !Vector3 {{x: {x:g}, y: 0, z: {z:g}}}" for x, z, _, _ in boxes]
    lines.append(f"      rotations: [{', '.join('0' for _ in boxes)}]")
    lines.append("      sizes:")
    lines += [f"      - !Vector3 {{x: {sx:g}, y: {height:g}, z: {sz:g}}}" for _, _, sx, sz in boxes]
    return "\n".join(lines)


def bridge_arena_yaml(task: BridgeTask) -> str:
    """左右の橋のアリーナ。橋の先に、触れると終わる緑の球を報酬の大きさで置く。報酬が 0 の
    側には何も置かない。"""
    high = (task.reward_high, task.width_high)
    low = (task.reward_low, task.width_low)
    (left_reward, left_width), (right_reward, right_width) = (
        (low, high) if task.high_on_right else (high, low)
    )
    lava_z = (BRIDGE_START_Z + BRIDGE_END_Z) / 2
    lava_length = BRIDGE_END_Z - BRIDGE_START_Z
    edges = [
        (0.0, BRIDGE_LEFT_X - left_width / 2),
        (BRIDGE_LEFT_X + left_width / 2, BRIDGE_RIGHT_X - right_width / 2),
        (BRIDGE_RIGHT_X + right_width / 2, BRIDGE_ARENA_SIZE),
    ]
    lava = [((lo + hi) / 2, lava_z, hi - lo, lava_length) for lo, hi in edges]
    divider = [
        (
            BRIDGE_ARENA_SIZE / 2,
            (BRIDGE_END_Z + BRIDGE_ARENA_SIZE) / 2,
            0.5,
            BRIDGE_ARENA_SIZE - BRIDGE_END_Z,
        )
    ]
    goals = [
        (x, size)
        for x, size in ((BRIDGE_LEFT_X, left_reward), (BRIDGE_RIGHT_X, right_reward))
        if size > 0
    ]
    goal_lines = ["    - !Item\n      name: GoodGoal\n      positions:"]
    goal_lines += [f"      - !Vector3 {{x: {x:g}, y: 0, z: {BRIDGE_GOAL_Z:g}}}" for x, _ in goals]
    goal_lines.append("      sizes:")
    goal_lines += [
        f"      - !Vector3 {{x: {size:g}, y: {size:g}, z: {size:g}}}" for _, size in goals
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
                "\n".join(goal_lines),
            ]
        )
        + "\n"
    )


# エージェントの半径。マグマの縁で死んだときも、その道に入ったとみなすための余裕
BRIDGE_AGENT_RADIUS = 0.5


def bridge_path(
    high_on_right: bool, width_high: float, width_low: float, x: float, z: float
) -> int:
    """エピソードの終わりの位置 (x, z) が、どちらの道にあるか。ハイリスク側なら 1、
    ローリスク側なら -1、どちらでもない（スタートの床、橋の間や外側のマグマ）なら 0。
    橋の先の床では、中央の壁のどちら側かで決める。"""
    high_x, low_x = (
        (BRIDGE_RIGHT_X, BRIDGE_LEFT_X) if high_on_right else (BRIDGE_LEFT_X, BRIDGE_RIGHT_X)
    )
    if z > BRIDGE_END_Z:
        return 1 if (x > BRIDGE_ARENA_SIZE / 2) == high_on_right else -1
    if z < BRIDGE_START_Z - BRIDGE_AGENT_RADIUS:
        return 0
    if abs(x - high_x) <= width_high / 2 + BRIDGE_AGENT_RADIUS:
        return 1
    if abs(x - low_x) <= width_low / 2 + BRIDGE_AGENT_RADIUS:
        return -1
    return 0


if __name__ == "__main__":
    COMPETITION_DIR.mkdir(parents=True, exist_ok=True)
    names = [
        f"11-{task:02d}-{variant}"
        for task in range(1, 2 * len(BRIDGE_PAIR_WIDTHS) + 1)
        for variant in BRIDGE_VARIANT_HIGH_REWARD
    ]
    for name in names:
        (COMPETITION_DIR / f"{name}.yaml").write_text(bridge_arena_yaml(bridge_task(name)))
    print(f"wrote {len(names)} arenas to {COMPETITION_DIR}")
