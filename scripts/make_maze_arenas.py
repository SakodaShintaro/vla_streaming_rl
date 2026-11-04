# SPDX-License-Identifier: MIT
"""迷路の走行を手がかりにする記憶の課題（レベル 15）のアリーナの YAML を書き出す。

「日」の字を横に倒した形の迷路で、中央レーンの上端から左右に分かれ、外側の廊下をぐるっと回って
中央レーンの下端に戻ってこられる。左右の外側の廊下の中ほどに、ブロックへ少しくぼませた所を
両側とも作り、片側にだけ緑の玉を置く。くぼみの奥にあるので、廊下の真ん中を通っても取らない。

エピソードの始めは、ルールで左右を1周ずつ走らせる（``demoWaypoints``。env.py の
``demo_action``）。2周して中央レーンの最初の位置に戻ってから操作できるようになる。観測の履歴に
2周が入っているので、玉を見た側を覚えていれば、いきなり正しい側へ曲がれる。

課題は、先に回る側と玉の側の 4 通り（15-01 から 15-04）。版はコースの大きさで、
``MAZE_SIZES`` に足せば増える。競技のアリーナと同じ ``external/animal-ai/configs/competition``
に書く。表を変えたら、これを実行し直す。

    uv run python scripts/make_maze_arenas.py
"""

from vla_streaming_rl.envs.animalai.curriculum import COMPETITION_DIR

# 迷路の中心（アリーナは 40 x 40）
MAZE_CENTER = 20.0
# 壁の厚さと高さ。高さはエージェントの目線より高くして、壁越しに隣の廊下が見えないようにする
MAZE_WALL_THICKNESS = 1.0
MAZE_WALL_HEIGHT = 3.0
MAZE_WALL_COLOR = 153
# くぼみの深さ（x）と長さ（z）。玉（直径 1）が廊下へはみ出さない深さにする
MAZE_NOTCH_DEPTH = 1.5
MAZE_NOTCH_LENGTH = 2.0
MAZE_GOAL_SIZE = 1.0

# 版ごとの (廊下の幅, ブロックの幅, ブロックの長さ, 体力が尽きるまでのステップ数)。
# 最小の版はデモに約 130 ステップかかり、デモの走り方なら正しい側へ曲がってから約 45 ステップで
# 玉に届く。学習の始めの方策でも迷路を探って玉に出会えるように、デモの後に約 270 ステップ残す
# （間違えて1周してからでも間に合う。早く取るほど体力の減りが少ないことだけが差になる）
MAZE_SIZES = {"01": (3.0, 3.0, 10.0, 400)}

# 課題ごとの (先に回る側, 玉の側)。-1 が左（-x）、+1 が右（+x）
MAZE_TASKS = {"15-01": (-1, -1), "15-02": (-1, 1), "15-03": (1, -1), "15-04": (1, 1)}


def _n(value: float) -> str:
    return f"{value:g}"


def _vector(x: float, y: float, z: float) -> str:
    return f"!Vector3 {{x: {_n(x)}, y: {_n(y)}, z: {_n(z)}}}"


def maze_arena_yaml(
    corridor: float, block_width: float, block_length: float, t: int, first: int, goal: int
) -> str:
    c = MAZE_CENTER
    # 中心から外側の廊下の外縁までの x の距離と、上下の廊下の外縁までの z の距離
    half_x = corridor / 2 + block_width + corridor
    half_z = block_length / 2 + corridor
    # 外側の廊下の真ん中の x のずれと、上下の廊下の真ん中の z
    side_x = corridor / 2 + block_width + corridor / 2
    top_z = c + block_length / 2 + corridor / 2
    bottom_z = c - block_length / 2 - corridor / 2
    start_z = c - block_length / 2 + 1.0

    # (中心の x, 中心の z, x の大きさ, z の大きさ)。左右の外壁が角を覆い、上下の外壁はその間に
    # 収める（重なると Unity は置かないか別の場所に置く）
    walls = [
        (c - half_x - MAZE_WALL_THICKNESS / 2, c, MAZE_WALL_THICKNESS, 2 * half_z + 2),
        (c + half_x + MAZE_WALL_THICKNESS / 2, c, MAZE_WALL_THICKNESS, 2 * half_z + 2),
        (c, c - half_z - MAZE_WALL_THICKNESS / 2, 2 * half_x, MAZE_WALL_THICKNESS),
        (c, c + half_z + MAZE_WALL_THICKNESS / 2, 2 * half_x, MAZE_WALL_THICKNESS),
    ]
    # ブロックは、中央レーン側の細長い部分と、くぼみの上下の外側の部分に分けて置く
    strip_length = (block_length - MAZE_NOTCH_LENGTH) / 2
    for side in (-1, 1):
        inner_width = block_width - MAZE_NOTCH_DEPTH
        walls.append((c + side * (corridor / 2 + inner_width / 2), c, inner_width, block_length))
        strip_x = c + side * (corridor / 2 + block_width - MAZE_NOTCH_DEPTH / 2)
        for sign in (-1, 1):
            strip_z = c + sign * (MAZE_NOTCH_LENGTH / 2 + strip_length / 2)
            walls.append((strip_x, strip_z, MAZE_NOTCH_DEPTH, strip_length))
    goal_x = c + goal * (corridor / 2 + block_width - MAZE_NOTCH_DEPTH / 2)

    def lap(side: int) -> list[tuple[float, float]]:
        return [
            (c, top_z),
            (c + side * side_x, top_z),
            (c + side * side_x, bottom_z),
            (c, bottom_z),
        ]

    waypoints = lap(first) + lap(-first) + [(c, start_z)]
    waypoint_text = ", ".join(f"[{_n(x)}, {_n(z)}]" for x, z in waypoints)
    color = f"!RGB {{r: {MAZE_WALL_COLOR}, g: {MAZE_WALL_COLOR}, b: {MAZE_WALL_COLOR}}}"
    positions = "".join(f"      - {_vector(x, 0, z)}\n" for x, z, _, _ in walls)
    sizes = "".join(f"      - {_vector(sx, MAZE_WALL_HEIGHT, sz)}\n" for _, _, sx, sz in walls)
    colors = "".join(f"      - {color}\n" for _ in walls)
    rotations = ", ".join("0" for _ in walls)
    return f"""!ArenaConfig
arenas:
  0: !Arena
    pass_mark: 0
    t: {t}
    demoWaypoints: [{waypoint_text}]
    items:
    - !Item
      name: Agent
      positions:
      - {_vector(c, 0, start_z)}
      rotations: [0]
    - !Item
      name: Wall
      positions:
{positions}      rotations: [{rotations}]
      sizes:
{sizes}      colors:
{colors}    - !Item
      name: GoodGoal
      positions:
      - {_vector(goal_x, 0, c)}
      sizes:
      - {_vector(MAZE_GOAL_SIZE, MAZE_GOAL_SIZE, MAZE_GOAL_SIZE)}
"""


if __name__ == "__main__":
    COMPETITION_DIR.mkdir(parents=True, exist_ok=True)
    for variant, (corridor, block_width, block_length, t) in MAZE_SIZES.items():
        for task, (first, goal) in MAZE_TASKS.items():
            (COMPETITION_DIR / f"{task}-{variant}.yaml").write_text(
                maze_arena_yaml(corridor, block_width, block_length, t, first, goal)
            )
    print(f"wrote {len(MAZE_SIZES) * len(MAZE_TASKS)} arenas to {COMPETITION_DIR}")
