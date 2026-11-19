# SPDX-License-Identifier: MIT
"""黄色を集める課題（レベル food）のアリーナの YAML を書き出す。

01-23-01（黄色 10 個と赤 10 個がランダムな位置にある）へ段階的に近づけるための 33 ステージ。

- food-01-01: 黄色 1 個が目の前にある（01-01-01 の緑を黄色にしたもの）
- food-02-01: 自分の位置と向きも黄色の位置もランダム
- food-03-01: 自分は 01-16-01 と同じ隅で壁向き、黄色の位置はランダム
- food-04-01 から food-33-01: 黄色 n 個と赤 n 個（n = 1 から 30）。自分の位置と向きも全部の玉の
  位置もランダム

位置を書かない物は Unity がエピソードごとにランダムに置く。玉の大きさは 01-23-01 と同じ 1。
pass_mark は黄色の数から 1 を引いた値で、01-2x と同じく合計 t ステップ以内に全部取れば合格する
（体力は黄色を取るたびに戻るが、報酬からは毎ステップ 1/t が引かれ続ける）。
``external/animal-ai/configs/custom`` に書く。表を変えたら、これを実行し直す。

    uv run python scripts/make_food_arenas.py
"""

from vla_streaming_rl.envs.animalai.curriculum import CUSTOM_ARENA_DIR

FOOD_GOAL_SIZE = 1
# 黄色と赤を n 個ずつ置くステージの n
FOOD_COUNTS = range(1, 31)
# t の最小値。01-2x では黄色 5 個が 250
FOOD_MIN_T = 250


def food_arena_t(num_yellow: int) -> int:
    """黄色 n 個のアリーナの t。01-20-01（5 個で 250）、01-21-01（10 個で 500）、01-22-01（40 個で
    1000）を通る折れ線で、10 個までは 1 個あたり 50、その先は 10 個と 40 個の間の直線。
    FOOD_MIN_T を下回らない。"""
    if num_yellow <= 10:
        return max(FOOD_MIN_T, 50 * num_yellow)
    return round(500 + (num_yellow - 10) * 50 / 3)


def _sizes(count: int) -> str:
    size = FOOD_GOAL_SIZE
    return "".join(f"      - !Vector3 {{x: {size}, y: {size}, z: {size}}}\n" for _ in range(count))


def food_arena_yaml(agent: str, num_yellow: int, num_red: int, yellow_position: str) -> str:
    """``agent`` はエージェントの項目（空ならランダム）、``yellow_position`` は黄色の positions の
    項目（空ならランダム）。"""
    red = (
        f"""    - !Item
      name: BadGoal
      sizes:
{_sizes(num_red)}"""
        if num_red > 0
        else ""
    )
    return f"""!ArenaConfig
arenas:
  0: !Arena
    pass_mark: {num_yellow - 1}
    t: {food_arena_t(num_yellow)}
    items:
{agent}    - !Item
      name: GoodGoalMulti
{yellow_position}      sizes:
{_sizes(num_yellow)}{red}"""


# 01-01-01 と同じ、中央で +z 向き
CENTER_AGENT = """    - !Item
      name: Agent
      positions:
      - !Vector3 {x: 20, y: 0, z: 20}
      rotations: [0]
"""
# 01-01-01 の緑と同じ、目の前
FRONT_POSITION = """      positions:
      - !Vector3 {x: 20, y: 0, z: 22}
"""
# 01-16-01 と同じ、隅で壁向き
CORNER_AGENT = """    - !Item
      name: Agent
      positions:
      - !Vector3 {x: 1, y: 0, z: 1}
      rotations: [225]
"""


def food_arenas() -> dict[str, str]:
    arenas = {
        "food-01-01": food_arena_yaml(CENTER_AGENT, 1, 0, FRONT_POSITION),
        "food-02-01": food_arena_yaml("", 1, 0, ""),
        "food-03-01": food_arena_yaml(CORNER_AGENT, 1, 0, ""),
    }
    for task, n in enumerate(FOOD_COUNTS, start=len(arenas) + 1):
        arenas[f"food-{task:02d}-01"] = food_arena_yaml("", n, n, "")
    return arenas


if __name__ == "__main__":
    CUSTOM_ARENA_DIR.mkdir(parents=True, exist_ok=True)
    arenas = food_arenas()
    for name, text in arenas.items():
        (CUSTOM_ARENA_DIR / f"{name}.yaml").write_text(text)
    print(f"wrote {len(arenas)} arenas to {CUSTOM_ARENA_DIR}")
