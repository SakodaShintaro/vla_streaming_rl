# SPDX-License-Identifier: MIT
"""動く玉の課題（レベル moving）のアリーナの YAML を書き出す。

黄色 1 個が目の前から横へ動いていき、始めの数ステップは自分が動けない。自分は後ろの壁に接して
いるので、下がって玉を視野に戻すこともできない。止めるステップ数が 4 になると、動けるようになった
時点で玉はもう視野から出ている。これが主となる moving-09-01 と moving-10-01 で、そこへ段階的に
近づける 10 ステージ。

出た側を覚えていなくても、片側へ 1〜2 ステップ回って玉が見えなければ切り返す動きで制限時間に間に合う
（make_food_arenas.py の ``MOVING_T``）。記憶を使っているかは合否では分からないので、動けるように
なった最初の旋回が玉の出た側を向いているかで見る。

- moving-01-01: 黄色 1 個が目の前にある（food-01-01 と同じ）
- moving-02-01: 自分の位置と向きも黄色の位置もランダム（food-02-01 と同じ）
- moving-03-01 から moving-10-01: 止めるステップ数 1, 2, 3, 4 のそれぞれで、玉が左へ動くもの、
  右へ動くものの順

止めるステップ数が 3 までは、動けるようになった時点でまだ玉が視野の端に映っている。
停止（``frozenAgentDelays``）は Python の環境の側でステップ数として行う（env.py）。
``external/animal-ai/configs/custom`` に書く。表を変えたら、これを実行し直す。

    uv run python scripts/make_moving_arenas.py
"""

from make_food_arenas import (
    CENTER_AGENT,
    FRONT_POSITION,
    food_arena_yaml,
    moving_arena_yaml,
    write_arenas,
)

# 始めにエージェントを止めておくステップ数の段階。止めたままだと玉はフレーム 4 で視野から消える
FROZEN_STEP_LADDER = (1, 2, 3, 4)
# 玉の動く向き（左、右）
MOVING_ROTATIONS = (270, 90)


def moving_arenas() -> dict[str, str]:
    arenas = {
        "moving-01-01": food_arena_yaml(CENTER_AGENT, 1, 0, FRONT_POSITION),
        "moving-02-01": food_arena_yaml("", 1, 0, ""),
    }
    for frozen_steps in FROZEN_STEP_LADDER:
        for rotation in MOVING_ROTATIONS:
            arenas[f"moving-{len(arenas) + 1:02d}-01"] = moving_arena_yaml(rotation, frozen_steps)
    return arenas


if __name__ == "__main__":
    write_arenas(moving_arenas())
