#!/bin/bash
set -ux

# Usage: ./compare_maze_memory.sh <agent> <seed> [hydra overrides...]
# 迷路の記憶の課題（レベル 15）で、記憶の持ち方を変えた手法を同じシードで順に実行する。
#   no_history     : ベースライン1。VLM の流れを毎ステップ空にし、履歴を使わない
#                    （vla.stream_window_steps=0）。デモの2周を覚えられないので、玉の側は当て推量になる
#   frozen_history : ベースライン2。凍結した VLM が流れを読み続け、GDN の層の状態をエピソードの
#                    間ずっと持ち越す。full attention の層の KV も、体力が尽きるまでの 400
#                    ステップ（scripts/make_maze_arenas.py の t）を覆う窓にして、エピソード全体を残す
# まずは少ないエピソード上限で回す。追加の引数はすべての手法に渡り、後ろにあるので上の指定も
# 上書きできる（例: episode_limit=1200）。
agent=${1}
seed=${2}
shift 2
cd $(dirname $0)

stamp=$(date +%Y%m%d_%H%M%S)
result_dir=results/compare_maze_memory_${stamp}_seed${seed}

methods=(
  "no_history vla.stream_window_steps=0"
  "frozen_history vla.stream_window_steps=400"
)

# 1手法が落ちても残りは走らせ、最後に失敗したものを報告する
failed=()
for method in "${methods[@]}"; do
  read -r name overrides <<< "${method}"
  ./train_animalai.sh 15 ${agent} ${name}_seed${seed} \
    seed=${seed} \
    episode_limit=400 \
    result_dir=${result_dir} \
    wandb_group=compare_maze_memory_${stamp} \
    ${overrides} \
    "$@" || failed+=(${name})
done

if [ ${#failed[@]} -ne 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
