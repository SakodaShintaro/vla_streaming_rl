#!/bin/bash
set -ux

# Usage: ./compare_animalai.sh <seed> [hydra overrides...]
# 主比較の2手法を同じシードで順に実行する。
#   std_off_policy : 低レベル方策のみ
#   hierarchical_off_policy : 提案手法
# 比較の指標は Testbed なので終了時評価は常に入れる。
# 追加の引数は2手法すべてに渡り、後ろにあるので上の指定も上書きできる
# （例: env_factory.train_levels=[01,02]）。
seed=${1}
shift 1
cd $(dirname $0)

stamp=$(date +%Y%m%d_%H%M%S)
result_dir=results/compare_${stamp}_seed${seed}

# 1手法が落ちても残りは走らせ、最後に失敗したものを報告する
failed=()
for agent in std_off_policy hierarchical_off_policy; do
  ./train_animalai.sh ${agent} ${agent} \
    seed=${seed} \
    result_dir=${result_dir} \
    wandb_group=compare_${stamp} \
    "$@" || failed+=(${agent})
done

if [ ${#failed[@]} -ne 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
