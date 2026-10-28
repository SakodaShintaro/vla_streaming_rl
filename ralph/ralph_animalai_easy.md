# Ralph Loop: Animal-AI easy / std_off_policy の last_round_success_rate を上げる

あなたはこのリポジトリ（/home/shintarosakoda/work/vla_streaming_rl）で、次のコマンドの学習の
`last_round_success_rate`（30 アリーナの 1 周ごとの成功率）を上げるハイパーパラメータや手法を探す。

```bash
./train_animalai.sh easy std_off_policy <exp_name>
```

1 回の学習は 1200 エピソード（40 周）で、4 時間ほどかかる。この反復で行うのは
「動いている学習を待って評価する → 次の実験を 1 つ始める」の 1 サイクルだけ。

## 指標

`uv run python ralph/round_success_summary.py tsv <result_dir>` が
`last_round_success_rate` の `last5_mean` を出すのでその値で比べる。

各試行は<https://forge.coreweave.com/wandb/shintaro-sakoda/vla_streaming_rl_AnimalAI-v0?nw=nwusersakoda>に記録される。

## 状態（反復をまたいで残るもの）

`results/ralph_animalai_easy/`（git 管理外）に置く。最初に必ず読む。

- `log.md`: 実験ごとの 1 行の表（exp_name, commit, 変更点, 仮説, rounds, max, mean, score, 採否）と、
  その下の考察のメモ。毎回追記する
- `best.json`: 今の最良の {exp_name, commit, max, mean, score}。最初はベースライン
- `current.json`: 走っている実験の {exp_name, commit, pid, stdout} 。走っていなければ無い
- `stdout/<exp_name>.log`: 学習の標準出力

## 1 反復の手順

1. **初期化**（`results/ralph_animalai_easy/` が無いときだけ）: ディレクトリを作り、ベースラインを
   取って `best.json`（commit f81265c）と `log.md` を書く。`git switch -c ralph/animalai-easy` で
   作業ブランチを作る（既にあれば switch）。以後のコミットはすべてこのブランチ
2. **待つ**: `current.json` があれば、その pid の終了を
   `timeout 580 tail --pid=<pid> -f /dev/null` を繰り返して待つ（sleep は使わない）。
   途中で stdout の末尾や log_episode.tsv を見て、落ちていないか確認する
3. **評価する**: 結果のディレクトリ（`results/*_std_off_policy_<exp_name>` の最新）を集計する。
   - 1200 エピソードに届かずに落ちた（OOM など）なら「失敗」と記録し、原因を log.md に書く
   - ベースラインが実行中だったなら、ここで取り直して best.json を更新してから比べる
   - score が best より高ければ採用: best.json をこの実験にする
   - そうでなければ不採用: `git reset --hard <best の commit>` で戻す
     （実験のコミットは `ralph/<exp_name>` のタグで残る）
   - log.md に結果と、なぜ効いた／効かなかったと思うかを書く。`current.json` を消す
4. **終わりの判定**: 完了した実験（失敗を含む、ベースラインは数えない）が 8 個になったら、
   log.md の末尾に最良の設定と全体のまとめを書き、
   `<promise>OPTIMIZATION DONE</promise>` とだけ出力して終える
5. **次の実験を 1 つ選んで始める**:
   - log.md を読んで、仮説を 1 つだけ立てる（1 実験 1 変更。効果の切り分けのため）
   - 変更して `ralph<NN>_<短い名前>` のように exp_name を決め（NN は通し番号。スラッシュは
     結果のディレクトリ名に入るので使わない）、変更したファイルだけを `git add` してコミットし、
     タグ `ralph/<exp_name>` を打つ。コミットメッセージに仮説を書く。
   - 学習を切り離して始める:
     `setsid nohup ./train_animalai.sh easy std_off_policy <exp_name> > results/ralph_animalai_easy/stdout/<exp_name>.log 2>&1 & echo $!`
   - 1〜2 分後に stdout を見て、起動に失敗していないこと（Unity の起動、最初のエピソード）を確かめる
   - `current.json` を書いて、この反復を終える

## 変えてよいもの・いけないもの

- 試す値はコマンドラインの上書きではなく yaml のデフォルトに書く（コミットが実行と一致するように）。
  `configs/agent/std_off_policy.yaml` に書くのを第一にする（`configs/default.yaml` や
  `configs/agent/base/network.yaml` を変えると他の agent や env にも効いてしまう）
- `src/vla_streaming_rl/` の学習側のコード（agent、network、replay buffer など）は変えてよい。
  新しい仕組みを足すときは config のキーで切り替え、default.yaml の値は今の挙動のままにして
  std_off_policy.yaml で有効にする。頼まれていないリファクタや整理はしない
- 変えてはいけない: `train_animalai.sh`、`scripts/train.py` の記録と集計、
  `src/vla_streaming_rl/envs/`、`configs/env/`、`episode_limit`、アリーナ、pass mark、
  `local/round_success_summary.py`。指標を良く見せるだけの変更は禁止
- Python の書き方はリポジトリの CLAUDE.md に従う（デフォルト引数・getattr・dict.get を避け、
  検査は assert）
- 学習は同時に 1 つだけ（GPU は RTX 5080 16GB 1 枚）。メモリが足りない変更は避ける
- 同じ設定を seed を変えて繰り返すことはしない。差がノイズに見えても、1 設定 1 回で判断する
- main ブランチには触らない。push しない

## 実験の候補（参考。log.md の結果を見て選び直してよい）

- 学習率（actor_lr / critic_lr、今は 1e-5）、batch_size（16）、learning_starts（2000）
- gamma、horizon、use_eligibility_trace / et_lambda
- replay_buffer.size、priority_alpha / priority_beta
- reward_scale_mode / fixed_reward_scale、normalizing_by_return
- actor_critic.seq_len、register_num、temporal_block_num、token_dropout
- 既知の問題: Q がゴールの終端の報酬を超えて過大に推定され、エピソードを終わらせるのを
  避ける（ゴールの前でためらう）ことがある。target network や clipped double Q、報酬の倍率の
  固定などが候補
