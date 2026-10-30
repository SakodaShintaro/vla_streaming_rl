# Ralph Loop: Animal-AI easy / vla_off_policy の last_round_success_rate を上げる

あなたはこのリポジトリ（~/work/vla_streaming_rl）で、次のコマンドの学習の `last_round_success_rate`（30 アリーナの 1 周ごとの成功率）を上げるハイパーパラメータや手法を探す。

```bash
./train_animalai.sh easy vla_off_policy <exp_name>
```

## 指標

`uv run python ralph/round_success_summary.py tsv <result_dir>` が `last_round_success_rate` の `last5_mean` を出すのでその値で比べる。

各試行は<https://forge.coreweave.com/wandb/shintaro-sakoda/vla_streaming_rl_AnimalAI-v0?nw=nwusersakoda>に記録される。

## 状態（反復をまたいで残るもの）

最初に必ず読む。

Git 管理するもの（`ralph/animalai_easy/`。更新したら `ralph: log <exp_name>` のようなメッセージでコミットする）:

- `log.md`: 実験ごとの 1 行の表（exp_name, commit, 変更点, 仮説, rounds, max, mean, last5_mean, 採否）と、その下の考察のメモ。毎回追記する
- `best.json`: 今の最良の {exp_name, commit, max, mean, last5_mean}。最初はベースライン（wandb から選んだときは run id と run の commit も持つ）

Git 管理しないもの（`results/ralph_animalai_easy/`）:

- `current.json`: 走っている実験の {exp_name, commit, pid, stdout} 。走っていなければ無い
- `stdout/<exp_name>.log`: 学習の標準出力

## 1 反復の手順

1. **初期化**（`ralph/animalai_easy/best.json` が無いときだけ）: `ralph/animalai_easy/` と `results/ralph_animalai_easy/stdout/` を作り、`git switch -c ralph/animalai-easy` で作業ブランチを作る（既にあれば switch）。以後のコミットはすべてこのブランチ。続いて wandb を一通り見てベースラインを選ぶ:
   - wandb の API で上のプロジェクトの run を一覧し、次をすべて満たすものを候補にする: group が `easy`、名前が `vla_off_policy_` で始まる、state が finished で 1200 エピソードを走り切っている、その run の commit から今の HEAD までの `git diff --stat` が `ralph/` と `.gitignore` 以外に触れていない（学習のコードと config がブランチを切った時点と同じ）
   - 候補が複数あれば、最も新しいものを選ぶ
   - 選んだ run を `uv run python ralph/round_success_summary.py wandb <run_id>` で集計し、`best.json` を書く。commit には今の HEAD を書く。run id と run の commit も best.json と `log.md` に残し、候補に入らなかった run とその理由も log.md に書く。log.md と best.json をコミットする
   - 候補が無ければ、何も変えずに exp_name `ralph00_baseline` で学習を始め（手順 4 の起動と同じやり方）、`current.json` を書いてこの反復を終える
2. **待つ**: `current.json` があれば、その pid の終了を `timeout 580 tail --pid=<pid> -f /dev/null` を繰り返して待つ（sleep は使わない）。途中で stdout の末尾や log_episode.tsv を見て、落ちていないか確認する
3. **評価する**: 結果のディレクトリ（`results/*_vla_off_policy_<exp_name>` の最新）を集計する。
   - 1200 エピソードに届かずに落ちた（OOM など）なら「失敗」と記録し、原因を log.md に書く
   - `ralph00_baseline` が実行中だったなら、これを集計して best.json を書き、手順 4 へ進む
   - last5_mean が best より高ければ採用: best.json をこの実験にする
   - そうでなければ不採用: `git revert --no-edit <current.json の commit>` で実験の変更を打ち消す（reset はログのコミットまで消すので使わない。実験のコミットは `ralph/<exp_name>` のタグでも残る）
   - log.md に結果と、なぜ効いた／効かなかったと思うかを書き、log.md と best.json をコミットする。`current.json` を消す
4. **次の実験を 1 つ選んで始める**:
   - まず `git merge --no-edit main` で main を取り込む。衝突したら `git merge --abort` して log.md にその旨を書き、取り込まずに進む。取り込んだ変更が `ralph/` 以外（学習のコードや config）に及んだら、best との比較条件が変わったことを log.md に書く
   - log.md を読んで、仮説を 1 つだけ立てる（1 実験 1 変更。効果の切り分けのため）
   - 変更して `ralph<NN>_<短い名前>` のように exp_name を決め（NN は通し番号。スラッシュは結果のディレクトリ名に入るので使わない）、変更したファイルだけを `git add` してコミットし、タグ `ralph/<exp_name>` を打つ。コミットメッセージに仮説を書く。
   - 学習を切り離して始める: `setsid nohup ./train_animalai.sh easy vla_off_policy <exp_name> > results/ralph_animalai_easy/stdout/<exp_name>.log 2>&1 & echo $!`
   - 1〜2 分後に stdout を見て、起動に失敗していないこと（Unity の起動、最初のエピソード）を確かめる
   - `current.json` を書いて、この反復を終える

## 変えてよいもの・いけないもの

- 試す値はコマンドラインの上書きではなく yaml のデフォルトに書く（コミットが実行と一致するように）。`configs/agent/vla_off_policy.yaml` に書くのを第一にする（`configs/default.yaml` や `configs/agent/base/network.yaml` を変えると他の agent や env にも効いてしまう）
- `src/vla_streaming_rl/` の学習側のコード（agent、network、replay buffer など）は変えてよい。新しい仕組みを足すときは config のキーで切り替え、default.yaml の値は今の挙動のままにして vla_off_policy.yaml で有効にする。頼まれていないリファクタや整理はしない
- 変えてはいけない: `train_animalai.sh`、`scripts/train.py` の記録と集計、`src/vla_streaming_rl/envs/`、`configs/env/`、`episode_limit`、アリーナ、pass mark、`ralph/round_success_summary.py`。指標を良く見せるだけの変更は禁止
- main ブランチには触らない。push しない
