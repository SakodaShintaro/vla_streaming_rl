# Ralph Loop

あなたはこのリポジトリ（~/work/vla_streaming_rl）で、次のコマンドの学習の `max_stage_reached` を上げる手法を探す。

```bash
./train_animalai.sh food siglip <exp_name>
```

## 指標

`uv run python ralph/summary.py tsv <result_dir>` が `max_stage_reached`（学習を通してどれかのゲームが到達した最も先のステージ。減ることはない）の最後の値と、その値に初めて届いたエピソード `reached_at_episode` を出す。`max_stage_reached` が大きいほうを良いとし、同じなら `reached_at_episode` が小さいほうを良いとする。

各試行は<https://forge.coreweave.com/wandb/shintaro-sakoda/vla_streaming_rl_AnimalAI-v0?nw=nwusersakoda>に記録される。

## 状態（反復をまたいで残るもの）

- `log.md`: 実験ごとの 1 行の表（exp_name, commit, 変更点, 仮説, episodes, max_stage_reached, reached_at_episode, games_ended, 採否）と、その下の考察のメモ。毎回追記する
- `best.json`: 今の最良の {exp_name, commit, max_stage_reached, reached_at_episode, games_ended}。最初はベースライン（wandb から選んだときは run id と run の commit も持つ）

## 1 反復の手順

1. **初期化**（`ralph/log/best.json` が無いときだけ）: `ralph/log/` を作り、`git switch -c ralph/animalai-life` で作業ブランチを作る（既にあれば switch）。以後のコミットはすべてこのブランチ。続いて wandb を一通り見てベースラインを選ぶ:
   - 選んだ run を `uv run python ralph/summary.py wandb <run_id>` で集計し、`best.json` を書く。commit には今の HEAD を書く。run id と run の commit も best.json と `log.md` に残し、候補に入らなかった run とその理由も log.md に書く。log.md と best.json をコミットする
   - 候補が無ければ、何も変えずに exp_name `ralph00_baseline` で学習を始め（手順 4 の起動と同じやり方）、`current.json` を書いてこの反復を終える
2. **待つ**: `current.json` があれば、その pid の終了を `timeout 580 tail --pid=<pid> -f /dev/null` を繰り返して待つ（sleep は使わない）。途中で stdout の末尾や log_episode.tsv を見て、落ちていないか確認する
3. **評価する**: 結果のディレクトリ（`results/*_<exp_name>` の最新）を集計する。
   - `ralph00_baseline` が実行中だったなら、これを集計して best.json を書き、手順 4 へ進む
   - best より良ければ（`max_stage_reached` が大きい、または同じで `reached_at_episode` が小さい）採用: best.json をこの実験にする
   - そうでなければ不採用: `git revert --no-edit <current.json の commit>` で実験の変更を打ち消す
   - log.md に結果と、なぜ効いた／効かなかったと思うかを書き、log.md と best.json をコミットする。`current.json` を消す
4. **次の実験を 1 つ選んで始める**:
   - まず `git merge --no-edit main` で main を取り込む。衝突したら `git merge --abort` して log.md にその旨を書き、取り込まずに進む。取り込んだ変更が `ralph/` 以外（学習のコードや config）に及んだら、best との比較条件が変わったことを log.md に書く
   - log.md を読んで、仮説を 1 つだけ立てる（1 実験 1 変更。効果の切り分けのため）
   - 変更して `ralph<NN>_<短い名前>` のように exp_name を決め（NN は通し番号。スラッシュは結果のディレクトリ名に入るので使わない）、変更したファイルだけを `git add` してコミットする。コミットメッセージに仮説を書く。
   - 1〜2 分後に stdout を見て、起動に失敗していないこと（Unity の起動、最初のエピソード）を確かめる
   - `current.json` を書いて、この反復を終える

## 変えてよいもの・いけないもの

- 試す値はコマンドラインの上書きではなく yaml のデフォルトに書く（コミットが実行と一致するように）。
- `src/vla_streaming_rl/` の学習側のコード（agent、network、replay buffer など）は変えてよい
- 変えてはいけない: `train_animalai.sh`、`scripts/train.py` の記録と集計、`src/vla_streaming_rl/envs/`、`configs/env/`、アリーナ、pass mark、`ralph/summary.py`。指標を良く見せるだけの変更は禁止
- main ブランチには触らない。push しない

## その他

- できればGated DeltaNetを使っていきたい。しかし、今までGDNを入れた方が良い結果になったことはない。メモリを使うようにドロップアウトとかする必要があるかもしれない
- 学習率の調整はかなりどうでもいいので、よっぽど手法を変えたときに必要であるか、他にやることがないときにだけやる
- このPCでは2つあるいは3つ、もしかするともっと並列実行できるかもしれません
- コード差分はgit worktreeで管理してください
- 結果が良かったものは差分をできるだけ小さくしてからmainブランチへのプルリクエストとしてください

行動の表現がイマイチだと思われていたので、最近は以下の変更を試していた

```diff
diff --git a/src/vla_streaming_rl/networks/modules/self_attention.py b/src/vla_streaming_rl/networks/modules/self_attention.py
index 779fda7..3deae6b 100644
--- a/src/vla_streaming_rl/networks/modules/self_attention.py
+++ b/src/vla_streaming_rl/networks/modules/self_attention.py
@@ -49,6 +49,28 @@ def get_fourier_embeds_from_coordinates(embed_dim: int, coords: torch.Tensor) ->
     return emb
 
 
+def get_fourier_embeds_from_unit_coordinates(embed_dim: int, coords: torch.Tensor) -> torch.Tensor:
+    """−1〜1 の範囲の値（行動など）を sin/cos の特徴にする。
+
+    :func:`get_fourier_embeds_from_coordinates` は周波数が 1 から 1/10000 まで並ぶので、−1〜1 の
+    範囲ではほとんどの次元がほぼ一定になり、値の違いが特徴に出ない。ここでは周波数を π/2 から
+    32π まで等比に並べ、一番低い周波数で範囲全体が単調に、高い周波数で細かな違いが出るようにする。
+
+    Args:
+        embed_dim: 特徴の幅（偶数）
+        coords: (B, T, coord_dim)
+    Returns:
+        (B, T, coord_dim, embed_dim)
+    """
+    assert embed_dim % 2 == 0, embed_dim
+    half_dim = embed_dim // 2
+    freqs = torch.pi * torch.logspace(
+        -1, 5, half_dim, base=2.0, dtype=coords.dtype, device=coords.device
+    )
+    emb = coords.unsqueeze(-1) * freqs
+    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
+
+
 class SelfAttention(nn.Module):
     """
     Self-Attention module
diff --git a/src/vla_streaming_rl/networks/modules/value_head.py b/src/vla_streaming_rl/networks/modules/value_head.py
index 4db9a5d..7155cad 100644
--- a/src/vla_streaming_rl/networks/modules/value_head.py
+++ b/src/vla_streaming_rl/networks/modules/value_head.py
@@ -7,7 +7,7 @@ from torch import nn
 
 from .blocks import HyperLERPBlock, HypersphericalEmbedding, NormedLinear, Scaler, SimbaBlock
 from .head_output import HeadOutput
-from .self_attention import SpatialTransformerBlock, get_fourier_embeds_from_coordinates
+from .self_attention import SpatialTransformerBlock, get_fourier_embeds_from_unit_coordinates
 from .sparse_utils import apply_one_shot_pruning
 
 # 分布の critic のビンが覆う値の範囲の半幅。報酬の倍率 1 の 13-01 では、黄色 4 個（1.5 ずつ）と
@@ -289,7 +289,7 @@ class TokenActionValueHead(DistributionalValueHead):
     """状態のトークンに評価する行動のトークンを並べ、Transformer の層で混ぜてから MLP で読む critic。
 
     状態 (B, state_dim) は ``token_dim`` 幅のトークンを平らに並べたものとして読み戻す。行動の
-    チャンクは成分ごとに sin/cos の特徴で1トークンにする（前の行動のトークンと同じ作り方）。
+    チャンクは成分ごとに sin/cos の特徴で1トークンにする（−1〜1 の範囲に合わせた周波数）。
     混ぜる層は :class:`SpatialTransformerBlock` で、位置の埋め込みは 0、残差は layer scale で
     小さく始めるので、学習の始めは「状態のトークンと行動のトークンを並べて MLP で読む」のと
     ほぼ同じ。注意が、どの行動がどの観測のトークンと関わるかを学ぶ分だけ育つ。
@@ -358,7 +358,7 @@ class TokenActionValueHead(DistributionalValueHead):
         bs = a.size(0)
         a_flat = a.view(bs, -1)  # (B, action_dim)
         state_tokens = self.state_norm(x).view(bs, -1, self.token_dim)
-        action_tokens = get_fourier_embeds_from_coordinates(
+        action_tokens = get_fourier_embeds_from_unit_coordinates(
             self.token_dim, a.reshape(bs, 1, -1).to(x.dtype)
         )[:, 0]  # (B, action_dim, token_dim)
         tokens = torch.cat([state_tokens, action_tokens], dim=1) + self.token_position
```
