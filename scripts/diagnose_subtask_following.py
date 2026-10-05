# SPDX-License-Identifier: MIT
"""学習済みの階層的方策で、低レベル方策のヘッドがサブタスクに追従しているかを確かめる。

``test_trained_agent.py`` と同じように評価のアリーナを1回ずつ回し、ネットワークが行動を
出すたびに、同じノイズでサブタスクの活性を0にした入力の行動も出して比べる。活性を0にするのは、
学習時に ``subtask_keep`` でサブタスクを落とすのと同じ入力になる。実際に実行するのは
ふだんどおりの行動で、比べる行動は記録するだけ。

ステップごとに次を記録する（``subtask_diagnosis.tsv``）。
  - gap: サブタスクありとなしの行動の差（チャンクの最初の行動の、成分ごとの差の絶対値の平均）
  - pull: 返答の行動を保持しているステップで、サブタスクがあることで、ヘッドの行動が返答の
    行動にどれだけ近づいたか（なしの距離 − ありの距離。正なら近づいている）
gap がほぼ 0 なら、ヘッドはサブタスクを読んでいない。gap があっても pull が 0 付近なら、
読んではいるが返答の行動の方向には動いていない。
"""

from vla_streaming_rl.script_setup import setup_runtime

setup_runtime()

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from test_trained_agent import load_eval_config, load_global_step, run_testbed

from vla_streaming_rl.agents.build import build_all
from vla_streaming_rl.checkpoint import load_checkpoint_weights
from vla_streaming_rl.envs.animalai_curriculum import training_levels
from vla_streaming_rl.script_setup import disable_render_if_headless, resolve_seed, seed_everything
from vla_streaming_rl.wrappers import make_env


def main(cfg, checkpoint_path: Path, result_dir: Path, seed: int, render: bool) -> None:
    seed_everything(seed)
    global_step = load_global_step(checkpoint_path.parent)
    env = make_env(cfg.env_id, cfg.env_factory, result_dir=None, seed=seed)
    env.action_space.seed(seed)
    env.unwrapped.set_global_step(global_step)
    network, agent = build_all(env, cfg)
    load_checkpoint_weights(checkpoint_path, network)
    network.eval()
    assert agent.high_level_policy is not None, "the checkpoint carries no high-level policy"

    rows = []
    infer = network.infer

    def infer_and_compare(data):
        # ふだんの行動と、同じノイズでサブタスクを0にした行動。乱数の進みはふだんどおりに戻す
        device = data.s_seq.device
        noise_state = torch.cuda.get_rng_state(device)
        result = infer(data)
        after_state = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(noise_state, device)
        without = infer(
            dataclasses.replace(
                data, subtask_activations_seq=torch.zeros_like(data.subtask_activations_seq)
            )
        )
        torch.cuda.set_rng_state(after_state, device)

        action = result.action[0, 0].cpu().numpy()
        action_without = without.action[0, 0].cpu().numpy()
        # 選択の前に、エージェントはこのステップの返答の行動と保持の有無を決めている
        vlm_action = agent._to_net_action(agent.prev_vlm_action)
        holding = agent.prev_vlm_holding
        pull = (
            float(np.abs(action_without - vlm_action).mean() - np.abs(action - vlm_action).mean())
            if holding
            else float("nan")
        )
        rows.append(
            dict(
                arena=env.unwrapped.arena_name,
                gap=float(np.abs(action - action_without).mean()),
                pull=pull,
                holding=int(holding),
                action=action,
                action_without=action_without,
                vlm_action=vlm_action,
            )
        )
        return result

    network.infer = infer_and_compare
    run_testbed(
        agent,
        env,
        seed,
        render,
        cfg.render_scale,
        cfg.env_id,
        global_step,
        result_dir,
        cfg.env_factory.train_variant,
        training_levels(
            cfg.env_factory.mode, cfg.env_factory.train_variant, list(cfg.env_factory.train_levels)
        ),
    )
    env.close()

    with open(result_dir / "subtask_diagnosis.tsv", "w") as f:
        f.write("arena\tgap\tpull\tholding\taction\taction_without\tvlm_action\n")
        for row in rows:
            vectors = [
                ",".join(f"{v:.4f}" for v in row[key])
                for key in ("action", "action_without", "vlm_action")
            ]
            f.write(
                f"{row['arena']}\t{row['gap']:.6f}\t{row['pull']:.6f}\t{row['holding']}\t"
                + "\t".join(vectors)
                + "\n"
            )
    gaps = np.array([row["gap"] for row in rows])
    pulls = np.array([row["pull"] for row in rows if row["holding"]])
    summary = [
        f"steps: {len(rows)}",
        f"gap mean: {gaps.mean():.4f} (median {np.median(gaps):.4f})",
        f"holding steps: {len(pulls)}",
        f"pull mean: {pulls.mean():.4f} (closer with subtask: {np.mean(pulls > 0):.3f})",
    ]
    (result_dir / "subtask_diagnosis_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="checkpoint.pt saved by scripts/train.py")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    checkpoint_path = cli_args.checkpoint.resolve()
    cfg = load_eval_config(checkpoint_path.parent)
    print(OmegaConf.to_yaml(cfg))
    main(
        cfg,
        checkpoint_path,
        checkpoint_path.parent / "eval" / "subtask_diagnosis",
        resolve_seed(cli_args.seed),
        disable_render_if_headless(not cli_args.no_render),
    )
