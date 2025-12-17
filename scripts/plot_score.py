#!/usr/bin/env python
"""Plot eval_score vs step for runs under a logdir."""
import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_eval_series(metrics_path: pathlib.Path) -> Tuple[List[int], List[float]]:
    steps, scores = [], []
    if not metrics_path.exists():
        return steps, scores
    with metrics_path.open() as f:
        for line in f:
            rec: Dict = json.loads(line)
            if "eval_episode/score" in rec:
                steps.append(rec.get("step", 0))
                scores.append(rec["eval_episode/score"])
    return steps, scores


def plot(logdir: pathlib.Path, outdir: pathlib.Path, per_run: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    runs = [p for p in sorted(logdir.iterdir()) if (p / "metrics.jsonl").exists()]
    if not runs:
        raise SystemExit(f"No runs with metrics.jsonl under {logdir}")

    if per_run:
        for run in runs:
            steps, scores = read_eval_series(run / "metrics.jsonl")
            if not steps:
                continue
            plt.figure(figsize=(6, 4))
            plt.plot(steps, scores, label=run.name)
            plt.xlabel("step")
            plt.ylabel("eval_score")
            plt.title(run.name)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"score_{run.name}.png", dpi=200)
            plt.close()
    else:
        plt.figure(figsize=(7, 5))
        for run in runs:
            steps, scores = read_eval_series(run / "metrics.jsonl")
            if not steps:
                continue
            plt.plot(steps, scores, label=run.name)
        plt.xlabel("step")
        plt.ylabel("eval_score")
        plt.title(f"Eval score vs step ({logdir.name})")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / "score.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot eval_score vs step.")
    parser.add_argument("--logdir", type=pathlib.Path, default=pathlib.Path("./logdir_phase3_prescreen"))
    parser.add_argument("--outdir", type=pathlib.Path, default=None)
    parser.add_argument("--per-run", action="store_true", help="Save one figure per run instead of combined.")
    args = parser.parse_args()

    outdir = args.outdir or args.logdir / "plots"
    plot(args.logdir, outdir, per_run=args.per_run)


if __name__ == "__main__":
    main()
