#!/usr/bin/env python
"""Plot lambda (lag_lambda_post_* or nu_*) vs step from diag.jsonl."""
import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_lambda(diag_path: pathlib.Path) -> Tuple[List[int], List[float], List[float]]:
    steps, lam_hard, lam_ghost = [], [], []
    if not diag_path.exists():
        return steps, lam_hard, lam_ghost
    with diag_path.open() as f:
        for line in f:
            rec: Dict = json.loads(line)
            step = rec.get("step", 0)
            hard = rec.get("lag_lambda_post_hard", rec.get("nu_hard"))
            ghost = rec.get("lag_lambda_post_ghost", rec.get("nu_ghost"))
            if hard is not None:
                steps.append(step)
                lam_hard.append(hard)
            if ghost is not None:
                # align ghost steps with ghost values; reuse same step list if lengths match
                if len(lam_ghost) < len(steps):
                    lam_ghost.append(ghost)
                else:
                    lam_ghost.append(ghost)
    return steps, lam_hard, lam_ghost


def plot(logdir: pathlib.Path, outdir: pathlib.Path, per_run: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    runs = [p for p in sorted(logdir.iterdir()) if (p / "diag.jsonl").exists()]
    if not runs:
        raise SystemExit(f"No runs with diag.jsonl under {logdir}")

    if per_run:
        for run in runs:
            steps, hard, ghost = read_lambda(run / "diag.jsonl")
            if not steps:
                continue
            plt.figure(figsize=(6, 4))
            plt.plot(steps, hard, label=f"{run.name}-hard")
            if ghost:
                plt.plot(steps[: len(ghost)], ghost, label=f"{run.name}-ghost", linestyle="--")
            plt.xlabel("step")
            plt.ylabel("lambda")
            plt.title(run.name)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"lambda_{run.name}.png", dpi=200)
            plt.close()
    else:
        plt.figure(figsize=(7, 5))
        for run in runs:
            steps, hard, ghost = read_lambda(run / "diag.jsonl")
            if not steps:
                continue
            plt.plot(steps, hard, label=f"{run.name}-hard")
            if ghost:
                plt.plot(steps[: len(ghost)], ghost, linestyle="--", label=f"{run.name}-ghost")
        plt.xlabel("step")
        plt.ylabel("lambda")
        plt.title(f"Lambda vs step ({logdir.name})")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / "lambda.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot lambda diagnostics vs step.")
    parser.add_argument("--logdir", type=pathlib.Path, default=pathlib.Path("./logdir_phase3_prescreen"))
    parser.add_argument("--outdir", type=pathlib.Path, default=None)
    parser.add_argument("--per-run", action="store_true", help="Save one figure per run instead of combined.")
    args = parser.parse_args()

    outdir = args.outdir or args.logdir / "plots"
    plot(args.logdir, outdir, per_run=args.per_run)


if __name__ == "__main__":
    main()
