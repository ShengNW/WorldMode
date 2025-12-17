#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot Phase3b metrics and emit a one-line summary.

Usage:
  python scripts/plot_phase3b.py --logdir LOGDIR [--outdir OUTDIR] [--smooth 0.95]

Deps: stdlib + numpy + matplotlib.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Keys we try to plot, in a readable order.
PLOT_KEYS = [
    ("eval_episode/score_ema", "Eval Score (EMA)"),
    ("eval_episode/score", "Eval Score"),
    ("eval_episode/cost_ema", "Eval Cost (EMA)"),
    ("eval_episode/cost", "Eval Cost"),
    ("episode/cost_ema", "Train Cost (EMA)"),
    ("episode/cost", "Train Cost (raw)"),
    ("lagrange_multiplier", "Nu Hard"),
    ("ghost_lagrange_multiplier", "Nu Ghost"),
    ("ghost_usage_mean", "Ghost Usage"),
    ("ghost_usage_masked_mean", "Ghost Usage Masked"),
    ("ghost_usage_unmasked_mean", "Ghost Usage Unmasked"),
    ("ghost_pred_mean", "Ghost Pred Mean"),
    ("ghost_label_mean", "Ghost Label Mean"),
    ("ghost_mask_mean", "Ghost Mask Mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, type=Path, help="Path to run dir containing metrics.jsonl")
    parser.add_argument("--outdir", type=Path, default=None, help="Where to save plots/summary (default: <logdir>/plots)")
    parser.add_argument("--smooth", type=float, default=0.95, help="EMA smoothing in [0,1); 0 disables. Default 0.95.")
    return parser.parse_args()


def _match(rec_key: str, target: str) -> bool:
    return rec_key == target or rec_key.endswith("/" + target)


def read_metrics(metrics_path: Path) -> List[Dict]:
    lines = metrics_path.read_text().splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def extract_series(records: Iterable[Dict], key: str) -> Tuple[np.ndarray, np.ndarray]:
    steps: List[float] = []
    vals: List[float] = []
    for rec in records:
        if "step" not in rec:
            continue
        for k, v in rec.items():
            if _match(k, key):
                steps.append(rec["step"])
                vals.append(v)
                break
    if not steps:
        return np.array([]), np.array([])
    return np.asarray(steps, dtype=float), np.asarray(vals, dtype=float)


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0 or alpha <= 0 or alpha >= 1:
        return values
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def plot_one(steps: np.ndarray, values: np.ndarray, title: str, out_path: Path, smooth: float) -> None:
    plt.figure(figsize=(6, 4))
    marker = "o" if len(steps) == 1 else None
    if smooth > 0 and smooth < 1:
        smoothed = ema(values, smooth)
        plt.plot(steps, smoothed, label=f"EMA {smooth}", marker=marker)
        plt.plot(steps, values, alpha=0.3, linewidth=1.0, label="raw", marker=marker)
    else:
        plt.plot(steps, values, label="raw", marker=marker)
    plt.title(title)
    plt.xlabel("step")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def best_eval_score(records: List[Dict]) -> Tuple[float, float, float]:
    """Return best_score, step_at_best, eval_cost_at_best (cost_ema preferred)."""
    best = None
    best_step = None
    best_cost = None
    score_candidates = ["eval_episode/score_ema", "eval_episode/score"]
    cost_candidates = ["eval_episode/cost_ema", "eval_episode/cost"]

    score_key = None
    for cand in score_candidates:
        if any(_match(k, cand) for rec in records for k in rec.keys()):
            score_key = cand
            break
    if score_key is None:
        return np.nan, np.nan, np.nan

    for rec in records:
        for k, v in rec.items():
            if not _match(k, score_key):
                continue
            score = v
            if best is None or score > best:
                best = score
                best_step = rec.get("step", np.nan)
                best_cost = np.nan
                for cand in cost_candidates:
                    for ck, cv in rec.items():
                        if _match(ck, cand):
                            best_cost = cv
                            break
    if best is None:
        return np.nan, np.nan, np.nan
    return best, best_step, best_cost


def last_value(records: List[Dict], key: str) -> float:
    last = np.nan
    for rec in records:
        for k, v in rec.items():
            if _match(k, key):
                last = v
    return last


def last_value_any(records: List[Dict], keys: List[str]) -> float:
    last = np.nan
    for rec in records:
        for key in keys:
            for k, v in rec.items():
                if _match(k, key):
                    last = v
    return last


def main():
    args = parse_args()
    logdir: Path = args.logdir
    outdir: Path = args.outdir or logdir / "plots"
    metrics_path = logdir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found")

    records = read_metrics(metrics_path)

    # Plot curves
    series_cache = {}
    for key, title in PLOT_KEYS:
        steps, vals = extract_series(records, key)
        if steps.size == 0:
            continue
        series_cache[key] = (steps, vals)
        smooth_this = args.smooth
        if key.endswith("_ema"):
            smooth_this = 0.0
        fname = key.replace("/", "_") + ".png"
        plot_one(steps, vals, f"{title}", outdir / fname, smooth_this)

    # Combined plot for train cost raw vs ema if both exist.
    steps_raw, vals_raw = series_cache.get("episode/cost", (np.array([]), np.array([])))
    steps_ema, vals_ema = series_cache.get("episode/cost_ema", (np.array([]), np.array([])))
    if steps_raw.size and vals_raw.size and steps_ema.size and vals_ema.size:
        plt.figure(figsize=(6, 4))
        marker_raw = "o" if len(steps_raw) == 1 else None
        marker_ema = "o" if len(steps_ema) == 1 else None
        if args.smooth > 0 and args.smooth < 1:
            plt.plot(steps_raw, ema(vals_raw, args.smooth), label=f"episode/cost raw EMA {args.smooth}", marker=marker_raw)
        plt.plot(steps_raw, vals_raw, label="episode/cost (raw)", alpha=0.5, marker=marker_raw)
        plt.plot(steps_ema, vals_ema, label="episode/cost_ema", marker=marker_ema)
        plt.title("Train Cost (raw vs ema)")
        plt.xlabel("step")
        plt.legend()
        plt.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(outdir / "episode_cost_combined.png")
        plt.close()

    # Build summary
    best_score, best_step, eval_cost_at_best = best_eval_score(records)
    summary = {
        "logdir": logdir.name,
        "last_step": last_value(records, "step"),
        "best_eval_score": best_score,
        "best_eval_step": best_step,
        "eval_cost_at_best": eval_cost_at_best,
        "last_eval_score": last_value_any(records, ["eval_episode/score_ema", "eval_episode/score"]),
        "last_eval_cost": last_value_any(records, ["eval_episode/cost_ema", "eval_episode/cost"]),
        "last_cost_ema": last_value(records, "episode/cost_ema"),
        "last_nu_hard": last_value(records, "lagrange_multiplier"),
        "last_nu_ghost": last_value(records, "ghost_lagrange_multiplier"),
        "last_ghost_label_mean": last_value(records, "ghost_label_mean"),
        "last_ghost_pred_mean": last_value(records, "ghost_pred_mean"),
        "last_ghost_usage_mean": last_value_any(records, ["ghost_usage_mean", "ghost_usage_masked_mean", "ghost_usage_unmasked_mean"]),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved plots to {outdir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
