#!/usr/bin/env python
"""
Summarize SafeDreamer experiment runs into a CSV.

Expected run structure inside a logdir:
  <logdir>/<timestamp>_<method>_<task>_<seed>/
    metrics.jsonl
    diag.jsonl (optional)
    config.yaml

This script reads metrics.jsonl for eval metrics and diag.jsonl for
dual/lambda diagnostics, then writes summary.csv with key indicators.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List

try:
    import yaml
except ImportError:
    yaml = None  # We fall back to empty config if PyYAML is missing.


def read_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a .jsonl file, tolerating NaN/Inf tokens."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                fixed = (
                    line.replace("NaN", "null")
                    .replace("Infinity", "null")
                    .replace("-Infinity", "null")
                )
                yield json.loads(fixed)


def get_config(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def summarize_run(run_dir: pathlib.Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    diag_path = run_dir / "diag.jsonl"
    config_path = run_dir / "config.yaml"

    config = get_config(config_path)
    method = config.get("method")
    seed = config.get("seed")
    cost_limit = config.get("cost_limit")

    eval_records: List[Dict[str, Any]] = []
    last_step = None
    if metrics_path.exists():
        for rec in read_jsonl(metrics_path):
            step = rec.get("step")
            if step is not None:
                last_step = step
            if "eval_episode/score" in rec:
                eval_records.append(
                    {
                        "step": step,
                        "score": rec.get("eval_episode/score"),
                        "cost_ema": rec.get("eval_episode/cost_ema"),
                        "cost": rec.get("eval_episode/cost"),
                    }
                )

    best_eval_score = None
    eval_cost_at_best = None
    last_eval_cost_ema = None
    if eval_records:
        best = max(eval_records, key=lambda r: r.get("score", -math.inf))
        best_eval_score = best.get("score")
        eval_cost_at_best = (
            best.get("cost_ema") if best.get("cost_ema") is not None else best.get("cost")
        )
        last_eval_cost_ema = eval_records[-1].get("cost_ema")

    lambda_vals: List[float] = []
    g_vals: List[float] = []
    overflow_vals: List[float] = []
    diag_steps: List[int] = []
    if diag_path.exists():
        for rec in read_jsonl(diag_path):
            step = rec.get("step")
            if step is not None:
                diag_steps.append(step)
            for key, val in rec.items():
                if not isinstance(val, (int, float)) or (isinstance(val, float) and math.isnan(val)):
                    continue
                if (
                    key in {"nu_hard", "nu_ghost", "lagrange_multiplier", "ghost_lagrange_multiplier"}
                    or "lag_lambda" in key
                ):
                    lambda_vals.append(float(val))
                if key.startswith("lag_g_"):
                    g_vals.append(float(val))
            overflow = rec.get("grad_overflow")
            if isinstance(overflow, (int, float)) and not math.isnan(overflow):
                overflow_vals.append(float(overflow))

    lambda_max = max(lambda_vals) if lambda_vals else None
    g_positive_rate = (sum(1 for v in g_vals if v > 0) / len(g_vals)) if g_vals else None
    overflow_rate = (
        sum(1 for v in overflow_vals if v > 0) / len(overflow_vals)
    ) if overflow_vals else None

    return {
        "run": run_dir.name,
        "method": method,
        "seed": seed,
        "cost_limit": cost_limit,
        "best_eval_score": best_eval_score,
        "eval_cost_at_best": eval_cost_at_best,
        "last_eval_cost_ema": last_eval_cost_ema,
        "lambda_max": lambda_max,
        "g_positive_rate": g_positive_rate,
        "overflow_rate": overflow_rate,
        "last_step": last_step,
        "diag_steps": max(diag_steps) if diag_steps else None,
        "logdir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SafeDreamer runs into CSV.")
    parser.add_argument(
        "--logdir",
        type=pathlib.Path,
        default=pathlib.Path("./logdir_phase3_prescreen"),
        help="Root directory containing run subfolders.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Path to write summary CSV. Default: <logdir>/summary.csv",
    )
    args = parser.parse_args()

    logdir: pathlib.Path = args.logdir
    if not logdir.exists():
        raise SystemExit(f"Logdir not found: {logdir}")

    runs = [p for p in sorted(logdir.iterdir()) if (p / "metrics.jsonl").exists()]
    if not runs:
        raise SystemExit(f"No runs with metrics.jsonl under {logdir}")

    records = [summarize_run(r) for r in runs]

    out_path = args.output or (logdir / "summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run",
        "method",
        "seed",
        "cost_limit",
        "last_step",
        "best_eval_score",
        "eval_cost_at_best",
        "last_eval_cost_ema",
        "lambda_max",
        "g_positive_rate",
        "overflow_rate",
        "diag_steps",
        "logdir",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} rows to {out_path}")


if __name__ == "__main__":
    main()
