#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick ghost signal stats from metrics.jsonl.

Usage:
  python scripts/ghost_stats_phase3b.py --logdir <run_dir>

Outputs min/mean/max for ghost_label_mean, ghost_mask_mean, ghost_pred_mean, ghost_usage_mean (and masked/unmasked if present).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


TARGET_KEYS = [
    "ghost_label_mean",
    "ghost_mask_mean",
    "ghost_pred_mean",
    "ghost_usage_mean",
    "ghost_usage_masked_mean",
    "ghost_usage_unmasked_mean",
]


def load_records(metrics_path: Path) -> List[Dict]:
    lines = metrics_path.read_text().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def stats(values: List[float]):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmin(arr)), float(np.nanmean(arr)), float(np.nanmax(arr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, required=True)
    args = parser.parse_args()
    metrics_path = args.logdir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    records = load_records(metrics_path)
    print(f"# ghost stats for {args.logdir.name}")
    for key in TARGET_KEYS:
        vals = []
        for rec in records:
            for k, v in rec.items():
                if k == key or k.endswith("/" + key):
                    vals.append(v)
        if not vals:
            continue
        lo, mean, hi = stats(vals)
        print(f"{key}: min={lo:.4f}, mean={mean:.4f}, max={hi:.4f}, n={len(vals)}")


if __name__ == "__main__":
    main()
