#!/usr/bin/env bash
set -e
LOGDIR_ROOT=/root/autodl-tmp/projects/SafeDreamer/logdir_phase3_final_full
mkdir -p "$LOGDIR_ROOT"
for s in 0 1 2; do
  echo "[final ours_full] seed=$s"
  MUJOCO_GL=osmesa conda run -n safedreamer python SafeDreamer/train.py \
    --configs osrp_vector osrp_vector_phase2_common ours_full \
    --logdir "$LOGDIR_ROOT/" \
    --seed $s \
    --run.steps 100000 \
    --cost_limit 2.0 \
    --lagrange_multiplier_init 1e-3 \
    --penalty_multiplier_init 1e-3 \
    --ghost_budget 0.25 \
    --ghost_lagrange_multiplier_init 1e-3 \
    --ghost_penalty_multiplier_init 1e-3
done
