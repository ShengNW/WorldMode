#!/usr/bin/env bash
set -e
LOGDIR_ROOT=/root/autodl-tmp/projects/SafeDreamer/logdir_phase3_final_v0
mkdir -p "$LOGDIR_ROOT"
for s in 0 1 2; do
  echo "[final ours_v0] seed=$s"
  MUJOCO_GL=osmesa conda run -n safedreamer python SafeDreamer/train.py \
    --configs osrp_vector osrp_vector_phase2_common ours_v0 \
    --logdir "$LOGDIR_ROOT/" \
    --seed $s \
    --run.steps 100000 \
    --cost_limit 2.0 \
    --lagrange_multiplier_init 1e-3 \
    --penalty_multiplier_init 1e-3
    # ghost params ignored in v0
    # --ghost_budget 0.25
    # --ghost_lagrange_multiplier_init 1e-3
    # --ghost_penalty_multiplier_init 1e-3
    
done
