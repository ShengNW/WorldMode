#!/usr/bin/env bash
# Phase3 prescreen reruns with second-based logging. Sequential to avoid OOM.
set -euo pipefail

LOGDIR="./logdir_phase3_prescreen_v2/"
STEPS=20000
EVAL_EVERY=2000          # steps
LOG_EVERY=5              # seconds
SAVE_EVERY=1200          # seconds
COST_LIMIT=2.0
LR_INIT=1e-3
PENALTY_INIT=1e-3
GHOST_BUDGET=0.25
GHOST_LR_INIT=1e-3
GHOST_PEN_INIT=1e-3

cd /root/autodl-tmp/projects/SafeDreamer

run() {
  echo "[chain_v2] start $*"
  MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 stdbuf -oL -eL conda run -n safedreamer \
    python SafeDreamer/train.py "$@"
  echo "[chain_v2] done $*"
}

# 1) ours_v0 seed=1
run --configs osrp_vector osrp_vector_phase2_common ours_v0 \
  --logdir "${LOGDIR}" --seed 1 \
  --run.steps "${STEPS}" --run.eval_every "${EVAL_EVERY}" \
  --run.log_every "${LOG_EVERY}" --run.save_every "${SAVE_EVERY}" \
  --cost_limit "${COST_LIMIT}" \
  --lagrange_multiplier_init "${LR_INIT}" \
  --penalty_multiplier_init "${PENALTY_INIT}"

# 2) ours_full seed=0
run --configs osrp_vector osrp_vector_phase2_common ours_full \
  --logdir "${LOGDIR}" --seed 0 \
  --run.steps "${STEPS}" --run.eval_every "${EVAL_EVERY}" \
  --run.log_every "${LOG_EVERY}" --run.save_every "${SAVE_EVERY}" \
  --cost_limit "${COST_LIMIT}" \
  --lagrange_multiplier_init "${LR_INIT}" \
  --penalty_multiplier_init "${PENALTY_INIT}" \
  --ghost_budget "${GHOST_BUDGET}" \
  --ghost_lagrange_multiplier_init "${GHOST_LR_INIT}" \
  --ghost_penalty_multiplier_init "${GHOST_PEN_INIT}"

# 3) ours_full seed=1
run --configs osrp_vector osrp_vector_phase2_common ours_full \
  --logdir "${LOGDIR}" --seed 1 \
  --run.steps "${STEPS}" --run.eval_every "${EVAL_EVERY}" \
  --run.log_every "${LOG_EVERY}" --run.save_every "${SAVE_EVERY}" \
  --cost_limit "${COST_LIMIT}" \
  --lagrange_multiplier_init "${LR_INIT}" \
  --penalty_multiplier_init "${PENALTY_INIT}" \
  --ghost_budget "${GHOST_BUDGET}" \
  --ghost_lagrange_multiplier_init "${GHOST_LR_INIT}" \
  --ghost_penalty_multiplier_init "${GHOST_PEN_INIT}"
