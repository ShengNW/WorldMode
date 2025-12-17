#!/usr/bin/env bash
# Sequentially launch the remaining prescreen runs after a given PID exits.
set -euo pipefail

TARGET_PID="${1:-}"
LOGDIR="./logdir_phase3_prescreen/"
STEPS=20000
EVAL_EVERY=2000
LOG_EVERY=500
SAVE_EVERY=2000
COST_LIMIT=2.0
LR_INIT=1e-3
PENALTY_INIT=1e-3
GHOST_BUDGET=0.25
GHOST_LR_INIT=1e-3
GHOST_PEN_INIT=1e-3

cd /root/autodl-tmp/projects/SafeDreamer

if [[ -n "${TARGET_PID}" ]]; then
  echo "[chain] waiting for PID ${TARGET_PID}"
  while kill -0 "${TARGET_PID}" 2>/dev/null; do
    sleep 60
  done
fi

run() {
  echo "[chain] start $*"
  MUJOCO_GL=osmesa conda run -n safedreamer python SafeDreamer/train.py "$@"
}

run --configs osrp_vector osrp_vector_phase2_common ours_v0 \
  --logdir "${LOGDIR}" --seed 1 \
  --run.steps "${STEPS}" --run.eval_every "${EVAL_EVERY}" \
  --run.log_every "${LOG_EVERY}" --run.save_every "${SAVE_EVERY}" \
  --cost_limit "${COST_LIMIT}" \
  --lagrange_multiplier_init "${LR_INIT}" \
  --penalty_multiplier_init "${PENALTY_INIT}"

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
