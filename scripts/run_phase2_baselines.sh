#!/usr/bin/env bash
set -euo pipefail

# Run short Phase 2 baseline training jobs.
# Usage: ./scripts/run_phase2_baselines.sh {unsafe|clean|hallu_strict|hallu_relaxed} [seed] [steps]

BASELINE=${1:-clean}
SEED=${2:-0}
STEPS=${RUN_STEPS:-50000}
if [ $# -ge 3 ]; then
  STEPS=$3
fi

LOGDIR_ROOT=${LOGDIR_ROOT:-./logdir_phase2/}
EVAL_EVERY=${EVAL_EVERY:-2000}
LOG_EVERY=${LOG_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-2000}
PYTHON_BIN=${PYTHON_BIN:-python}

COMMON_CONFIGS="osrp_vector osrp_vector_phase2_common"

case "${BASELINE}" in
  unsafe)
    CONFIGS="${COMMON_CONFIGS} osrp_vector_unsafe"
    ;;
  clean)
    CONFIGS="${COMMON_CONFIGS} osrp_vector_safedreamer_clean"
    ;;
  hallu_strict)
    CONFIGS="${COMMON_CONFIGS} osrp_vector_safedreamer_hallu_strict"
    ;;
  hallu_relaxed)
    CONFIGS="${COMMON_CONFIGS} osrp_vector_safedreamer_hallu_relaxed"
    ;;
  *)
    echo "Baseline must be one of: unsafe | clean | hallu_strict | hallu_relaxed" >&2
    exit 1
    ;;
esac

echo "[run_phase2] baseline=${BASELINE} seed=${SEED} steps=${STEPS}"

${PYTHON_BIN} SafeDreamer/train.py \
  --configs ${CONFIGS} \
  --logdir "${LOGDIR_ROOT}" \
  --seed "${SEED}" \
  --run.steps "${STEPS}" \
  --run.eval_every "${EVAL_EVERY}" \
  --run.log_every "${LOG_EVERY}" \
  --run.save_every "${SAVE_EVERY}"
