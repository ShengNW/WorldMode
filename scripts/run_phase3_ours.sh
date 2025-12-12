#!/usr/bin/env bash
set -euo pipefail

# Run Phase 3 Ours lines (v0/full) with short sanity-friendly defaults.
# Usage: ./scripts/run_phase3_ours.sh {ours_v0|ours_full} [seed] [steps]

LINE=${1:-ours_v0}
SEED=${2:-0}
STEPS=${RUN_STEPS:-10000}
if [ $# -ge 3 ]; then
  STEPS=$3
fi

LOGDIR_ROOT=${LOGDIR_ROOT:-./logdir_phase3/}
EVAL_EVERY=${EVAL_EVERY:-2000}
LOG_EVERY=${LOG_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-2000}
PYTHON_BIN=${PYTHON_BIN:-python}

COMMON_CONFIGS="osrp_vector osrp_vector_phase2_common"

case "${LINE}" in
  ours_v0)
    CONFIGS="${COMMON_CONFIGS} ours_v0"
    ;;
  ours_full)
    CONFIGS="${COMMON_CONFIGS} ours_full"
    ;;
  *)
    echo "Line must be one of: ours_v0 | ours_full" >&2
    exit 1
    ;;
esac

echo "[run_phase3_ours] line=${LINE} seed=${SEED} steps=${STEPS}"

${PYTHON_BIN} SafeDreamer/train.py \
  --configs ${CONFIGS} \
  --logdir "${LOGDIR_ROOT}" \
  --seed "${SEED}" \
  --run.steps "${STEPS}" \
  --run.eval_every "${EVAL_EVERY}" \
  --run.log_every "${LOG_EVERY}" \
  --run.save_every "${SAVE_EVERY}"
