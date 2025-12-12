#!/usr/bin/env bash
set -euo pipefail

# Eval-only helper for Phase 3 Ours runs.
# Usage: ./scripts/eval_phase3_ours.sh {ours_v0|ours_full} <ckpt_dir_or_file> [seed] [eval_steps]

LINE=${1:-}
TARGET_PATH=${2:-}
SEED=${3:-0}
EVAL_STEPS=${RUN_STEPS:-2000}
if [ $# -ge 4 ]; then
  EVAL_STEPS=$4
fi

if [ -z "${LINE}" ] || [ -z "${TARGET_PATH}" ]; then
  echo "Usage: $0 {ours_v0|ours_full} <ckpt_dir_or_file> [seed] [eval_steps]" >&2
  exit 1
fi

LOGDIR_ROOT=${EVAL_LOGDIR_ROOT:-./logdir_phase3_eval/}
EVAL_EPISODES=${EVAL_EPISODES:-5}
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

if [ -d "${TARGET_PATH}" ]; then
  CKPT_PATH=${CKPT_PATH:-$(ls -t "${TARGET_PATH}"/checkpoint*.ckpt 2>/dev/null | head -n 1 || true)}
  if [ -z "${CKPT_PATH}" ]; then
    CKPT_PATH="${TARGET_PATH}/checkpoint.ckpt"
  fi
else
  CKPT_PATH="${TARGET_PATH}"
fi

echo "[eval_phase3_ours] line=${LINE} seed=${SEED} ckpt=${CKPT_PATH} steps=${EVAL_STEPS}"

${PYTHON_BIN} SafeDreamer/train.py \
  --configs ${CONFIGS} \
  --logdir "${LOGDIR_ROOT}" \
  --seed "${SEED}" \
  --run.script eval_only \
  --run.steps "${EVAL_STEPS}" \
  --run.eval_eps "${EVAL_EPISODES}" \
  --run.from_checkpoint "${CKPT_PATH}"
