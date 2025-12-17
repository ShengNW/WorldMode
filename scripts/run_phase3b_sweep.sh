#!/usr/bin/env bash
set -euo pipefail

# Phase3b small grid sweep for ours_full (extensible to ours_v0).
# Usage:
#   bash scripts/run_phase3b_sweep.sh [method] [steps] [seeds]
# Defaults: method=ours_full, steps=20000, seeds="0"
#
# Each run:
# - launches SafeDreamer/train.py with overrides
# - dumps overrides into logdir/override.txt
# - plots metrics via plot_phase3b.py
# - appends the run's summary.csv row into sweep_summary.csv

METHOD=${1:-ours_full}
STEPS=${2:-20000}
SEEDS=${3:-0}           # space-separated list, e.g., "0 1"
LOGDIR_ROOT=${LOGDIR_ROOT:-logdir_phase3}
# Normalize to absolute path (avoid cwd-dependent mix-ups).
if [[ "${LOGDIR_ROOT}" != /* ]]; then
  LOGDIR_ROOT="$(pwd)/${LOGDIR_ROOT}"
fi
CONDA_ENV=${CONDA_ENV:-safedreamer}
PYBIN=${PYBIN:-python}

# Pre-picked 6 combos (cost_limit, nu_init, pen_init, ghost_budget, ghost_nu_init, ghost_pen_init)
GRID=(
  "cl=2.0,nu=1e-3,pen=1e-3,gb=0.25,gnu=1e-3,gpen=1e-3"
  "cl=2.0,nu=1e-3,pen=5e-3,gb=0.10,gnu=1e-3,gpen=5e-3"
  "cl=2.0,nu=1e-2,pen=1e-2,gb=0.05,gnu=1e-2,gpen=1e-2"
  "cl=4.0,nu=1e-3,pen=1e-3,gb=0.25,gnu=1e-3,gpen=1e-3"
  "cl=4.0,nu=1e-3,pen=5e-3,gb=0.10,gnu=1e-3,gpen=5e-3"
  "cl=4.0,nu=1e-2,pen=1e-2,gb=0.05,gnu=1e-2,gpen=1e-2"
)

SWEEP_SUMMARY=${LOGDIR_ROOT}/sweep_summary.csv

echo "[sweep] method=${METHOD} steps=${STEPS} seeds=${SEEDS}"
mkdir -p "${LOGDIR_ROOT}"

for combo in "${GRID[@]}"; do
  IFS=',' read -r -a parts <<< "${combo}"
  unset cl nu pen gb gnu gpen
  for p in "${parts[@]}"; do
    key=${p%%=*}
    val=${p#*=}
    case "${key}" in
      cl) cl=${val} ;;
      nu) nu=${val} ;;
      pen) pen=${val} ;;
      gb) gb=${val} ;;
      gnu) gnu=${val} ;;
      gpen) gpen=${val} ;;
      *) echo "Unknown key ${key}" >&2; exit 1 ;;
    esac
  done

  for seed in ${SEEDS}; do
    echo "[sweep] combo=${combo} seed=${seed}"
    MUJOCO_GL=${MUJOCO_GL:-osmesa} \
    conda run -n "${CONDA_ENV}" ${PYBIN} SafeDreamer/train.py \
      --configs osrp_vector osrp_vector_phase2_common "${METHOD}" \
      --logdir "${LOGDIR_ROOT}/" \
      --seed "${seed}" \
      --run.steps "${STEPS}" \
      --cost_limit "${cl}" \
      --lagrange_multiplier_init "${nu}" \
      --penalty_multiplier_init "${pen}" \
      --ghost_budget "${gb}" \
      --ghost_lagrange_multiplier_init "${gnu}" \
      --ghost_penalty_multiplier_init "${gpen}"

    # pick freshest logdir
    full_run_dir=$(ls -td ${LOGDIR_ROOT}/* | head -n 1)
    echo "[sweep] latest run_dir=${full_run_dir}"

    # Record overrides
    {
      echo "method=${METHOD}"
      echo "seed=${seed}"
      echo "steps=${STEPS}"
      echo "cost_limit=${cl}"
      echo "lagrange_multiplier_init=${nu}"
      echo "penalty_multiplier_init=${pen}"
      echo "ghost_budget=${gb}"
      echo "ghost_lagrange_multiplier_init=${gnu}"
      echo "ghost_penalty_multiplier_init=${gpen}"
    } > "${full_run_dir}/override.txt"

    # Plot and summarize
    conda run -n "${CONDA_ENV}" ${PYBIN} scripts/plot_phase3b.py --logdir "${full_run_dir}"

    summary_path="${full_run_dir}/plots/summary.csv"
    if [ -f "${summary_path}" ]; then
      if [ ! -f "${SWEEP_SUMMARY}" ]; then
        cp "${summary_path}" "${SWEEP_SUMMARY}"
      else
        tail -n +2 "${summary_path}" >> "${SWEEP_SUMMARY}"
      fi
    fi
  done
done

echo "[sweep] done. Aggregated: ${SWEEP_SUMMARY}"
