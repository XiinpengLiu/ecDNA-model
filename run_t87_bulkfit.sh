#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CORES="${CORES:-8}"
FIT_WORKERS="${FIT_WORKERS:-8}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

echo "[run] cwd=$PWD"
echo "[run] CORES=$CORES FIT_WORKERS=$FIT_WORKERS"
echo "[run] MPLCONFIGDIR=$MPLCONFIGDIR"

if [[ "${UNLOCK:-0}" == "1" ]]; then
  echo "[run] checking for running snakemake processes"
  pgrep -af snakemake || true
  echo "[run] unlocking Snakemake working directory"
  snakemake \
    --snakefile workflow/Snakefile \
    --configfile configs/t87_drug_bulkfit.yaml \
    --unlock
  exit 0
fi

if [[ "${CHECK_RUNNING:-1}" == "1" ]]; then
  echo "[run] running snakemake processes, if any:"
  pgrep -af snakemake || true
fi

echo "[run] starting Snakemake workflow"
snakemake \
  --snakefile workflow/Snakefile \
  --configfile configs/t87_drug_bulkfit.yaml \
  --cores "$CORES" \
  --config fit_workers="$FIT_WORKERS" \
  --rerun-incomplete \
  --printshellcmds
