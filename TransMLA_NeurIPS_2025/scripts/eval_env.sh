#!/usr/bin/env bash
# Source this script from login shells and Slurm jobs.
set -euo pipefail
transmla_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate vllm
export TRANSMLA_PYTHON="$transmla_project/.venv-eval/bin/python"
export TMPDIR="$transmla_project/.cache/runtime-tmp/${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
export HF_MODULES_CACHE="$transmla_project/.cache/hf-modules"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$HF_MODULES_CACHE"
cd "$transmla_project"
