#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/eval_env.sh"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
"$TRANSMLA_PYTHON" -m pytest -q tests/test_pretrained_conversion.py tests/test_experiment_support.py tests/test_gpu_validation.py \
  --basetemp="$TMPDIR/pytest-experiments" "$@"
