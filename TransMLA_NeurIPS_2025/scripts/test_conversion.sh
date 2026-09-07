#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
mkdir -p .cache/test-tmp .cache/hf-modules
export TMPDIR="$project_root/.cache/test-tmp"
export HF_MODULES_CACHE="$project_root/.cache/hf-modules"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
python -m pytest -q tests/test_pretrained_conversion.py --basetemp="$TMPDIR/pytest" "$@"
