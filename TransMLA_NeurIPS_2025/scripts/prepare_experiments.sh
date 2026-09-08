#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/eval_env.sh"
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
"$TRANSMLA_PYTHON" -m transmla.experiments.stage_sources
if [[ ! -f .cache/experiment-data/manifest.json ]]; then
    "$TRANSMLA_PYTHON" -m transmla.experiments.stage_data --out .cache/experiment-data
fi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
"$TRANSMLA_PYTHON" -m transmla.experiments.campaign prepare \
    --data .cache/experiment-data "$@"
