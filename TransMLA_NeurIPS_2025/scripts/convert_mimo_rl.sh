#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
python -m transmla.convert_pretrained --model-path "${TRANSMLA_SOURCE_MODEL:-XiaomiMiMo/MiMo-7B-RL-0530}" "$@"
