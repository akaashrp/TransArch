#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/eval_env.sh"
python -m venv --system-site-packages .venv-eval
"$TRANSMLA_PYTHON" -m pip install -r requirements-evaluation.txt
"$TRANSMLA_PYTHON" -m pip freeze > .cache/evaluation-environment.txt
