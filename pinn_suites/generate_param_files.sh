#!/usr/bin/env bash

set -euo pipefail

SUITES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SUITES_DIR/../config.sh"
activate_conda

conda run --no-capture-output -n "$PINNINIT_ENV" python3 -u \
    "$SUITES_DIR/generate_param_files.py" "$@"
