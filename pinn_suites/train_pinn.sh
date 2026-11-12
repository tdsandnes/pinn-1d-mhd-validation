#!/usr/bin/env bash

set -euo pipefail

# Arguments
PARAM_FILE="$(realpath "$1")"
MODEL_DIR="$(realpath "$2")"

SUITES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SUITES_DIR/../config.sh"

# Python environment. Activated here rather than using conda run, so that
# python replaces this script below and gets the signals sent to it
activate_conda
eval "$(conda shell.bash hook)"
set +u # Activation scripts can use unset variables
conda activate "$PINNINIT_ENV"
set -u

# Run training
exec python3 -u "$SUITES_DIR/train_pinn.py" \
    --param_file "$PARAM_FILE" --out_dir "$MODEL_DIR"
