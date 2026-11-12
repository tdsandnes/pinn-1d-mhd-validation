#!/usr/bin/env bash

set -euo pipefail

# Arguments
SUITE_NAME="$1"

SUITES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SUITES_DIR/../config.sh"
activate_conda

SUITE_DIR="$SUITES_DIR/$SUITE_NAME"

# Make sure the suite exists and is set up
for f in "$SUITE_DIR/suite.yml" "$SUITE_DIR/slurm_submit_array.sbatch"; do
    [ -f "$f" ] || { echo "Error: '$f' not found" >&2; exit 1; }
done

cd "$SUITE_DIR"

shopt -s nullglob
PARAM_FILES=(./param_files/*.yml)

if [ ${#PARAM_FILES[@]} -eq 0 ]; then
    echo "Error: No .yml files found in $SUITE_DIR/param_files" >&2
    exit 1
fi

# Number of models each array task trains, set in the suite's sbatch file
MODELS_PER_JOB=$(sed -n 's/^MODELS_PER_JOB=\([1-9][0-9]*\)$/\1/p' slurm_submit_array.sbatch)
if [ -z "$MODELS_PER_JOB" ]; then
    echo "Error: No MODELS_PER_JOB=<n> line in $SUITE_DIR/slurm_submit_array.sbatch" >&2
    exit 1
fi

mkdir -p logs

N_TASKS=$(((${#PARAM_FILES[@]} + MODELS_PER_JOB - 1) / MODELS_PER_JOB))
ARRAY_MAX=$((N_TASKS - 1))
echo "Submitting job array for ${#PARAM_FILES[@]} parameter files," \
    "$MODELS_PER_JOB per task: 0-$ARRAY_MAX"

sbatch --array=0-"$ARRAY_MAX" "$SUITE_DIR/slurm_submit_array.sbatch"
