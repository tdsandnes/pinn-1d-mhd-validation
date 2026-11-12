#!/usr/bin/env bash
# Trains the models of one job array task, one after another. Run by a suite's
# slurm_submit_array.sbatch, from the suite directory, with the number of
# models per task as its argument.

set -euo pipefail

# Exit code train_pinn.py uses when it is stopped before training is done
STOPPED_EXIT_CODE=3

# Arguments
MODELS_PER_JOB="$1"

SUITES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the parameter files for this array task
PARAM_FILES=(./param_files/*.yml)
FIRST=$((SLURM_ARRAY_TASK_ID * MODELS_PER_JOB))
LAST=$((FIRST + MODELS_PER_JOB - 1))
if [ "$LAST" -ge "${#PARAM_FILES[@]}" ]; then
    LAST=$((${#PARAM_FILES[@]} - 1))
fi
MODEL_DIR="./models"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameter files: $FIRST to $LAST of ${#PARAM_FILES[@]}"
echo "=========================================="

OUT_OF_TIME=false
TRAIN_PID=""
# Pass the time limit warning on to the model being trained, if there is one
trap 'OUT_OF_TIME=true; [ -z "$TRAIN_PID" ] || kill -TERM "$TRAIN_PID" 2>/dev/null || true' USR1

FAILED=0
for ((i = FIRST; i <= LAST; i++)); do
    if $OUT_OF_TIME; then
        break
    fi

    PARAM_FILE=${PARAM_FILES[$i]}
    MODEL_NAME=$(basename "$PARAM_FILE" .yml)
    MODEL_NAME=${MODEL_NAME#param_}

    # Trained by an earlier job of this task
    if [ -f "$MODEL_DIR/$MODEL_NAME.pth" ]; then
        echo "Already trained, skipping: $PARAM_FILE"
        continue
    fi

    echo "------------------------------------------"
    echo "Parameter file $i: $PARAM_FILE"
    echo "------------------------------------------"

    # Train in the background, so this script can pass the time limit warning on
    "$SUITES_DIR/train_pinn.sh" "$PARAM_FILE" "$MODEL_DIR" &
    TRAIN_PID=$!

    # The trap interrupts wait, so wait until training has really exited
    while true; do
        STATUS=0
        wait "$TRAIN_PID" || STATUS=$?
        kill -0 "$TRAIN_PID" 2>/dev/null || break
    done
    TRAIN_PID=""

    if $OUT_OF_TIME && [ "$STATUS" -eq "$STOPPED_EXIT_CODE" ]; then
        break
    elif [ "$STATUS" -ne 0 ]; then
        # Carry on with the rest, rather than losing them to one failure
        echo "Training failed with exit code $STATUS: $PARAM_FILE" >&2
        FAILED=1
    fi
done

# The next job starts again from this task's first untrained model
if $OUT_OF_TIME && [ "$i" -le "$LAST" ]; then
    echo "Out of time: resubmitting array task $SLURM_ARRAY_TASK_ID"
    sbatch --array="$SLURM_ARRAY_TASK_ID" ./slurm_submit_array.sbatch
fi

exit "$FAILED"
