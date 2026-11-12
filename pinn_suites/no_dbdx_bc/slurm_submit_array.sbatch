#!/usr/bin/env bash
#SBATCH --job-name=nobc
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:<gpu_type>:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/PINN_%A_%a.out
#SBATCH --error=logs/PINN_%A_%a.err
# Warn the job 10 minutes before the time limit, so it can resubmit itself
#SBATCH --signal=B:USR1@600

# Number of models each array task trains, one after another. Also read by
# slurm_submit.sh to size the job array
MODELS_PER_JOB=9

# Train this task's models, resubmitting if it runs out of time
exec ../train_array_task.sh "$MODELS_PER_JOB"
