#!/bin/bash
#SBATCH --job-name=sims
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/sims_%j.out
#SBATCH --error=logs/sims_%j.err

# Load modules
SIM_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SIM_DIR/../config.sh"
activate_conda

# Make log directory
mkdir -p "$SIM_DIR/logs"

# Run simulations and sample
bash "$SIM_DIR/run_all.sh"
