# Simulations

Runs the 1D MHD simulations and samples training, validation and test sets
from their snapshots.

1. `simulate.py` runs one simulation and writes snapshots to its `data_dir/snapshot_dir`.
2. `sample_data.py` samples datasets from snapshots and writes them to the same `data_dir`.

## Naming

- A simulation is associated with its own directory, which contains exactly
  one config file named `sim_config_*.py`.
- Every config file defines a dataclass named `Config`.

## Running

To submit a job that runs and samples all simulations:

```bash
sbatch slurm_submit.sh
```

To run and sample all simulations:

```bash
./run_all.sh
```

To run a single simulation:

```bash
source ../config.sh
conda activate "$DEDALUS_ENV"
python3 simulate.py 3freq_1Ma_500Re_500Rm/sim_config_3freq_1Ma_500Re_500Rm.py
```

To sample a single simulation with a single sampling config:

```bash
source ../config.sh
conda activate "$PINNINIT_ENV"
python3 sample_data.py t20to30_eachcol032768sobol_data001025_bc016384/sampling_config_t20to30_eachcol032768sobol_data001025_bc016384_repeat01.py \
    3freq_1Ma_500Re_500Rm/sim_config_3freq_1Ma_500Re_500Rm.py
```

Sampling reads the snapshots, so the simulation has to have been run first.
The sampling configs also have to exist: `run_all.sh` generates them, otherwise
run `sampling_configs/generate_configs.sh`.

## Adding a simulation

1. Make the directory and its `sim_config_*.py`.
2. Add it to `SIM_CONFIGS` in `run_all_config.sh`, pointing it at a sampling group.

## Adding a sampling config

1. Add to `sampling_configs/generate_configs.sh`.
2. Add the subdirectory name to the relevant sampling groups in `run_all_config.sh`.

`generate_configs.sh` wipes and rebuilds every subdirectory in `sampling_configs/` from `template.py`.
`run_all.sh` runs it before anything else.
