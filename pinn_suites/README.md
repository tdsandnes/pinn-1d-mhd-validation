# PINN suites

Trains the PINNs and plots the results. Each subdirectory is a suite: a sweep
over some set of parameters, plus the plotting scripts for it.

1. `generate_param_files.py` reads a suite's `suite.yml` and writes one parameter
   file per model to `<suite>/param_files`.
2. `train_pinn.py` trains one model from one parameter file and saves it to
   `<suite>/models`.

## suite.yml

- `base_model_name` is required.
- Anything else overrides a default in `suite_defaults.py`. Anything not set
  falls back on those defaults.
- A list gets swept over, and every combination of the lists becomes a model.
  Swept quantities also get added to the model name, e.g. `layers0006`.
- `methods` is the exception. A flat list is one set of methods switched on
  together. To sweep, give a list of lists.
- A `sampling_configs` entry can be a whole sampling config directory, which gets
  expanded into its repeats. `repeats: N` uses repeats 1 to N, `repeats: [...]`
  uses the ones listed.
- `template` picks the template from `../pinn_templates`.

## Running

To generate the parameter files and submit a job array that trains every model
in a suite:

```bash
./generate_param_files.sh fiducial
./slurm_submit.sh fiducial
```

Set the partition and GPU type in each suite's `slurm_submit_array.sbatch`
to match your cluster before submitting.

`generate_param_files.sh` deletes the suite's old parameter files first.
It will also fail if a data file is missing, so the simulations have to have
been run and sampled.

Each array task trains `MODELS_PER_JOB` models one after another, set in the
suite's `slurm_submit_array.sbatch`. A model that is already in
`<suite>/models` is skipped, so start a suite with an empty `models` directory.

To train a single model without slurm:

```bash
./train_pinn.sh fiducial/param_files/param_<model_name>.yml fiducial/models
```

To make a plot once training is done:

```bash
source ../config.sh
conda activate "$PINNINIT_ENV"
python3 fiducial/plot_snapshots.py
```

Plots are saved in the suite directory.

## Time limits

A job that reaches the time limit resubmits itself. Slurm warns the job 10
minutes before the limit, training stops at the end of the epoch and saves a
restart checkpoint to `<suite>/restarts`, and a new job for the same array task
carries on from there, skipping the task's models that are already trained.
Training also saves the restart checkpoint every `checkpoint_freq` epochs, so a
job that crashes or is cancelled can be carried on by submitting its task again,
from the suite directory:

```bash
sbatch --array=<task> slurm_submit_array.sbatch
```

`train_pinn.sh` also carries on from a restart checkpoint if there is one. The
restart checkpoint is deleted once the model is saved, and
`generate_param_files.sh` deletes any left over along with the old parameter
files.

## Suites

- `fiducial`: the default setup on the 1, 2 and 3 frequency simulations.
- `no_dbdx_bc`: same as `fiducial` but without the `db/dx` boundary condition.
- `nn_size`: number of layers against neurons per layer, 3 frequencies only.
- `col_sampling`, `data_sampling`, `bc_sampling`: number of collocation, data
  and boundary points.
- `col_sampling_schemes`: how the collocation points are placed (Sobol, Latin
  hypercube, aligned grid and offset grid).
- `methods`: number of data points with each method switched off in turn,
  2 frequency case only.

## Adding a suite

1. Make the directory and its `suite.yml`.
2. Copy `slurm_submit_array.sbatch` over from another suite and change the job
   name and `MODELS_PER_JOB`.
