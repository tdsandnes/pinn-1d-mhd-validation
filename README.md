# Validation of Physics-Informed Neural Networks in a One-Dimensional Magnetohydrodynamic Inverse Problem

Dedalus is used for simulations. The PINNs are trained with
[pinninit v0.1.0](https://github.com/tdsandnes/pinninit/tree/v0.1.0), which
should be cloned next to this repo.

## Modules

- `simulations/` runs the Dedalus simulations and samples training, validation
  and test sets from them. Has its own README.
- `equations/` contains equations needed for PINN training.
- `pinn_methods/` defines PINN methods that can be switched on and off.
- `pinn_templates/` holds the pinninit parameter file templates. `default.yml` is
  the standard setup and `no_dbdx_bc.yml` is the same without the `db/dx` term.
- `pinn_suites/` contains the suites of PINN tests to run to produce the results
  of the paper. Has its own README.
- `plotting/` has shared plotting style and utils files.

## Setup

Clone pinninit v0.1.0 next to this repo. From inside this repo:

```bash
git clone --branch v0.1.0 https://github.com/tdsandnes/pinninit.git ../pinninit
```

Two conda environments are needed, one with Dedalus for the simulations and one
with pinninit for everything else. Their names, and where conda lives, are set
in `config.sh`, so change those to match your set up.

For the paper, the simulations were run with Dedalus v3.0.5 (see the
[Dedalus install instructions](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html)).
The pinninit environment is created from `environment.yml` in pinninit v0.1.0.

`config.sh` also puts this repo and pinninit on `PYTHONPATH`, so source it before
running anything:

```bash
source config.sh
```

The shell scripts in `simulations/` and `pinn_suites/` already do this.

## Workflow

1. Run and sample the simulations (see README in `simulations/`).
2. Generate the parameter files for a suite and submit the training (see README in `pinn_suites/`).
3. Run that suite's plotting scripts once the models are trained.

## Development

Linting and type checking are done with ruff and mypy through pre-commit:

```bash
pre-commit install
pre-commit run --all-files
```
