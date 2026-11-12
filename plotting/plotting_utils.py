import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pinninit import PINNConfig
from pinninit.geometry.coordinate_system import CoordSystemLabels
from pinninit.nn.neural_network import NNTypeLabels

from pinn_suites.suite_defaults import SUITE_DEFAULTS, SUITES_DIR, TAG_PATTERNS
from simulations.sampling_configs.sampling_dataclasses import DataDatasets
from simulations.sampling_configs.template import Config as SamplingConfigTemplate
from simulations.sim_io import load_sim_config

DATA_VAL_FRACTION: float = SamplingConfigTemplate().data_val_fraction


def suite_values(suite: str, key: str) -> list[Any]:
    """Every value a suite sweeps for one quantity."""
    if key not in SUITE_DEFAULTS:
        raise KeyError(
            f"'{key}' is not a suite quantity. Use one of {', '.join(SUITE_DEFAULTS)}"
        )
    suite_file = SUITES_DIR / suite / "suite.yml"
    if not suite_file.is_file():
        raise SystemExit(f"No such suite: {suite_file}")
    settings: dict[str, Any] = yaml.safe_load(suite_file.read_text())
    value: Any = settings.get(key, SUITE_DEFAULTS[key])
    return list(value) if isinstance(value, list) else [value]


def suite_n_repeats(suite: str) -> int:
    """How many repeats a suite runs per sweep combination."""
    suite_file = SUITES_DIR / suite / "suite.yml"
    if not suite_file.is_file():
        raise SystemExit(f"No such suite: {suite_file}")
    settings: dict[str, Any] = yaml.safe_load(suite_file.read_text())
    value = settings.get("repeats", SUITE_DEFAULTS["repeats"])
    return len(value) if isinstance(value, list) else int(value)


def simulation_freq(simulation: str) -> int:
    """How many forcing frequencies a simulation config uses."""
    return len(load_sim_config(simulation).FREQS)


def model_int(model_name: str, key: str) -> int:
    """One quantity read back out of a model name, e.g. layers0006 -> 6."""
    if key not in TAG_PATTERNS:
        raise KeyError(
            f"'{key}' has no model name tag. Use one of {', '.join(TAG_PATTERNS)}"
        )
    match = re.search(TAG_PATTERNS[key], model_name)
    if match is None:
        raise ValueError(f"No {key} tag in model name {model_name!r}")
    return int(match.group(1))


def frequency_filter(freq: int) -> Callable[[str], bool]:
    """A model-name filter keeping the models trained on one frequency case."""
    return lambda name: model_int(name, "freq") == freq


def freq_label(freq: int) -> str:
    """A frequency case's legend label."""
    return f"{freq} frequency" if freq == 1 else f"{freq} frequencies"


def training_data_points(model_name: str) -> int:
    """Number of training data points a model used, excluding validation."""
    total_data_points = model_int(model_name, "total_data_points")
    return DataDatasets(
        label="data_loss",
        x_bounds=(1.0, 1.0),
        t_bounds=(0.0, 1.0),
        N_x=1,
        N_data=total_data_points,
        val_fraction=DATA_VAL_FRACTION,
    ).N_train


def n_parameters_from_yaml(yaml_path: Path) -> int:
    """The parameter count of the network a parameter file describes."""
    config = PINNConfig.from_yaml(yaml_path)

    if config.nn.nn_type != NNTypeLabels.FEEDFORWARD:
        raise NotImplementedError(
            f"Only '{NNTypeLabels.FEEDFORWARD.value}' networks are supported, "
            f"got '{config.nn.nn_type.value}'."
        )
    if len(config.coord_systems) != 1:
        raise NotImplementedError(
            "n_parameters_from_yaml does not support multiple coords systems yet."
        )
    if config.nn.weight_norm:
        raise NotImplementedError(
            "n_parameters_from_yaml does not support weight_norm=true yet."
        )

    # Only Cartesian keeps the internal geometric coords equal to the geometric ones
    coord_system = config.coord_systems[0]
    if coord_system.coord_system != CoordSystemLabels.CARTESIAN:
        raise NotImplementedError(
            f"Only '{CoordSystemLabels.CARTESIAN.value}' coords are supported, "
            f"got '{coord_system.coord_system.value}'."
        )

    if any(coordinates.domain.periodic_axes for coordinates in config.coordinates):
        raise NotImplementedError(
            "n_parameters_from_yaml does not support periodic axes yet."
        )

    # Input width: one per internal coord (Cartesian keeps internal == geometric)
    n_input = len(coord_system.geometric_axis_labels) + len(
        coord_system.non_geometric_axis_labels
    )

    # Output width: one per component of every output field
    dim_geometric = len(coord_system.geometric_axis_labels)
    n_output = sum(
        dim_geometric**output_field.rank for output_field in config.fields.output_fields
    )

    # Get the number of parameters from the layer widths
    widths = [n_input, *config.nn.hidden_layer_widths, n_output]
    n_parameters: int = sum(
        widths[i] * widths[i + 1] + widths[i + 1] for i in range(len(widths) - 1)
    )
    # Learnable constants are registered on the model as scalar parameters
    n_parameters += sum(1 for const in config.constants if const.learnable)

    return n_parameters
