from pathlib import Path
from typing import Literal

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

from pinn_methods.method_settings import method_str
from plotting import style
from plotting.evaluate_test_dataset import (
    load_test_data,
    model_errors,
)
from plotting.plots import draw_error_line
from plotting.plotting_utils import (
    model_int,
    suite_n_repeats,
    suite_values,
    training_data_points,
)
from simulations.sim_io import sim_data_dir

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name

MEASURE: Literal["mean", "median"] = "mean"

MARKER_SIZE = 12.0

PER_TIME_LABEL = rf"${style.N_DATA}$ per Alfvén crossing time"

# (methods, legend label) in legend order, each matching a methods entry in suite.yml
METHOD_CASES: list[tuple[list[str], str]] = [
    ([], "Basic"),
    (["rba", "w23", "siren"], "No SOAP"),
    (["rba", "w23", "soap"], "No SIREN"),
    (["rba", "siren", "soap"], "No dynamic loss-term weighting"),
    (["w23", "siren", "soap"], "No RBA weighting"),
    (["rba", "w23", "siren", "soap"], "All methods"),
]

METHOD_COLORS = ["#929292", "#5fae8c", "#7abee6", "#dfa03c", "#7b67cc", "#d1495b"]


def method_cases() -> list[tuple[str, str, str]]:
    """(method id, label, colour) of each case, in legend order."""
    return [
        (method_str(methods), label, color)
        for (methods, label), color in zip(METHOD_CASES, METHOD_COLORS, strict=True)
    ]


def training_time_span(model_names: list[str]) -> float:
    """Length of the training time window, shared by every model in the suite."""
    spans = {
        model_int(name, "t_max") - model_int(name, "t_min") for name in model_names
    }
    if len(spans) != 1:
        raise RuntimeError(
            f"{SUITE} models are expected to share one training time window, "
            f"got spans {sorted(spans)}."
        )
    return float(spans.pop())


def add_per_time_axis(ax: Axes, time_span: float, data_points: list[int]) -> None:
    """Top axis of data points per Alfven crossing time, ticked at each point."""

    def to_per_time(n: np.ndarray) -> np.ndarray:
        return n / time_span

    def from_per_time(rate: np.ndarray) -> np.ndarray:
        return rate * time_span

    # The top axis gets its own ticks, so drop the bottom axis' mirrored ones
    ax.tick_params(axis="x", which="both", top=False)

    top = ax.secondary_xaxis("top", functions=(to_per_time, from_per_time))
    # One tick per data point, written out in full, and no minor ticks
    top.xaxis.set_major_locator(FixedLocator([n / time_span for n in data_points]))
    top.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel(PER_TIME_LABEL, labelpad=6.5)


def draw_sweep_panel(
    ax: Axes,
    errors_by_method: dict[str, dict[int, list[float]]],
    n_repeats: int,
    time_span: float,
) -> None:
    """Draw every method case on one data-point sweep."""
    all_data_points = sorted(
        {n for errors in errors_by_method.values() for n in errors}
    )

    for method_id, label, color in method_cases():
        errors = errors_by_method.get(method_id)
        if not errors:
            print(f"Warning: no models for method={method_id} ({label})")
            continue

        data_points = sorted(errors)

        draw_error_line(
            ax,
            data_points,
            [errors[count] for count in data_points],
            n_repeats,
            color=color,
            label=label,
            measure=MEASURE,
            show_range=n_repeats > 1,
            marker_size=MARKER_SIZE,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(
        LogLocator(
            base=2,
            numticks=100,
        )
    )

    ax.set_yscale("log")

    add_per_time_axis(ax, time_span, all_data_points)

    ax.set_xlabel(style.N_DATA_LABEL)
    ax.set_ylabel(rf"{MEASURE.capitalize()} {style.L2_RELATIVE_ERROR_LABEL}")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )


def plot_sweep_figure(
    errors_by_method: dict[str, dict[int, list[float]]],
    n_repeats: int,
    time_span: float,
    out_path: Path,
) -> None:
    """Plot the data-point sweep on its own, with the legend outside."""
    left, right = 0.5, 0.10
    legend_width = 1.45
    # Extra room at the top for the per-crossing-time axis label
    top = 0.48
    margins = dict(
        style.MARGINS, left=left, right=right + legend_width, top=top, bottom=0.38
    )
    panel_width = 0.5 * style.TEXT_WIDTH - left - right
    panel_height = panel_width / 1.4

    fig, axes, _ = style.panel_grid(
        panel_width=panel_width,
        panel_height=panel_height,
        margins=margins,
    )

    draw_sweep_panel(
        axes[0][0],
        errors_by_method,
        n_repeats,
        time_span,
    )

    style.save(
        fig,
        out_path,
    )


def main() -> None:
    n_batches = 10
    t_min = 21.0
    t_max = 29.0

    out_dir = SUITE_DIR
    param_dir = SUITE_DIR / "param_files"
    model_dir = SUITE_DIR / "models"

    simulations = suite_values(
        SUITE,
        "simulations",
    )
    if len(simulations) != 1:
        raise RuntimeError(
            f"{SUITE} is expected to have exactly one simulation, "
            f"got {len(simulations)}."
        )
    simulation_name = str(simulations[0])

    n_repeats = suite_n_repeats(SUITE)

    style.use_style()
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_file = sim_data_dir(simulation_name) / "test.hdf5"
    test = load_test_data(
        test_file,
        device,
        t_min=t_min,
        t_max=t_max,
    )

    results = model_errors(
        param_dir,
        model_dir,
        test,
        n_batches=n_batches,
    )

    errors_by_method: dict[str, dict[int, list[float]]] = {}
    for name, result in results.items():
        method_id = f"{model_int(name, 'method'):05d}"
        data_points = training_data_points(name)

        errors_by_method.setdefault(method_id, {}).setdefault(
            data_points,
            [],
        ).append(result.mean_error)

    plot_sweep_figure(
        errors_by_method,
        n_repeats,
        training_time_span(list(results)),
        out_dir / "test_error_vs_data_methods.pdf",
    )


if __name__ == "__main__":
    main()
