from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator
from matplotlib.transforms import offset_copy

from pinn_suites.suite_defaults import SUITE_DEFAULTS, SUITES_DIR
from plotting import style
from plotting.evaluate_test_dataset import (
    TestData,
    load_test_data,
    model_errors,
)
from plotting.plots import draw_error_line, log_ylim
from plotting.plotting_utils import (
    freq_label,
    frequency_filter,
    model_int,
    simulation_freq,
    suite_n_repeats,
    suite_values,
    training_data_points,
)
from simulations.sim_io import sim_data_dir

MEASURE: Literal["mean", "median"] = "mean"

MARKER_SIZE = 12.0

# Upper y-limits: the data and collocation panels share one, the BC panel has its own
SWEEP_YMAX = 3e0
BC_YMAX = 3e-1

# Suite whose per-frequency errors are drawn as reference lines on the BC panel
NO_BC_SUITE = "no_dbdx_bc"
NO_BC_KWARGS: dict[str, Any] = {
    "ls": "--",
    "zorder": 0.25,
}

# Panel letters sit this many points above each panel, clearing the top-axis
# tick labels
PANEL_LETTER_RAISE = 14.0

TOP_TICK_LABEL_SIZE = 7.0

ErrorsByFreq = dict[int, dict[int, list[float]]]


# Each sweep, in panel order: (suite, x value of a model, x-axis label)
SWEEPS: list[tuple[str, Callable[[str], int], str]] = [
    ("data_sampling", training_data_points, style.N_DATA_LABEL),
    (
        "col_sampling",
        lambda name: model_int(name, "collocation_points"),
        style.N_ZPM_LABEL,
    ),
    ("bc_sampling", lambda name: model_int(name, "bc_points"), style.N_BC_LABEL),
]


def suite_errors(
    suite: str,
    x_value: Callable[[str], int],
    tests: dict[str, TestData],
    n_batches: int,
) -> ErrorsByFreq:
    """Collect repeat errors at each sweep value for every frequency case."""
    suite_dir = SUITES_DIR / suite
    param_dir = suite_dir / "param_files"
    model_dir = suite_dir / "models"

    errors_by_freq: ErrorsByFreq = {}

    for simulation in suite_values(suite, "simulations"):
        simulation_name = str(simulation)
        freq = simulation_freq(simulation_name)

        results = model_errors(
            param_dir,
            model_dir,
            tests[simulation_name],
            n_batches=n_batches,
            model_filter=frequency_filter(freq),
        )

        errors_by_freq[freq] = {}

        for name, result in results.items():
            errors_by_freq[freq].setdefault(
                x_value(name),
                [],
            ).append(result.mean_error)

    return errors_by_freq


def suite_measured_errors(
    suite: str,
    tests: dict[str, TestData],
    n_batches: int,
) -> dict[int, float]:
    """Reduce every model's error in a suite to one value per frequency case."""
    reducer = np.mean if MEASURE == "mean" else np.median

    errors_by_freq = suite_errors(
        suite,
        lambda _: 0,
        tests,
        n_batches,
    )

    return {
        freq: float(
            reducer([value for repeats in errors.values() for value in repeats])
        )
        for freq, errors in errors_by_freq.items()
    }


def draw_sweep_panel(
    ax: Axes,
    errors_by_freq: ErrorsByFreq,
    n_repeats: int,
    xlabel: str,
    fiducial: int,
) -> None:
    """Draw every frequency case on one sweep, boxing the fiducial count on top."""
    for freq, errors in sorted(errors_by_freq.items()):
        sweep_points = sorted(errors)

        draw_error_line(
            ax,
            sweep_points,
            [errors[count] for count in sweep_points],
            n_repeats,
            color=style.FREQ_COLORS[freq],
            label=freq_label(freq),
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

    ax.set_xlabel(xlabel)

    add_exact_count_axis(
        ax,
        sorted({n for errors in errors_by_freq.values() for n in errors}),
        fiducial,
    )


def add_exact_count_axis(ax: Axes, counts: list[int], fiducial: int) -> None:
    """Top axis with the exact count written out above each sweep point, and the
    fiducial count boxed."""
    # The top axis gets its own ticks, so drop the bottom axis' mirrored ones
    ax.tick_params(axis="x", which="both", top=False)

    top = ax.secondary_xaxis("top")
    top.xaxis.set_major_locator(FixedLocator(counts))
    top.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    top.xaxis.set_minor_locator(NullLocator())
    # Smaller than the bottom tick labels, so the five-digit counts fit side by side
    top.tick_params(axis="x", labelsize=TOP_TICK_LABEL_SIZE)

    for count, tick in zip(counts, top.xaxis.get_major_ticks(), strict=True):
        if count == fiducial:
            tick.label2.set_bbox(fiducial_box())


def fiducial_box() -> dict[str, Any]:
    """Box around a fiducial count, outlined like the legend frame."""
    linewidth = mpl.rcParams.get("legend.linewidth") or mpl.rcParams["patch.linewidth"]
    return {
        "boxstyle": "square,pad=0.2",
        "facecolor": "none",
        "edgecolor": mpl.rcParams["legend.edgecolor"],
        "linewidth": linewidth,
    }


def panel_letter_above(ax: Axes, letter: str) -> None:
    """Mark a panel (a), (b), ... above its top-axis tick labels, left-aligned."""
    above_ticks = offset_copy(
        ax.transAxes, fig=ax.figure, y=PANEL_LETTER_RAISE, units="points"
    )
    ax.text(0.0, 1.0, f"({letter})", transform=above_ticks, ha="left", va="bottom")


def even_power_label(x: float, _pos: int) -> str:
    """Label a base-2 tick only when its exponent is even."""
    exponent = round(np.log2(x))

    if exponent % 2:
        return ""

    return rf"$\mathdefault{{2^{{{exponent}}}}}$"


def draw_no_bc_lines(
    ax: Axes,
    no_bc_errors: dict[int, float],
) -> None:
    """Mark each frequency case's error without the BC loss."""
    for freq, error in sorted(no_bc_errors.items()):
        ax.axhline(
            error,
            color=style.FREQ_COLORS[freq],
            **NO_BC_KWARGS,
        )


def plot_sweeps_figure(
    all_errors: list[ErrorsByFreq],
    n_repeats: list[int],
    no_bc_errors: dict[int, float],
    out_path: Path,
) -> None:
    """Plot the data, collocation and BC sweeps side by side."""
    n_cols = len(SWEEPS)

    margins = dict(
        style.MARGINS,
        left=0.5,
        right=0.10,
        bottom=0.40,
        top=0.42,
    )

    # The data and collocation panels share a y-axis, so sit close together. The BC
    # panel has its own y tick labels, so it sits further from its neighbour
    gaps = [0.15, 0.36]

    panel_width = (
        style.TEXT_WIDTH - margins["left"] - margins["right"] - sum(gaps)
    ) / n_cols
    panel_height = panel_width / 1.4

    fig_width = style.TEXT_WIDTH
    fig_height = margins["bottom"] + panel_height + margins["top"]

    fig = plt.figure(figsize=(fig_width, fig_height))

    axes: list[Axes] = []
    x0 = margins["left"]

    for j in range(n_cols):
        axes.append(
            fig.add_axes(
                (
                    x0 / fig_width,
                    margins["bottom"] / fig_height,
                    panel_width / fig_width,
                    panel_height / fig_height,
                )
            )
        )

        if j < n_cols - 1:
            x0 += panel_width + gaps[j]

    for j, ((_, x_value, xlabel), errors_by_freq, repeats) in enumerate(
        zip(SWEEPS, all_errors, n_repeats, strict=True)
    ):
        # The fiducial count, read off the default sampling config's name
        fiducial = x_value(str(SUITE_DEFAULTS["sampling_configs"]))

        draw_sweep_panel(
            axes[j],
            errors_by_freq,
            repeats,
            xlabel,
            fiducial,
        )

        panel_letter_above(
            axes[j],
            style.PANEL_LABELS[j],
        )

    # Compare the BC sweep against the suite trained without the BC loss
    draw_no_bc_lines(
        axes[-1],
        no_bc_errors,
    )

    axes[0].legend(loc="upper right")

    # Every panel starts at the same lower y-limit
    y_min, _ = log_ylim(
        [
            value
            for errors_by_freq in all_errors
            for errors in errors_by_freq.values()
            for repeats in errors.values()
            for value in repeats
        ]
        + list(no_bc_errors.values())
    )

    # The data and collocation panels share one y-axis
    ax_data, ax_col, ax_bc = axes

    ax_col.sharey(ax_data)
    ax_data.set_ylim(y_min, SWEEP_YMAX)

    ax_col.tick_params(
        which="both",
        labelleft=False,
    )

    # The BC panel has its own upper limit, so it keeps its own tick labels
    ax_bc.set_ylim(y_min, BC_YMAX)

    # Keep every BC tick but only label the even powers of two
    ax_bc.xaxis.set_major_formatter(FuncFormatter(even_power_label))

    # Keep only the leftmost y-axis label
    ax_data.set_ylabel(rf"{MEASURE.capitalize()} {style.L2_RELATIVE_ERROR_LABEL}")

    style.save(
        fig,
        out_path,
    )


def main() -> None:
    n_batches = 10
    t_min = 21.0
    t_max = 29.0

    out_dir = SUITES_DIR

    style.use_style()
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load each simulation's test set once and reuse it across suites
    tests: dict[str, TestData] = {}

    for suite in [*(suite for suite, _, _ in SWEEPS), NO_BC_SUITE]:
        for simulation in suite_values(suite, "simulations"):
            simulation_name = str(simulation)

            if simulation_name in tests:
                continue

            tests[simulation_name] = load_test_data(
                sim_data_dir(simulation_name) / "test.hdf5",
                device,
                t_min=t_min,
                t_max=t_max,
            )

    all_errors = [
        suite_errors(
            suite,
            x_value,
            tests,
            n_batches,
        )
        for suite, x_value, _ in SWEEPS
    ]

    n_repeats = [suite_n_repeats(suite) for suite, _, _ in SWEEPS]

    no_bc_errors = suite_measured_errors(
        NO_BC_SUITE,
        tests,
        n_batches,
    )

    for freq, error in sorted(no_bc_errors.items()):
        print(f"{NO_BC_SUITE} freq={freq}: {MEASURE} error = {error:.3e}")

    plot_sweeps_figure(
        all_errors,
        n_repeats,
        no_bc_errors,
        out_dir / "test_error_vs_sampling.pdf",
    )


if __name__ == "__main__":
    main()
