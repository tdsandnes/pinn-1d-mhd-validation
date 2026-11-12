from pathlib import Path
from typing import Literal

import torch
from matplotlib.ticker import LogLocator

from plotting import style
from plotting.evaluate_test_dataset import load_test_data, model_errors
from plotting.plots import draw_error_line
from plotting.plotting_utils import (
    freq_label,
    frequency_filter,
    simulation_freq,
    suite_n_repeats,
    suite_values,
    training_data_points,
)
from simulations.sim_io import sim_data_dir

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name

MEASURE: Literal["mean", "median"] = "mean"

MARKER_SIZE = 12.0


def plot_sweep_figure(
    errors_by_freq: dict[int, dict[int, list[float]]],
    n_repeats: int,
    out_path: Path,
) -> None:
    """Test error against the data-point count, one line per frequency case."""
    # Room below for the subscript in the x-axis label
    margins = dict(style.MARGINS, left=0.5, right=0.10, bottom=0.40)
    panel_width = 0.5 * style.TEXT_WIDTH - margins["left"] - margins["right"]

    fig, axes, _ = style.panel_grid(
        panel_width=panel_width,
        panel_height=panel_width / 1.4,
        margins=margins,
    )
    ax = axes[0][0]

    for freq, errors in sorted(errors_by_freq.items()):
        counts = sorted(errors)
        draw_error_line(
            ax,
            counts,
            [errors[count] for count in counts],
            n_repeats,
            color=style.FREQ_COLORS[freq],
            label=freq_label(freq),
            measure=MEASURE,
            show_range=n_repeats > 1,
            marker_size=MARKER_SIZE,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=100))
    ax.set_yscale("log")

    ax.set_xlabel(style.N_DATA_LABEL)
    ax.set_ylabel(rf"{MEASURE.capitalize()} {style.L2_RELATIVE_ERROR_LABEL}")
    ax.legend()

    style.save(fig, out_path)


def main() -> None:
    n_batches = 10
    t_min = 21.0
    t_max = 29.0

    param_dir = SUITE_DIR / "param_files"
    model_dir = SUITE_DIR / "models"

    n_repeats = suite_n_repeats(SUITE)

    style.use_style()
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    errors_by_freq: dict[int, dict[int, list[float]]] = {}
    for simulation in suite_values(SUITE, "simulations"):
        simulation_name = str(simulation)
        freq = simulation_freq(simulation_name)

        test = load_test_data(
            sim_data_dir(simulation_name) / "test.hdf5",
            device,
            t_min=t_min,
            t_max=t_max,
        )
        results = model_errors(
            param_dir,
            model_dir,
            test,
            n_batches=n_batches,
            model_filter=frequency_filter(freq),
        )

        errors_by_freq[freq] = {}
        for name, result in results.items():
            count = training_data_points(name)
            errors_by_freq[freq].setdefault(count, []).append(result.mean_error)

    plot_sweep_figure(errors_by_freq, n_repeats, SUITE_DIR / "test_error_vs_data.pdf")


if __name__ == "__main__":
    main()
