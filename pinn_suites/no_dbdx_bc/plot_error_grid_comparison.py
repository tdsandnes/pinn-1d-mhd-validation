from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.colors import LogNorm

from pinn_suites.no_dbdx_bc.plot_error_grid import (
    ERROR_CMAP,
    REPEAT,
    T_MAX,
    T_MIN,
    draw_error_panel,
    reshape_to_grid,
)
from pinn_suites.suite_defaults import SUITES_DIR
from plotting import style
from plotting.evaluate_test_dataset import (
    load_test_data,
    pointwise_errors,
)
from plotting.plotting_utils import (
    frequency_filter,
    model_int,
    simulation_freq,
    suite_n_repeats,
    suite_values,
)
from simulations.sim_io import sim_data_dir

SUITE_DIR = Path(__file__).resolve().parent

# The frequency case compared
FREQ = 2

# (column title, suite), left to right
COLUMNS: list[tuple[str, str]] = [
    ("With bc loss term", "fiducial"),
    ("Without bc loss term", "no_dbdx_bc"),
]

# Panels are sized as in the 3-column plot_error_grid figure
REFERENCE_COLS = 3


def freq_simulation(suite: str, freq: int) -> str:
    """The simulation a suite runs for the requested frequency."""
    for simulation in suite_values(suite, "simulations"):
        if simulation_freq(str(simulation)) == freq:
            return str(simulation)
    raise RuntimeError(f"Suite {suite} has no freq={freq} simulation")


def suite_grids(
    suite: str,
    freq: int,
    repeat: int,
    n_batches: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One suite's (u, b) error grids for a chosen repeat and averaged over repeats.

    Returns (u_repeat, b_repeat, u_mean, b_mean, x_unique, t_unique).
    """
    param_dir = SUITES_DIR / suite / "param_files"
    model_dir = SUITES_DIR / suite / "models"

    simulation = freq_simulation(suite, freq)
    test_file = sim_data_dir(simulation) / "test.hdf5"

    test = load_test_data(test_file, device, t_min=T_MIN, t_max=T_MAX)

    errors = pointwise_errors(
        param_dir,
        model_dir,
        test,
        n_batches=n_batches,
        model_filter=frequency_filter(freq),
    )

    errors_by_repeat = {
        model_int(name, "repeat"): (u_error, b_error)
        for name, (u_error, b_error) in errors.items()
    }

    if repeat not in errors_by_repeat:
        raise RuntimeError(
            f"{suite} freq={freq}: repeat {repeat} missing "
            f"(have {sorted(errors_by_repeat)})"
        )

    u_repeat, x_unique, t_unique = reshape_to_grid(errors_by_repeat[repeat][0], test)
    b_repeat, _, _ = reshape_to_grid(errors_by_repeat[repeat][1], test)

    u_mean_error = np.mean([u for u, _ in errors_by_repeat.values()], axis=0)
    b_mean_error = np.mean([b for _, b in errors_by_repeat.values()], axis=0)
    u_mean, _, _ = reshape_to_grid(u_mean_error, test)
    b_mean, _, _ = reshape_to_grid(b_mean_error, test)

    n_repeats = suite_n_repeats(suite)
    if len(errors_by_repeat) != n_repeats:
        print(
            f"Warning: {suite} freq={freq} has {len(errors_by_repeat)} repeats, "
            f"suite.yml requests {n_repeats}; mean uses what is trained."
        )

    print(
        f"{suite} freq={freq}: t in [{T_MIN:g}, {T_MAX:g}] covers "
        f"{len(x_unique)} x points x {len(t_unique)} t points"
    )

    return u_repeat, b_repeat, u_mean, b_mean, x_unique, t_unique


def shared_lognorm(
    u_grids: list[np.ndarray],
    b_grids: list[np.ndarray],
) -> LogNorm:
    """One log colour range covering every panel, as in plot_error_grid."""
    # Leave out the u hard-BC column (x = 1), as in plot_error_grid
    values = np.concatenate(
        [grid[:, :-1].ravel() for grid in u_grids] + [grid.ravel() for grid in b_grids]
    )
    positive = values[values > 0]
    return LogNorm(vmin=positive.min(), vmax=positive.max())


def plot_comparison_figure(
    u_grids: list[np.ndarray],
    b_grids: list[np.ndarray],
    column_labels: list[str],
    x_unique: np.ndarray,
    t_unique: np.ndarray,
    cbar_label: str,
    out_path: Path,
) -> None:
    """A 2x2 grid: rows are u and b, columns are with/without the BC."""
    n_cols = len(column_labels)

    hgap = 0.1
    vgap = 0.22
    margins = dict(style.MARGINS, right=0.6, top=0.25)

    available = (
        style.TEXT_WIDTH
        - margins["left"]
        - margins["right"]
        - (REFERENCE_COLS - 1) * hgap
        - style.CBAR_GAP
        - style.CBAR_WIDTH
    )
    panel_width = available / REFERENCE_COLS

    fig, axes, caxes = style.panel_grid(
        nrows=2,
        ncols=n_cols,
        panel_width=panel_width,
        panel_height=panel_width,
        hgap=hgap,
        vgap=vgap,
        margins=margins,
        colorbar="shared",
    )

    norm = shared_lognorm(u_grids, b_grids)

    for j, label in enumerate(column_labels):
        draw_error_panel(
            axes[0][j],
            u_grids[j],
            x_unique,
            t_unique,
            norm,
            show_xlabel=False,
            show_ylabel=(j == 0),
            zero_column=-1,
        )
        draw_error_panel(
            axes[1][j],
            b_grids[j],
            x_unique,
            t_unique,
            norm,
            show_xlabel=True,
            show_ylabel=(j == 0),
        )

        axes[0][j].text(
            0.5,
            1.02,
            label,
            transform=axes[0][j].transAxes,
            ha="center",
            va="bottom",
        )

        style.panel_letter(axes[0][j], style.PANEL_LABELS[j])
        style.panel_letter(axes[1][j], style.PANEL_LABELS[n_cols + j])

        axes[0][j].text(
            1.0, 1.02, r"$u$", transform=axes[0][j].transAxes, ha="right", va="bottom"
        )
        axes[1][j].text(
            1.0, 1.02, r"$b$", transform=axes[1][j].transAxes, ha="right", va="bottom"
        )

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=ERROR_CMAP)
    style.add_colorbar(fig, mappable, caxes[0], label=cbar_label)

    style.save(fig, out_path)


def main() -> None:
    n_batches = 10

    style.use_style()
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    column_labels = [label for label, _ in COLUMNS]

    u_repeat: list[np.ndarray] = []
    b_repeat: list[np.ndarray] = []
    u_mean: list[np.ndarray] = []
    b_mean: list[np.ndarray] = []

    x_unique: np.ndarray | None = None
    t_unique: np.ndarray | None = None

    for _, suite in COLUMNS:
        u_r, b_r, u_m, b_m, x_grid, t_grid = suite_grids(
            suite, FREQ, REPEAT, n_batches, device
        )

        if x_unique is None:
            x_unique, t_unique = x_grid, t_grid
        elif not np.array_equal(x_unique, x_grid) or not np.array_equal(
            t_unique, t_grid
        ):
            raise RuntimeError(
                "Suites do not share a test grid; cannot plot them on common axes"
            )

        u_repeat.append(u_r)
        b_repeat.append(b_r)
        u_mean.append(u_m)
        b_mean.append(b_m)

    assert x_unique is not None and t_unique is not None

    plot_comparison_figure(
        u_repeat,
        b_repeat,
        column_labels,
        x_unique,
        t_unique,
        cbar_label=style.ABSOLUTE_ERROR_LABEL.capitalize(),
        out_path=SUITE_DIR / f"test_error_grid_bc_comparison_repeat{REPEAT:02d}.pdf",
    )

    plot_comparison_figure(
        u_mean,
        b_mean,
        column_labels,
        x_unique,
        t_unique,
        cbar_label=f"Mean {style.ABSOLUTE_ERROR_LABEL}",
        out_path=SUITE_DIR / "test_error_grid_bc_comparison_mean.pdf",
    )


if __name__ == "__main__":
    main()
