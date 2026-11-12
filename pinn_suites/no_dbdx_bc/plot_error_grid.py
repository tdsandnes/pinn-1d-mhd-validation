from pathlib import Path

import colorcet as cc
import matplotlib as mpl
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

from plotting import style
from plotting.evaluate_test_dataset import (
    TestData,
    load_test_data,
    pointwise_errors,
)
from plotting.plotting_utils import (
    freq_label,
    frequency_filter,
    model_int,
    simulation_freq,
    suite_n_repeats,
    suite_values,
)
from simulations.sim_io import sim_data_dir

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name
REPEAT = 1

T_MIN = 20.0
T_MAX = 30.0

# Masked cells (zero error, e.g. the u hard-BC column at x = 1) are drawn black
ERROR_CMAP = cc.cm.kbgyw.copy()
ERROR_CMAP.set_bad("black")

# Lines t = 21 - x and t = 29 + x, as (t at x = 0, slope dt/dx).
ALFVEN_SPEED_LINES = ((21.0, -1.0), (29.0, 1.0))
ALFVEN_SPEED_KWARGS = {"color": "black", "ls": "--"}


def reshape_to_grid(
    values: np.ndarray,
    test: TestData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape per-point values onto the test set's (t, x) grid."""
    t = test.coords.get_coord("t").detach().cpu().numpy()
    x = test.coords.get_coord("x").detach().cpu().numpy()

    t_unique = np.unique(t)
    x_unique = np.unique(x)

    n_t = len(t_unique)
    n_x = len(x_unique)

    if len(values) != n_t * n_x:
        raise RuntimeError(
            f"Test set has {len(values)} points but the unique (t, x) grid "
            f"is {n_t} x {n_x} = {n_t * n_x}; cannot reshape into a heatmap"
        )

    order = np.lexsort((x, t))
    grid = values[order].reshape(n_t, n_x)

    return grid, x_unique, t_unique


def draw_error_panel(
    ax: Axes,
    grid: np.ndarray,
    x_unique: np.ndarray,
    t_unique: np.ndarray,
    norm: LogNorm,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    zero_column: int | None = None,
    alfven_lines: bool = True,
) -> mpl.image.AxesImage:
    """Draw one absolute-error heatmap."""
    masked_grid = np.ma.masked_equal(grid, 0.0)
    if zero_column is not None:
        masked_grid[:, zero_column] = np.ma.masked

    image = ax.imshow(
        masked_grid,
        origin="lower",
        extent=(
            x_unique.min(),
            x_unique.max(),
            t_unique.min(),
            t_unique.max(),
        ),
        norm=norm,
        cmap=ERROR_CMAP,
        aspect="auto",
        interpolation="nearest",
    )

    if alfven_lines:
        x_line = np.array(ax.get_xlim())
        for t_at_x0, slope in ALFVEN_SPEED_LINES:
            ax.plot(
                x_line,
                t_at_x0 + slope * x_line,
                scalex=False,
                scaley=False,
                **ALFVEN_SPEED_KWARGS,
            )

    ax.minorticks_off()

    # Exclude end tick labels
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda v, pos: (
                ""
                if np.isclose(v, ax.get_xlim()[0]) or np.isclose(v, ax.get_xlim()[1])
                else f"{v:g}"
            )
        )
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, pos: (
                ""
                if np.isclose(v, ax.get_ylim()[0]) or np.isclose(v, ax.get_ylim()[1])
                else f"{v:g}"
            )
        )
    )

    if show_xlabel:
        ax.set_xlabel(r"$x$")
    else:
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(r"$t$")
    else:
        ax.tick_params(labelleft=False)

    return image


def plot_error_grid_figure(
    u_grids_by_freq: dict[int, np.ndarray],
    b_grids_by_freq: dict[int, np.ndarray],
    x_unique: np.ndarray,
    t_unique: np.ndarray,
    cbar_label: str,
    out_path: Path,
) -> None:
    """Plot u and b absolute-error grids for all frequency cases."""
    freqs = sorted(u_grids_by_freq)
    n_cols = len(freqs)

    hgap = 0.1
    vgap = 0.22
    margins = dict(style.MARGINS, right=0.6, top=0.25)

    available = (
        style.TEXT_WIDTH
        - margins["left"]
        - margins["right"]
        - (n_cols - 1) * hgap
        - style.CBAR_GAP
        - style.CBAR_WIDTH
    )
    panel_width = available / n_cols

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

    # Leave the u hard-BC column (x = 1) out of the colour range, so round-off
    # there cannot pull the log floor down to ~1e-15
    all_values = np.concatenate(
        [grid[:, :-1].ravel() for grid in u_grids_by_freq.values()]
        + [grid.ravel() for grid in b_grids_by_freq.values()]
    )
    positive_values = all_values[all_values > 0]
    norm = LogNorm(vmin=positive_values.min(), vmax=positive_values.max())

    for j, freq in enumerate(freqs):
        draw_error_panel(
            axes[0][j],
            u_grids_by_freq[freq],
            x_unique,
            t_unique,
            norm,
            show_xlabel=False,
            show_ylabel=(j == 0),
            zero_column=-1,
        )

        draw_error_panel(
            axes[1][j],
            b_grids_by_freq[freq],
            x_unique,
            t_unique,
            norm,
            show_xlabel=True,
            show_ylabel=(j == 0),
        )

        axes[0][j].text(
            0.5,
            1.02,
            freq_label(freq),
            transform=axes[0][j].transAxes,
            ha="center",
            va="bottom",
        )

        style.panel_letter(axes[0][j], style.PANEL_LABELS[j])
        style.panel_letter(axes[1][j], style.PANEL_LABELS[n_cols + j])

        axes[0][j].text(
            1.0,
            1.02,
            r"$u$",
            transform=axes[0][j].transAxes,
            ha="right",
            va="bottom",
        )
        axes[1][j].text(
            1.0,
            1.02,
            r"$b$",
            transform=axes[1][j].transAxes,
            ha="right",
            va="bottom",
        )

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=ERROR_CMAP)
    style.add_colorbar(
        fig,
        mappable,
        caxes[0],
        label=cbar_label,
    )

    style.save(fig, out_path)


def main() -> None:
    n_batches = 10

    param_dir = SUITE_DIR / "param_files"
    model_dir = SUITE_DIR / "models"
    simulations = suite_values(SUITE, "simulations")
    n_repeats = suite_n_repeats(SUITE)

    style.use_style()
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    u_repeat: dict[int, np.ndarray] = {}
    b_repeat: dict[int, np.ndarray] = {}
    u_mean: dict[int, np.ndarray] = {}
    b_mean: dict[int, np.ndarray] = {}

    x_unique: np.ndarray | None = None
    t_unique: np.ndarray | None = None

    for simulation in simulations:
        simulation = str(simulation)
        freq = simulation_freq(simulation)

        test_file = sim_data_dir(simulation) / "test.hdf5"

        test = load_test_data(
            test_file,
            device,
            t_min=T_MIN,
            t_max=T_MAX,
        )

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

        if REPEAT not in errors_by_repeat:
            raise RuntimeError(
                f"freq={freq}: repeat {REPEAT} missing "
                f"(have {sorted(errors_by_repeat)})"
            )

        u_grid, x_grid, t_grid = reshape_to_grid(errors_by_repeat[REPEAT][0], test)
        b_grid, _, _ = reshape_to_grid(errors_by_repeat[REPEAT][1], test)

        if x_unique is None:
            x_unique = x_grid
            t_unique = t_grid
        elif not np.array_equal(x_unique, x_grid) or not np.array_equal(
            t_unique, t_grid
        ):
            raise RuntimeError(
                "Test grids differ between frequency cases; "
                "cannot plot them on common axes"
            )

        u_repeat[freq] = u_grid
        b_repeat[freq] = b_grid

        u_mean_error = np.mean(
            [u_error for u_error, _ in errors_by_repeat.values()],
            axis=0,
        )
        b_mean_error = np.mean(
            [b_error for _, b_error in errors_by_repeat.values()],
            axis=0,
        )

        u_mean[freq], _, _ = reshape_to_grid(u_mean_error, test)
        b_mean[freq], _, _ = reshape_to_grid(b_mean_error, test)

        print(
            f"freq={freq}: t in [{T_MIN:g}, {T_MAX:g}] covers "
            f"{len(x_grid)} x points x {len(t_grid)} t points "
            f"= {len(x_grid) * len(t_grid)} test points"
        )

        if len(errors_by_repeat) != n_repeats:
            print(
                f"Warning: freq={freq} has {len(errors_by_repeat)} repeats, "
                f"suite.yml requests {n_repeats}; "
                "mean uses what is trained."
            )

    assert x_unique is not None and t_unique is not None

    plot_error_grid_figure(
        u_repeat,
        b_repeat,
        x_unique,
        t_unique,
        cbar_label=style.ABSOLUTE_ERROR_LABEL.capitalize(),
        out_path=SUITE_DIR / f"test_error_grid_repeat{REPEAT:02d}.pdf",
    )

    plot_error_grid_figure(
        u_mean,
        b_mean,
        x_unique,
        t_unique,
        cbar_label=f"Mean {style.ABSOLUTE_ERROR_LABEL}",
        out_path=SUITE_DIR / "test_error_grid_mean.pdf",
    )


if __name__ == "__main__":
    main()
