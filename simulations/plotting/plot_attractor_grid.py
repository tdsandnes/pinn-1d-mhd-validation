from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.ticker import FixedLocator

from plotting import style
from simulations.plotting.simulation_plotting_utils import (
    ATTRACTOR_XLIM,
    ATTRACTOR_YLIM,
    B_LABEL,
    SIM_CMAP,
    ZM_LABEL,
    find_sim_config,
    load_attractor,
    plot_attractor,
)
from simulations.sim_io import SnapshotReader, load_sim_config

PLOTTING_DIR = Path(__file__).resolve().parent

NROWS = 2
NCOLS = 3

# Runs in row order: nonlinear row then linear row
RUNS = (
    "1freq_1Ma_500Re_500Rm",
    "2freq_1Ma_500Re_500Rm",
    "3freq_1Ma_500Re_500Rm",
    "1freq_1Ma_500Re_500Rm_nononlin",
    "2freq_1Ma_500Re_500Rm_nononlin",
    "3freq_1Ma_500Re_500Rm_nononlin",
)
COLUMN_TITLES = ("1 frequency", "2 frequencies", "3 frequencies")
ROW_TITLES = ("Nonlinear", "Linear")


def main() -> None:
    t_min = 0.0
    t_max: float | None = 30.0  # None is each run's last snapshot time

    if len(RUNS) > NROWS * NCOLS:
        raise ValueError(f"Got {len(RUNS)} runs, grid holds {NROWS * NCOLS}.")

    style.use_style()

    # Fill the text width with panels plus a shared colorbar
    margins = dict(style.MARGINS, right=0.40, bottom=0.35, top=0.25)
    hgap = 0.1
    vgap = 0.1
    panel_width = (
        style.TEXT_WIDTH
        - margins["left"]
        - margins["right"]
        - (NCOLS - 1) * hgap
        - style.CBAR_GAP
        - style.CBAR_WIDTH
    ) / NCOLS
    panel_height = (
        panel_width
        * (ATTRACTOR_YLIM[1] - ATTRACTOR_YLIM[0])
        / (ATTRACTOR_XLIM[1] - ATTRACTOR_XLIM[0])
    )

    fig, axes, caxes = style.panel_grid(
        NROWS,
        NCOLS,
        panel_width=panel_width,
        panel_height=panel_height,
        hgap=hgap,
        vgap=vgap,
        margins=margins,
        colorbar="shared",
    )

    # Load every run first so the colour scale can span them all
    runs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for run in RUNS:
        config_path = find_sim_config(run)
        reader = SnapshotReader(config_path)
        config = load_sim_config(config_path)
        t, zm_drive, b_edge = load_attractor(reader, config)
        runs.append((run, t, zm_drive, b_edge))
        print(f"Loaded {run}")

    global_t_max = t_max if t_max is not None else max(t[-1] for _, t, _, _ in runs)
    norm = mpl.colors.Normalize(vmin=t_min, vmax=global_t_max)

    for i, (run, t, zm_drive, b_edge) in enumerate(runs):
        print(f"Plotting {run}")
        row, col = divmod(i, NCOLS)
        ax = axes[row][col]

        t_end = t_max if t_max is not None else float(t[-1])
        plot_attractor(ax, t, zm_drive, b_edge, t_min, t_end, norm=norm)

        # Axis labels on the outer edges only
        if col == 0:
            ax.set_ylabel(B_LABEL)
            ax.yaxis.labelpad = -3.0
        else:
            ax.tick_params(labelleft=False)
        if row == NROWS - 1:
            ax.set_xlabel(ZM_LABEL)
            ax.xaxis.labelpad = 1.0
        else:
            ax.tick_params(labelbottom=False)

        ax.xaxis.set_major_locator(FixedLocator([-1.0, 0.0, 1.0]))
        ax.yaxis.set_major_locator(FixedLocator([-1.0, 0.0, 1.0]))

        # Column titles above the top row, row titles inside the first column
        if row == 0:
            ax.text(
                0.5,
                1.02,
                COLUMN_TITLES[col],
                transform=ax.transAxes,
                ha="center",
                va="bottom",
            )
        if col == 0:
            ax.text(
                0.05,
                0.05,
                ROW_TITLES[row],
                transform=ax.transAxes,
                ha="left",
                va="bottom",
            )
        style.panel_letter(ax, style.PANEL_LABELS[i], inside=True)

    style.add_colorbar(
        fig, mpl.cm.ScalarMappable(norm=norm, cmap=SIM_CMAP), caxes[0], label=r"$t$"
    )
    style.save(fig, PLOTTING_DIR / "attractor_grid.pdf")


if __name__ == "__main__":
    main()
