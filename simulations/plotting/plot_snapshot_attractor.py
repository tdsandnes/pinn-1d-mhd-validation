from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator

from plotting import style
from simulations.plotting.simulation_plotting_utils import (
    ATTRACTOR_XLIM,
    ATTRACTOR_YLIM,
    B_LABEL,
    COLOR_B,
    COLOR_ZM,
    SIM_CMAP,
    ZM_LABEL,
    Snapshots,
    find_sim_config,
    load_snapshots,
    plot_attractor,
)
from simulations.sim_io import SnapshotReader, load_sim_config

PLOTTING_DIR = Path(__file__).resolve().parent

# Field panel geometry in inches
FIELD_ASPECT = 1.8  # panel width / height
HGAP = 0.4  # between the field columns
VGAP = 0.1  # between the field rows
ATT_GAP = 0.4  # between the field block and the attractor
FIELD_YLIM = (-1.8, 1.8)

# Field panels, laid out as they appear
FIELD_PANELS = (
    (("u", r"$u$"), ("b", r"$b$")),
    (("zm", r"$z^-$"), ("zp", r"$z^+$")),
)


def make_axes() -> tuple[Figure, list[list[Axes]], Axes]:
    """A 2x2 block of field panels beside a full-height attractor panel."""
    margins = dict(style.MARGINS, right=0.05, bottom=0.35)
    att_aspect = (ATTRACTOR_YLIM[1] - ATTRACTOR_YLIM[0]) / (
        ATTRACTOR_XLIM[1] - ATTRACTOR_XLIM[0]
    )

    panels_width = (
        style.TEXT_WIDTH - HGAP - ATT_GAP - margins["left"] - margins["right"]
    )
    field_width = (
        FIELD_ASPECT
        * (att_aspect * panels_width - VGAP)
        / (2 * (1 + FIELD_ASPECT * att_aspect))
    )
    att_width = panels_width - 2 * field_width
    fig_height = att_width * att_aspect
    field_height = (fig_height - VGAP) / 2

    # Columns: field, gap, field, big gap, attractor. Rows: field, gap, field.
    width_ratios = [field_width, HGAP, field_width, ATT_GAP, att_width]
    height_ratios = [field_height, VGAP, field_height]
    fig_w = sum(width_ratios) + margins["left"] + margins["right"]
    fig_h = sum(height_ratios) + margins["bottom"] + margins["top"]

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        len(height_ratios),
        len(width_ratios),
        figure=fig,
        wspace=0.0,
        hspace=0.0,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        left=margins["left"] / fig_w,
        right=1.0 - margins["right"] / fig_w,
        bottom=margins["bottom"] / fig_h,
        top=1.0 - margins["top"] / fig_h,
    )
    field_axes = [
        [fig.add_subplot(gs[2 * row, 2 * col]) for col in range(2)] for row in range(2)
    ]
    att_ax = fig.add_subplot(gs[:, 4])
    return fig, field_axes, att_ax


def _plot_field_panel(
    ax: Axes,
    snap: Snapshots,
    name: str,
    i_now: int,
    norm: mpl.colors.Normalize,
    history_lines: int,
    history_dt: float,
    history_alpha_min: float,
    history_alpha_max: float,
) -> None:
    """One field profile at the current time, trailed by faded past profiles."""
    profile = snap.fields[name]
    t_now = snap.t[i_now]

    # Older profiles first, fading in as they approach the current time
    for k in range(history_lines, 0, -1):
        t_hist = t_now - k * history_dt
        if t_hist < snap.t[0]:
            continue
        i_hist = int(np.argmin(np.abs(snap.t - t_hist)))
        frac = (history_lines - k) / max(history_lines - 1, 1)
        alpha = history_alpha_min + frac * (history_alpha_max - history_alpha_min)
        ax.plot(
            snap.x, profile[i_hist], color=SIM_CMAP(norm(snap.t[i_hist])), alpha=alpha
        )

    ax.plot(snap.x, profile[i_now], color=SIM_CMAP(norm(t_now)))


def _draw_attractor_panel(
    fig: Figure,
    ax: Axes,
    snap: Snapshots,
    i_now: int,
    t_min: float,
    t_max: float,
    norm: mpl.colors.Normalize,
) -> None:
    """The attractor up to now, with crosshairs marking the current state."""
    plot_attractor(ax, snap.t, snap.zm_drive, snap.b_edge, t_min, t_max, norm=norm)

    ax.set_xlabel(ZM_LABEL)
    ax.set_ylabel(B_LABEL)
    ax.xaxis.labelpad = 1.0
    ax.yaxis.labelpad = -3.0
    ax.xaxis.set_major_locator(FixedLocator([-1.0, 0.0, 1.0]))
    ax.yaxis.set_major_locator(FixedLocator([-1.0, 0.0, 1.0]))
    style.panel_letter(ax, style.PANEL_LABELS[4], inside=True)

    # Crosshairs from the axes onto the current (drive, wall field) point
    x_now = snap.zm_drive[i_now]
    y_now = snap.b_edge[i_now]
    ax.vlines(x_now, ax.get_ylim()[0], y_now, color=COLOR_ZM, linestyle="--", zorder=10)
    ax.hlines(y_now, ax.get_xlim()[0], x_now, color=COLOR_B, linestyle="--", zorder=10)

    _add_progress_colorbar(fig, ax, float(snap.t[i_now]), norm)


def _add_progress_colorbar(
    fig: Figure, ax: Axes, t_now: float, norm: mpl.colors.Normalize, n: int = 256
) -> None:
    """A small time colorbar acting as a progress bar: the future is blanked."""
    cax = ax.inset_axes([0.05, 0.05, 0.20, 0.04])
    cb_t = np.linspace(norm.vmin, norm.vmax, n)
    colors = SIM_CMAP(np.linspace(0.0, 1.0, n))
    colors[cb_t > t_now, 3] = 0.0  # hide times the run has not reached yet
    cmap = mpl.colors.ListedColormap(colors)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_label(r"$t$", fontsize=8)
    cbar.ax.tick_params(
        which="both",
        direction="out",
        labelsize=6,
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
    )
    cbar.ax.tick_params(which="minor", length=1)
    cbar.ax.tick_params(which="major", length=2, pad=-0.5)
    cbar.ax.xaxis.labelpad = -2.0
    cbar.ax.xaxis.set_label_position("top")


def draw_frame(
    snap: Snapshots,
    t_min: float,
    t_max: float,
    norm: mpl.colors.Normalize,
    *,
    history_lines: int = 0,
    history_dt: float = 1.0,
    history_alpha_min: float = 0.2,
    history_alpha_max: float = 0.7,
) -> Figure:
    """Draw the field panels and attractor at the current time."""
    i_now = int(np.argmin(np.abs(snap.t - t_max)))
    fig, field_axes, att_ax = make_axes()

    for row in range(2):
        for col in range(2):
            ax = field_axes[row][col]
            name, label = FIELD_PANELS[row][col]

            _plot_field_panel(
                ax,
                snap,
                name,
                i_now,
                norm,
                history_lines,
                history_dt,
                history_alpha_min,
                history_alpha_max,
            )

            # Mark the driven boundary and the tracked wall field
            if name == "zm":
                ax.plot(
                    0.0,
                    snap.zm_drive[i_now],
                    "o",
                    color=COLOR_ZM,
                    markersize=2,
                    zorder=10,
                    clip_on=False,
                )
            elif name == "b":
                ax.plot(
                    1.0,
                    snap.b_edge[i_now],
                    "o",
                    color=COLOR_B,
                    markersize=2,
                    zorder=10,
                    clip_on=False,
                )

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(FIELD_YLIM)
            ax.set_ylabel(label)
            ax.yaxis.labelpad = -3.0
            style.panel_letter(ax, style.PANEL_LABELS[row * 2 + col], inside=True)

            if row == 1:
                ax.set_xlabel(r"$x$")
                ax.xaxis.labelpad = 1.0
            else:
                ax.tick_params(labelbottom=False)

            ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
            ax.yaxis.set_major_locator(FixedLocator([-1.0, 0.0, 1.0]))

    _draw_attractor_panel(fig, att_ax, snap, i_now, t_min, t_max, norm)
    return fig


def main() -> None:
    run = "2freq_1Ma_500Re_500Rm"
    t_min = 20.0
    t_max = 30.0

    style.use_style()
    config_path = find_sim_config(run)
    reader = SnapshotReader(config_path)
    config = load_sim_config(config_path)
    snap = load_snapshots(reader, config)

    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    fig = draw_frame(
        snap,
        t_min,
        t_max,
        norm,
        history_lines=3,
        history_dt=0.05,
        history_alpha_min=0.05,
        history_alpha_max=0.20,
    )
    style.save(fig, PLOTTING_DIR / f"snapshot_attractor_{run}.pdf")


if __name__ == "__main__":
    main()
