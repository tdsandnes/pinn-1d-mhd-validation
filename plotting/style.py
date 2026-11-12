import string
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

STYLE_FILE = Path(__file__).resolve().parent / "paper.mplstyle"

# The paper's LaTeX \textwidth, in inches
TEXT_WIDTH = 6.475

# Panel letters, in the order the panels are drawn
PANEL_LABELS = string.ascii_lowercase

# Room for axis labels and tick labels, in inches
MARGINS = {"left": 0.35, "right": 0.10, "bottom": 0.35, "top": 0.10}

# Colorbar geometry, in inches
CBAR_GAP = 0.1
CBAR_WIDTH = 0.1

# Width of the black outline around markers, in points
MARKER_EDGE_WIDTH = 0.4

# Consistent loss term plotting style
TERMS = ("data_b", "zp_eqn", "zm_eqn", "dbdx_bc")
TERMS_WITHOUT_BC = ("data_b", "zp_eqn", "zm_eqn")

TERM_COLORS = {
    "data_b": "#5e5e6d",
    "zp_eqn": "#2e7d5b",
    "zm_eqn": "#4fb08a",
    "dbdx_bc": "#8c5a3f",
}

FREQ_COLORS = {
    1: "#4c72b0",
    2: "#ab008b",
    3: "#dd8452",
}

TERM_LABELS = {
    "data_b": "data loss",
    "zp_eqn": r"$z^+$ loss",
    "zm_eqn": r"$z^-$ loss",
    "dbdx_bc": "bc loss",
}

# Shared axis labels
L2_RELATIVE_ERROR_LABEL = r"relative $L_2$ error"
ABSOLUTE_ERROR_LABEL = "absolute error"
EPOCH_LABEL = "Training epoch"
LAYERS_LABEL = "Number of hidden layers"
NEURONS_LABEL = "Number of neurons per layer"
PARAMS_LABEL = "Number of parameters"

# Numbers of training points, as symbols for use inside other labels
N_DATA = r"N_{\mathrm{d}}"
N_ZPM = r"N_{z^\pm}"
N_BC = r"N_{\mathrm{bc}}"

# The same, as axis labels
N_DATA_LABEL = rf"${N_DATA}$"
N_ZPM_LABEL = rf"${N_ZPM}$"
N_BC_LABEL = rf"${N_BC}$"


def use_style() -> None:
    """Apply the style specified in the style file."""
    plt.style.use(STYLE_FILE)


def panel_grid(
    nrows: int = 1,
    ncols: int = 1,
    *,
    panel_width: float | Sequence[float],
    panel_height: float,
    hgap: float = 0.0,
    vgap: float = 0.0,
    margins: dict[str, float] | None = None,
    colorbar: Literal["none", "shared", "column"] = "none",
    cbar_gap: float = CBAR_GAP,
    cbar_width: float = CBAR_WIDTH,
) -> tuple[Figure, list[list[Axes]], list[Axes]]:
    """Create a grid of equal-height panels.

    Panel sizes, gaps, and margins are specified in inches.
    Panels are filled row-by-row from the top.
    """
    margins = MARGINS if margins is None else margins
    if isinstance(panel_width, Sequence):
        widths = [float(w) for w in panel_width]
    else:
        widths = [float(panel_width)] * ncols
    if len(widths) != ncols:
        raise ValueError(f"Got {len(widths)} panel widths for {ncols} columns")
    if colorbar not in ("none", "shared", "column"):
        raise ValueError(f"Unknown colorbar option {colorbar!r}")

    # Per-column colorbars sit inside each column's stride, a shared one after the grid
    per_column = colorbar == "column"
    cbar_space = cbar_gap + cbar_width
    strides = [w + (cbar_space if per_column else 0.0) for w in widths]
    grid_width = sum(strides) + (ncols - 1) * hgap
    grid_height = nrows * panel_height + (nrows - 1) * vgap

    fig_w = (
        margins["left"]
        + grid_width
        + (cbar_space if colorbar == "shared" else 0.0)
        + margins["right"]
    )
    fig_h = margins["bottom"] + grid_height + margins["top"]
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes = []
    caxes = []
    for row in range(nrows):
        # Rows are filled from the top, so row 0 sits highest
        y0 = margins["bottom"] + (nrows - 1 - row) * (panel_height + vgap)
        x0 = margins["left"]
        row_axes = []
        for col in range(ncols):
            row_axes.append(
                fig.add_axes(
                    [
                        x0 / fig_w,
                        y0 / fig_h,
                        widths[col] / fig_w,
                        panel_height / fig_h,
                    ]
                )
            )

            # One colorbar per column, spanning the full grid height
            if per_column and row == 0:
                caxes.append(
                    fig.add_axes(
                        [
                            (x0 + widths[col] + cbar_gap) / fig_w,
                            margins["bottom"] / fig_h,
                            cbar_width / fig_w,
                            grid_height / fig_h,
                        ]
                    )
                )
            x0 += strides[col] + hgap
        axes.append(row_axes)

    if colorbar == "shared":
        caxes.append(
            fig.add_axes(
                [
                    (margins["left"] + grid_width + cbar_gap) / fig_w,
                    margins["bottom"] / fig_h,
                    cbar_width / fig_w,
                    grid_height / fig_h,
                ]
            )
        )
    return fig, axes, caxes


def panel_letter(ax: Axes, letter: str, inside: bool = False) -> None:
    """Mark a panel (a), (b), ... above its top left corner, or inside it."""
    if inside:
        ax.text(0.05, 0.95, f"({letter})", transform=ax.transAxes, ha="left", va="top")
    else:
        ax.text(
            0.0, 1.02, f"({letter})", transform=ax.transAxes, ha="left", va="bottom"
        )


def add_colorbar(
    fig: Figure, mappable: ScalarMappable, cax: Axes, label: str | None = None
) -> Colorbar:
    """A colorbar with the ticks pointing out."""
    cbar = fig.colorbar(mappable, cax=cax, label=label)
    cbar.ax.tick_params(which="both", direction="out")
    return cbar


def save(fig: Figure, out_path: Path | str) -> None:
    """Save figure."""
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def truncate_colormap(
    cmap: mpl.colors.Colormap | str, minval: float, maxval: float, n: int = 256
) -> mpl.colors.Colormap:
    """Truncate a colormap between minval and maxval."""
    base = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    colors = base(np.linspace(minval, maxval, n))
    return mpl.colors.LinearSegmentedColormap.from_list(f"{base.name}_trunc", colors)


def ffmpeg_command(video_path: Path, fps: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-vf",
        "crop=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
