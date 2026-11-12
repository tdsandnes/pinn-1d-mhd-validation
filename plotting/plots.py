from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from plotting.style import MARKER_EDGE_WIDTH, panel_letter

# Default style of plot_field_profile
SIM_COLOR = "black"
PINN_LABEL = "PINN"
SIM_LABEL = "simulation"
PINN_KWARGS: dict[str, Any] = {"ls": "--", "lw": 0.8, "zorder": 2}
SIM_KWARGS: dict[str, Any] = {"ls": "-", "lw": 0.8, "zorder": 1}


def plot_field_profile(
    ax: Axes,
    x: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    ylabel: str,
    pinn_color: str | np.ndarray,
    sim_color: str | np.ndarray = SIM_COLOR,
    pinn_label: str | None = PINN_LABEL,
    sim_label: str | None = SIM_LABEL,
    pinn_kwargs: dict[str, Any] | None = PINN_KWARGS,
    sim_kwargs: dict[str, Any] | None = SIM_KWARGS,
    panel_label: str | None = None,
    symbol: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_legend: bool = False,
    pred_min: np.ndarray | None = None,
    pred_max: np.ndarray | None = None,
    band_alpha: float = 0.3,
    band_label: str | None = None,
) -> None:
    if pred_min is not None and pred_max is not None:
        ax.fill_between(
            x,
            pred_min,
            pred_max,
            color=pinn_color,
            alpha=band_alpha,
            linewidth=0,
            zorder=0,
            label=band_label,
        )

    ax.plot(x, true, color=sim_color, label=sim_label, **(sim_kwargs or {}))
    ax.plot(x, pred, color=pinn_color, label=pinn_label, **(pinn_kwargs or {}))

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.2, 1.2)

    if show_ylabel:
        ax.set_ylabel(ylabel)
        ax.yaxis.labelpad = -2.0
    else:
        ax.tick_params(labelleft=False)

    if show_xlabel:
        ax.set_xlabel(r"$x$")
        # Drop the end tick labels, which collide with the next column
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda v, pos: (
                    "" if np.isclose(v, 0) or np.isclose(v, 1) else f"{v:.2f}"
                )
            )
        )
    else:
        ax.tick_params(labelbottom=False)

    if panel_label is not None:
        panel_letter(ax, panel_label, inside=True)
    if symbol is not None:
        ax.text(
            0.95,
            0.95,
            symbol,
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=pinn_color,
        )
    if show_legend:
        ax.legend(
            loc="lower right",
            handlelength=2.1,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.2,
        )


def draw_error_line(
    ax: Axes,
    x: Sequence[float],
    errors: Sequence[Sequence[float]],
    n_repeats: int,
    *,
    color: Any,
    label: str | None = None,
    measure: Literal["mean", "median"] = "mean",
    show_range: bool = False,
    range_alpha: float = 0.30,
    marker_size: float | Sequence[float] = 8.0,
    marker_edge_width: float = MARKER_EDGE_WIDTH,
    zorder: float = 1.0,
) -> None:
    """Plot of error against x with shaded region for repeats."""
    if any(len(e) != n_repeats for e in errors):
        raise ValueError(f"Each x needs exactly {n_repeats} repeat error values.")

    x_arr = np.asarray(x)
    errors_arr = np.asarray(errors, dtype=float)  # shape (n_x, n_repeats)

    if n_repeats == 1:
        y = errors_arr[:, 0]
    elif measure == "mean":
        y = errors_arr.mean(axis=1)
    elif measure == "median":
        y = np.median(errors_arr, axis=1)
    else:
        raise ValueError(f"measure must be 'mean' or 'median', not {measure!r}")

    if show_range and n_repeats > 1:
        ax.fill_between(
            x_arr,
            errors_arr.min(axis=1),
            errors_arr.max(axis=1),
            color=color,
            alpha=range_alpha,
            linewidth=0,
            zorder=zorder - 0.5,
        )

    ax.plot(x_arr, y, color=color, zorder=zorder, label=label)
    ax.scatter(
        x_arr,
        y,
        s=marker_size,
        color=color,
        edgecolor="black",
        linewidths=marker_edge_width,
        zorder=zorder + 0.5,
    )


def log_ylim(values: Iterable[float]) -> tuple[float, float]:
    """Log-scale y-limits, padded on each side."""
    positive = np.array([v for v in values if v > 0], dtype=float)
    lb, ub = float(positive.min()), float(positive.max())
    pad = 10 ** (0.05 * np.log10(ub / lb))
    return lb / pad, ub * pad
