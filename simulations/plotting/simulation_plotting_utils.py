from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import colorcet as cc
import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from plotting.style import truncate_colormap
from simulations.sim_io import SIMULATION_BASE_DIR, SnapshotReader

SIM_CMAP = truncate_colormap(cc.cm.bmy_r, 0.15, 0.95)
COLOR_ZM = "#FF0000"  # the drive z^-(x=0)
COLOR_B = "#000000"  # the wall field b(x=1)
ATTRACTOR_XLIM = (-1.8, 1.8)
ATTRACTOR_YLIM = (-1.4, 1.4)
ZM_LABEL = r"$z^-_{\rm in}$"
B_LABEL = r"$b_{\rm obs}$"
FIELD_NAMES = ("u", "b", "zp", "zm")


@dataclass
class Snapshots:
    """A run's snapshots, everything the field/attractor figures need."""

    t: np.ndarray
    x: np.ndarray
    fields: dict[str, np.ndarray]
    zm_drive: np.ndarray
    b_edge: np.ndarray


def find_sim_config(run: str) -> Path:
    """The single sim_config_*.py inside a simulation run directory."""
    matches = sorted((SIMULATION_BASE_DIR / run).glob("sim_config_*.py"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one sim_config_*.py in {run!r}, found {len(matches)}."
        )
    return matches[0]


def _drive(config: Any, t: np.ndarray) -> np.ndarray:
    """The boundary drive z^-(x=0) at each snapshot time."""
    return np.array([config.zm_in(float(ti)) for ti in t])


def load_attractor(
    reader: SnapshotReader, config: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Just the values needed for the attractor: (t, zm_drive, b_edge)."""
    t = np.asarray(reader.times, dtype=float)
    zm_drive = _drive(config, t)
    b_edge = np.asarray(reader.load_fields_at_x(config.L, "b")["b"], dtype=float)
    return t, zm_drive, b_edge


def load_snapshots(reader: SnapshotReader, config: Any) -> Snapshots:
    """Every field over every snapshot, plus the attractor quantities."""
    t = np.asarray(reader.times, dtype=float)
    x = np.asarray(reader.x, dtype=float)

    idx_edge = int(np.argmin(np.abs(x - config.L)))
    if not np.isclose(x[idx_edge], config.L):
        raise ValueError(f"x = {config.L} is not a snapshot point.")

    fields = {name: np.empty((t.size, x.size)) for name in FIELD_NAMES}
    for index in range(len(reader.paths)):
        snapshot = reader.load_fields(index, *FIELD_NAMES)
        for name in FIELD_NAMES:
            fields[name][index] = snapshot[name]

    return Snapshots(
        t=t,
        x=x,
        fields=fields,
        zm_drive=_drive(config, t),
        b_edge=fields["b"][:, idx_edge],
    )


def plot_attractor(
    ax: Axes,
    t: np.ndarray,
    zm_drive: np.ndarray,
    b_edge: np.ndarray,
    t_min: float,
    t_max: float,
    *,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap = SIM_CMAP,
    xlim: tuple[float, float] = ATTRACTOR_XLIM,
    ylim: tuple[float, float] = ATTRACTOR_YLIM,
    alpha: float = 1.0,
    rasterized: bool = True,
) -> LineCollection | None:
    """Draw the (zm_drive, b_edge) trajectory over [t_min, t_max], coloured by t."""
    if t_min > t_max:
        raise ValueError(f"t_min={t_min} must not exceed t_max={t_max}.")

    window = (t >= t_min) & (t <= t_max)
    t, zm_drive, b_edge = t[window], zm_drive[window], b_edge[window]

    line: LineCollection | None = None
    if t.size >= 2:
        points = np.column_stack([zm_drive, b_edge]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line = LineCollection(
            segments, cmap=cmap, norm=norm, alpha=alpha, rasterized=rasterized, zorder=2
        )
        line.set_array(t[:-1])
        ax.add_collection(line)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return line
