import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from pinn_suites.fiducial.fiducial_plotting_utils import get_model
from plotting import style
from plotting.pinn_utils import evaluate_pinn_u_b, load_pinn, slice_coords
from plotting.plots import plot_field_profile
from plotting.plotting_utils import freq_label, simulation_freq, suite_values
from simulations.sim_io import SnapshotReader

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name
REPEAT = 1

# Frequency cases, one per column
FREQS = (1, 2, 3)

# Animated time range, the PINN training window
T_ANIM_MIN = 20.0
T_ANIM_MAX = 30.0

FRAME_STRIDE = 8
FPS = 24

FIELD_ASPECT = 1.8  # panel width / height
HGAP = 0.10
VGAP = 0.10

SIM_KWARGS: dict[str, Any] = {"ls": "-", "lw": 1.0, "zorder": 1}
PINN_KWARGS: dict[str, Any] = {"ls": "--", "lw": 1.0, "zorder": 2}


class FreqSource:
    """One frequency case's snapshots and trained PINN, loaded once for all frames."""

    def __init__(
        self, freq: int, simulation: str, device: torch.device, n_batches: int
    ) -> None:
        self.freq = freq
        self.reader = SnapshotReader(simulation)
        self.device = device
        self.n_batches = n_batches

        yaml_path, model_path, name = get_model(
            SUITE_DIR / "param_files", SUITE_DIR / "models", freq, REPEAT
        )
        print(f"freq={freq} repeat={REPEAT}: {name}")
        self.pinn = load_pinn(yaml_path, model_path, device)

    def fields_at(self, time: float) -> dict[str, np.ndarray]:
        """Simulation and PINN (u, b) at the snapshot nearest to a time."""
        idx = int(np.argmin(np.abs(self.reader.times - time)))
        true = self.reader.load_fields(idx, "u", "b")
        coords = slice_coords(self.reader.x, float(self.reader.times[idx]), self.device)
        u_pred, b_pred = evaluate_pinn_u_b(self.pinn, coords, self.n_batches)
        return {
            "u_true": true["u"],
            "b_true": true["b"],
            "u_pred": u_pred.cpu().numpy(),
            "b_pred": b_pred.cpu().numpy(),
        }


def render_frame(
    sources: list[FreqSource],
    time: float,
    field_width: float,
    field_height: float,
    margins: dict[str, float],
) -> Figure:
    """Columns are frequency cases, rows are the u and b fields."""
    ncols = len(sources)
    fig, axes, _ = style.panel_grid(
        2,
        ncols,
        panel_width=field_width,
        panel_height=field_height,
        hgap=HGAP,
        vgap=VGAP,
        margins=margins,
    )

    for col, source in enumerate(sources):
        fields = source.fields_at(time)
        for row, name in enumerate(("u", "b")):
            plot_field_profile(
                axes[row][col],
                source.reader.x,
                fields[f"{name}_true"],
                fields[f"{name}_pred"],
                rf"${name}$",
                style.FREQ_COLORS[source.freq],
                sim_kwargs=SIM_KWARGS,
                pinn_kwargs=PINN_KWARGS,
                panel_label=style.PANEL_LABELS[row * ncols + col],
                show_xlabel=(row == 1),
                show_ylabel=(col == 0),
                show_legend=(row == 0 and col == ncols - 1),
            )
        axes[0][col].text(
            0.5,
            1.02,
            freq_label(source.freq),
            transform=axes[0][col].transAxes,
            ha="center",
            va="bottom",
        )

    return fig


def main() -> None:
    n_batches = 10

    style.use_style()
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sim_by_freq = {
        simulation_freq(str(s)): str(s) for s in suite_values(SUITE, "simulations")
    }
    sources = [FreqSource(freq, sim_by_freq[freq], device, n_batches) for freq in FREQS]

    # Same layout as plot_snapshots, with room above for the column titles
    margins = dict(style.MARGINS, top=0.30)
    field_width = (
        style.TEXT_WIDTH - margins["left"] - margins["right"] - (len(FREQS) - 1) * HGAP
    ) / len(FREQS)
    field_height = field_width / FIELD_ASPECT

    # A single PDF at the final time
    fig = render_frame(sources, T_ANIM_MAX, field_width, field_height, margins)
    style.save(fig, SUITE_DIR / "field_snapshots_fiducial.pdf")

    # The animation over [T_ANIM_MIN, T_ANIM_MAX]
    times = sources[0].reader.times
    times = times[(times >= T_ANIM_MIN) & (times <= T_ANIM_MAX)][::FRAME_STRIDE]
    duration = len(times) / FPS
    print(
        f"{len(times)} frames at {FPS} fps -> {duration:.1f} s animation"
        f" ({duration / (T_ANIM_MAX - T_ANIM_MIN):.2f} s per sim time unit)"
    )

    video_path = SUITE_DIR / "animation_fiducial_snapshots.mp4"
    proc = subprocess.Popen(
        style.ffmpeg_command(video_path, FPS), stdin=subprocess.PIPE
    )
    assert proc.stdin is not None
    for time in times:
        fig = render_frame(sources, float(time), field_width, field_height, margins)
        fig.savefig(proc.stdin, format="png", dpi=300)
        plt.close(fig)
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    print(f"Saved {video_path}")


if __name__ == "__main__":
    main()
