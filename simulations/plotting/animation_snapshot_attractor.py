from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from plotting import style
from simulations.plotting.plot_snapshot_attractor import draw_frame
from simulations.plotting.simulation_plotting_utils import (
    find_sim_config,
    load_snapshots,
)
from simulations.sim_io import SnapshotReader, load_sim_config

PLOTTING_DIR = Path(__file__).resolve().parent


def animate(
    run: str,
    t_min: float = 0.0,
    t_max: float | None = None,
    frame_stride: int = 1,
    fps: int = 24,
    *,
    history_lines: int = 0,
    history_dt: float = 1.0,
    history_alpha_min: float = 0.2,
    history_alpha_max: float = 0.7,
) -> None:
    """Write animation_<run>.mp4 over the times [t_min, t_max]."""
    style.use_style()
    config_path = find_sim_config(run)
    reader = SnapshotReader(config_path)
    config = load_sim_config(config_path)
    snap = load_snapshots(reader, config)

    # Fix norm over the whole run to keep colours consistent across frames
    t_end = t_max if t_max is not None else float(snap.t[-1])
    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_end)

    frame_times = snap.t[(snap.t >= t_min) & (snap.t <= t_end)][::frame_stride]
    duration = len(frame_times) / fps
    print(
        f"{len(frame_times)} frames at {fps} fps, {duration:.1f} s animation "
        f"({duration / (t_end - t_min):.2f} s per sim time unit)"
    )

    video_path = PLOTTING_DIR / f"animation_{run}.mp4"
    proc = subprocess.Popen(
        style.ffmpeg_command(video_path, fps), stdin=subprocess.PIPE
    )
    assert proc.stdin is not None
    for t_frame in frame_times:
        fig = draw_frame(
            snap,
            t_min,
            float(t_frame),
            norm,
            history_lines=history_lines,
            history_dt=history_dt,
            history_alpha_min=history_alpha_min,
            history_alpha_max=history_alpha_max,
        )
        fig.savefig(proc.stdin, format="png", dpi=300)
        plt.close(fig)

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    print(f"Saved {video_path}")


def main() -> None:
    animate(
        "2freq_1Ma_500Re_500Rm",
        t_min=0.0,
        t_max=30.0,
        frame_stride=8,
        fps=24,
    )


if __name__ == "__main__":
    main()
