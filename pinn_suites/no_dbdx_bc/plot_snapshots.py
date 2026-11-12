from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pinn_suites.fiducial.fiducial_plotting_utils import Profile, repeat_profiles
from plotting import style
from plotting.plots import plot_field_profile
from plotting.plotting_utils import simulation_freq, suite_n_repeats, suite_values
from simulations.sim_io import SnapshotReader

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name
REF_FREQ = 2
REPEAT = 1

# Whether to shade the min/max spread of the PINN prediction across repeats
SHOW_REPEAT_BAND = True

# Snapshot times, one list per block of u and b rows
TIME_ROWS: list[list[float]] = [[22.0, 24.0, 26.0, 28.0]]


def plot_field_profiles_figure(
    profiles: dict[float, Profile],
    times_rows: list[list[float]],
    pinn_color: str,
    out_path: Path,
) -> None:
    """u and b profiles at each time, one block of rows per list of times."""
    n_cols = len(times_rows[0])
    n_blocks = len(times_rows)

    # Geometry in inches
    field_aspect = 1.6  # panel width / height
    hgap = 0.10  # between columns
    vgap_within = 0.10  # between u and b of the same block
    vgap_between = 0.45  # between blocks, room for the time label above
    margins = dict(style.MARGINS, top=0.25)

    field_width = (
        style.TEXT_WIDTH - margins["left"] - margins["right"] - (n_cols - 1) * hgap
    ) / n_cols
    field_height = field_width / field_aspect

    grid_height = (
        n_blocks * (2 * field_height + vgap_within) + (n_blocks - 1) * vgap_between
    )
    fig_w = style.TEXT_WIDTH
    fig_h = margins["bottom"] + grid_height + margins["top"]
    fig = plt.figure(figsize=(fig_w, fig_h))

    y_top = margins["bottom"] + grid_height
    for block_idx, times in enumerate(times_rows):
        y_u = y_top - field_height
        y_b = y_u - vgap_within - field_height
        for col_idx, t in enumerate(times):
            profile = profiles[t]
            x0 = margins["left"] + col_idx * (field_width + hgap)
            # Letters run row by row: this block's u row, then its b row
            u_letter_idx = 2 * block_idx * n_cols + col_idx
            b_letter_idx = u_letter_idx + n_cols

            ax_u = fig.add_axes(
                (x0 / fig_w, y_u / fig_h, field_width / fig_w, field_height / fig_h)
            )
            ax_b = fig.add_axes(
                (x0 / fig_w, y_b / fig_h, field_width / fig_w, field_height / fig_h)
            )

            plot_field_profile(
                ax_u,
                profile["x"],
                profile["u_true"],
                profile["u_pred"],
                r"$u$",
                pinn_color,
                panel_label=style.PANEL_LABELS[u_letter_idx],
                show_xlabel=False,
                show_ylabel=col_idx == 0,
                show_legend=(block_idx == n_blocks - 1 and col_idx == n_cols - 1),
                pred_min=profile.get("u_min"),
                pred_max=profile.get("u_max"),
            )

            plot_field_profile(
                ax_b,
                profile["x"],
                profile["b_true"],
                profile["b_pred"],
                r"$b$",
                pinn_color,
                panel_label=style.PANEL_LABELS[b_letter_idx],
                show_xlabel=True,
                show_ylabel=col_idx == 0,
                pred_min=profile.get("b_min"),
                pred_max=profile.get("b_max"),
            )

            # Time above each block's u row
            ax_u.text(
                0.5,
                1.02,
                rf"$t = {t:g}$",
                transform=ax_u.transAxes,
                ha="center",
                va="bottom",
            )

        y_top = y_b - vgap_between

    style.save(fig, out_path)


def main() -> None:
    n_batches = 10

    sim_by_freq = {
        simulation_freq(str(s)): str(s) for s in suite_values(SUITE, "simulations")
    }
    if REF_FREQ not in sim_by_freq:
        raise RuntimeError(
            f"{SUITE} has no freq={REF_FREQ} simulation; "
            f"available: {sorted(sim_by_freq)}"
        )

    style.use_style()
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    band_repeats = (
        list(range(1, suite_n_repeats(SUITE) + 1)) if SHOW_REPEAT_BAND else []
    )
    profiles = repeat_profiles(
        SUITE_DIR / "param_files",
        SUITE_DIR / "models",
        SnapshotReader(sim_by_freq[REF_FREQ]),
        REF_FREQ,
        [t for row in TIME_ROWS for t in row],
        REPEAT,
        band_repeats,
        n_batches,
        device,
    )

    plot_field_profiles_figure(
        profiles,
        TIME_ROWS,
        style.FREQ_COLORS[REF_FREQ],
        SUITE_DIR / f"field_snapshots_freq{REF_FREQ:02d}_repeat{REPEAT:02d}.pdf",
    )


if __name__ == "__main__":
    main()
