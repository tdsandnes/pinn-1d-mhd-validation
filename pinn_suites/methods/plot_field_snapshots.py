from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import offset_copy

from pinn_suites.methods.plot_data_sampling_methods import method_cases
from plotting import style
from plotting.pinn_utils import snapshot_field_profiles
from plotting.plots import PINN_LABEL, plot_field_profile
from plotting.plotting_utils import model_int, suite_values, training_data_points
from simulations.sim_io import SnapshotReader

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name
T_PROFILE = 22.0

FIELDS = ("u", "b")

# Each method gets a u and a b panel side by side, two methods to a row
METHODS_PER_ROW = 2
N_ROWS = 3

FIELD_ASPECT = 1.6  # panel width / height

# Gaps in inches. Every panel shares one y range, so only the leftmost column
# has y tick labels and the panels can sit close: the u and b panels of a
# method almost touch, with a little more room between methods
PAIR_GAP = 0.06
METHOD_GAP = 0.20
# Between rows, and above the top row, room for the method names and field
# symbols above the panels
ROW_GAP = 0.22

# Points between the top of a pair of panels and the text above it
ABOVE_PANEL_PAD = 2.0

# The PINN colour changes from method to method, so its legend line is grey
LEGEND_PINN_COLOR = "0.35"

Profile = dict[str, np.ndarray]


def mean_abs_error(profile: Profile) -> float:
    """The PINN's absolute error averaged over x and over u and b."""
    return float(
        np.mean(
            [
                np.mean(np.abs(profile[f"{field}_pred"] - profile[f"{field}_true"]))
                for field in FIELDS
            ]
        )
    )


def worst_repeat(repeats: dict[int, Profile]) -> int:
    """The repeat furthest from the simulation, over both u and b."""
    return max(repeats, key=lambda r: mean_abs_error(repeats[r]))


def draw_field_panel(
    ax: Axes,
    field: str,
    profile: Profile,
    color: Any,
    panel_label: str,
    show_xlabel: bool,
    show_yticklabels: bool,
    show_legend: bool,
) -> None:
    """The simulation and the PINN for one field."""
    plot_field_profile(
        ax,
        profile["x"],
        profile[f"{field}_true"],
        profile[f"{field}_pred"],
        rf"${field}$",
        pinn_color=color,
        panel_label=panel_label,
        show_xlabel=show_xlabel,
        show_ylabel=show_yticklabels,
        show_legend=show_legend,
    )
    # The field is named above the panel instead of on the y-axis
    ax.set_ylabel("")

    legend = ax.get_legend()
    if legend is not None:
        for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=True):
            if handle is not None and text.get_text() == PINN_LABEL:
                handle.set_color(LEGEND_PINN_COLOR)

    # The panels are narrow, so fix the x ticks rather than leave them to matplotlib
    ax.xaxis.set_major_locator(MultipleLocator(0.25))


def label_pair(u_ax: Axes, b_ax: Axes, label: str) -> None:
    """Method name centred above its u and b panels, with u above the top-left
    corner and b above the top-right."""
    fig = u_ax.figure
    u_box, b_box = u_ax.get_position(), b_ax.get_position()
    above = offset_copy(fig.transFigure, fig=fig, y=ABOVE_PANEL_PAD, units="points")
    for x, text, ha in (
        (u_box.x0, r"$u$", "left"),
        (0.5 * (u_box.x0 + b_box.x1), label, "center"),
        (b_box.x1, r"$b$", "right"),
    ):
        fig.text(x, u_box.y1, text, transform=above, ha=ha, va="bottom")


def plot_field_snapshots_figure(
    profile_by_method: dict[str, Profile],
    out_path: Path,
) -> None:
    """u and b against the simulation for every method case."""
    n_cols = METHODS_PER_ROW * len(FIELDS)
    margins = dict(style.MARGINS, left=0.30, top=ROW_GAP, bottom=0.38)
    gaps = [PAIR_GAP, METHOD_GAP] * (METHODS_PER_ROW - 1) + [PAIR_GAP]
    panel_width = (
        style.TEXT_WIDTH - margins["left"] - margins["right"] - sum(gaps)
    ) / n_cols
    panel_height = panel_width / FIELD_ASPECT

    fig_width = style.TEXT_WIDTH
    fig_height = (
        margins["bottom"]
        + N_ROWS * panel_height
        + (N_ROWS - 1) * ROW_GAP
        + margins["top"]
    )
    fig = plt.figure(figsize=(fig_width, fig_height))

    cases = method_cases()
    for i, (method_id, label, color) in enumerate(cases):
        row, slot = divmod(i, METHODS_PER_ROW)
        y0 = margins["bottom"] + (N_ROWS - 1 - row) * (panel_height + ROW_GAP)

        axes: list[Axes] = []
        for f in range(len(FIELDS)):
            col = slot * len(FIELDS) + f
            x0 = margins["left"] + col * panel_width + sum(gaps[:col])
            axes.append(
                fig.add_axes(
                    (
                        x0 / fig_width,
                        y0 / fig_height,
                        panel_width / fig_width,
                        panel_height / fig_height,
                    )
                )
            )

        profile = profile_by_method.get(method_id)
        if profile is None:
            print(f"Warning: no profile for method={method_id} ({label})")
            for ax in axes:
                ax.set_visible(False)
            continue

        for f, (ax, field) in enumerate(zip(axes, FIELDS, strict=True)):
            draw_field_panel(
                ax,
                field,
                profile,
                color,
                panel_label=style.PANEL_LABELS[i * len(FIELDS) + f],
                show_xlabel=row == N_ROWS - 1,
                show_yticklabels=slot == 0 and f == 0,
                # In the bottom-right panel, as in the fiducial snapshot figure
                show_legend=i == len(cases) - 1 and f == len(FIELDS) - 1,
            )
        label_pair(axes[0], axes[1], label)

    style.save(fig, out_path)


def main() -> None:
    n_batches = 10

    param_dir = SUITE_DIR / "param_files"
    model_dir = SUITE_DIR / "models"

    simulations = suite_values(SUITE, "simulations")
    if len(simulations) != 1:
        raise RuntimeError(
            f"{SUITE} is expected to have exactly one simulation, "
            f"got {len(simulations)}."
        )
    reader = SnapshotReader(str(simulations[0]))

    style.use_style()
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_paths = sorted(model_dir.glob("*.pth"))
    if not model_paths:
        raise RuntimeError(f"No trained models found in {model_dir.resolve()}")

    # Only the models trained on the most data points
    max_n_data = max(model_int(path.stem, "total_data_points") for path in model_paths)
    kept_paths = [
        path
        for path in model_paths
        if model_int(path.stem, "total_data_points") == max_n_data
    ]
    n_train = training_data_points(kept_paths[0].stem)
    print(f"Plotting u and b at t = {T_PROFILE} for N_data = {n_train}")

    # Every repeat of every kept model, by method and repeat
    profiles: dict[str, dict[int, Profile]] = {}
    for path in kept_paths:
        name = path.stem
        method_id = f"{model_int(name, 'method'):05d}"
        repeat = model_int(name, "repeat")

        print(f"Evaluating snapshot for {name}")
        profiles.setdefault(method_id, {})[repeat] = snapshot_field_profiles(
            param_dir / f"param_{name}.yml",
            path,
            reader,
            T_PROFILE,
            n_batches,
            device,
        )

    # Each method shows the repeat furthest from the simulation at this time
    worst_profiles: dict[str, Profile] = {}
    for method_id, repeats in profiles.items():
        repeat = worst_repeat(repeats)
        print(f"method={method_id}: worst is repeat {repeat} of {len(repeats)}")
        worst_profiles[method_id] = repeats[repeat]

    plot_field_snapshots_figure(
        worst_profiles, SUITE_DIR / "field_snapshots_methods.pdf"
    )


if __name__ == "__main__":
    main()
