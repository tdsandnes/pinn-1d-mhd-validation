from pathlib import Path

import numpy as np
import torch

from pinn_suites.fiducial.fiducial_plotting_utils import get_model
from plotting import style
from plotting.plots import log_ylim
from plotting.plotting_utils import freq_label, simulation_freq, suite_values

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name
REPEAT = 1

TERMS = style.TERMS

# term -> one loss-term weight per epoch
WeightHistory = dict[str, list[float]]


def load_w23_weights(checkpoint_path: Path) -> WeightHistory:
    """W23 loss-term weight history of every term from a checkpoint."""
    logs = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["logs"]
    w23_log: list[dict[str, float]] = logs["epoch_loss_term_weights_log"]
    if not w23_log:
        raise ValueError(f"{checkpoint_path.name} has no epoch_loss_term_weights_log")
    missing = [term for term in TERMS if term not in w23_log[-1]]
    if missing:
        raise KeyError(
            f"{checkpoint_path.name}: W23 terms {missing} not logged. "
            f"Available: {sorted(w23_log[-1])}"
        )

    return {term: [entry[term] for entry in w23_log] for term in TERMS}


def plot_w23_figure(per_freq: dict[int, WeightHistory], out_path: Path) -> None:
    """Plot every term's W23 weights, one panel per frequency case."""
    freqs = sorted(per_freq)
    n_cols = len(freqs)
    hgap = 0.10
    # Room below for the descenders in the x-axis label
    margins = dict(style.MARGINS, top=0.25, left=0.50, bottom=0.38)

    panel_width = (
        style.TEXT_WIDTH - margins["left"] - margins["right"] - (n_cols - 1) * hgap
    ) / n_cols
    fig, axes, _ = style.panel_grid(
        ncols=n_cols,
        panel_width=panel_width,
        panel_height=panel_width / 1.4,
        hgap=hgap,
        margins=margins,
    )

    y_lim = log_ylim(
        value
        for weights in per_freq.values()
        for trace in weights.values()
        for value in trace
    )

    for j, (ax, freq) in enumerate(zip(axes[0], freqs, strict=True)):
        weights = per_freq[freq]
        for term in TERMS:
            epochs = np.arange(1, len(weights[term]) + 1)
            ax.plot(
                epochs,
                weights[term],
                color=style.TERM_COLORS[term],
                label=style.TERM_LABELS[term],
            )
        ax.text(
            0.5,
            1.02,
            freq_label(freq),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )

        style.panel_letter(ax, style.PANEL_LABELS[j])

        ax.set_yscale("log")
        ax.set_ylim(*y_lim)
        ax.set_xlabel(style.EPOCH_LABEL)
        if j > 0:
            ax.tick_params(labelleft=False)

    axes[0][0].set_ylabel("Loss-term weight")
    axes[0][-1].legend(
        loc="center right",
        handlelength=1.8,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.2,
    )

    style.save(fig, out_path)


def main() -> None:
    param_dir = SUITE_DIR / "param_files"
    model_dir = SUITE_DIR / "models"

    style.use_style()

    per_freq: dict[int, WeightHistory] = {}
    for simulation in suite_values(SUITE, "simulations"):
        freq = simulation_freq(str(simulation))
        _, model_path, name = get_model(param_dir, model_dir, freq, REPEAT)
        print(f"freq={freq} repeat={REPEAT}: {name}")
        per_freq[freq] = load_w23_weights(model_path)

    plot_w23_figure(per_freq, SUITE_DIR / "w23_weights_vs_iteration.pdf")


if __name__ == "__main__":
    main()
