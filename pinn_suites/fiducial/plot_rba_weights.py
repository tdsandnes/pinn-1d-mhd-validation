from pathlib import Path

import numpy as np
import torch
from matplotlib.ticker import LogLocator, NullFormatter

from pinn_suites.fiducial.fiducial_plotting_utils import get_model
from plotting import style
from plotting.plots import log_ylim

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name
FREQ = 2
REPEAT = 1

TERMS = style.TERMS
BAND_ALPHA = 0.18

# term -> {"median" | "min" | "max": one value per epoch}
WeightHistory = dict[str, dict[str, list[float]]]


def load_rba_weights(checkpoint_path: Path) -> WeightHistory:
    """RBA point-weight median, min and max of every term from a checkpoint."""
    logs = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["logs"]
    rba_log: list[dict[str, dict[str, float]]] = logs["epoch_point_weight_info_log"]
    if not rba_log:
        raise ValueError(
            f"{checkpoint_path.name} has no epoch_point_weight_info_log; "
            "the suite needs log_point_weight_info: true"
        )
    missing = [term for term in TERMS if term not in rba_log[-1]]
    if missing:
        raise KeyError(
            f"{checkpoint_path.name}: RBA terms {missing} not logged. "
            f"Available: {sorted(rba_log[-1])}"
        )

    return {
        term: {
            stat: [entry[term][stat] for entry in rba_log]
            for stat in ("median", "min", "max")
        }
        for term in TERMS
    }


def plot_rba_figure(weights: WeightHistory, out_path: Path) -> None:
    """Plot each term's RBA weights in its own panel."""
    n_cols = len(TERMS)
    hgap = 0.1
    margins = dict(style.MARGINS, top=0.25, left=0.50)

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
        for term in TERMS
        for stat in ("min", "max")
        for value in weights[term][stat]
    )

    for j, (ax, term) in enumerate(zip(axes[0], TERMS, strict=True)):
        color = style.TERM_COLORS[term]
        epochs = np.arange(1, len(weights[term]["median"]) + 1)
        ax.fill_between(
            epochs,
            weights[term]["min"],
            weights[term]["max"],
            color=color,
            alpha=BAND_ALPHA,
            linewidth=0,
            rasterized=True,
        )
        ax.plot(epochs, weights[term]["median"], color=color)
        ax.text(
            0.5,
            1.02,
            style.TERM_LABELS[term],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )
        style.panel_letter(ax, style.PANEL_LABELS[j])

        ax.set_yscale("log")
        ax.set_ylim(*y_lim)
        ax.set_xlabel(style.EPOCH_LABEL)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        if j > 0:
            ax.tick_params(labelleft=False)

    axes[0][0].set_ylabel("RBA weight")

    style.save(fig, out_path)


def main() -> None:
    style.use_style()

    _, model_path, name = get_model(
        SUITE_DIR / "param_files", SUITE_DIR / "models", FREQ, REPEAT
    )
    print(f"freq={FREQ} repeat={REPEAT}: {name}")

    plot_rba_figure(
        load_rba_weights(model_path),
        SUITE_DIR / f"rba_weights_vs_iteration_freq{FREQ:02d}_repeat{REPEAT:02d}.pdf",
    )


if __name__ == "__main__":
    main()
