from pathlib import Path

import numpy as np
import torch
from matplotlib.axes import Axes

from pinn_suites.fiducial.fiducial_plotting_utils import get_model
from plotting import style
from plotting.plots import log_ylim
from plotting.plotting_utils import freq_label, simulation_freq, suite_values

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name
REPEAT = 1
# The raw per-epoch losses are drawn faintly, with a rolling geometric mean on top
RAW_LOSS_ALPHA = 0.25
RAW_LOSS_LINEWIDTH = 0.4
SMOOTH_LOSS_LINEWIDTH = 0.9
# Width of the rolling window, in epochs
SMOOTH_WINDOW_EPOCHS = 1000

TERMS = style.TERMS_WITHOUT_BC
GEOMEAN_KWARGS = {"color": "black", "ls": "--", "label": "lowest geometric mean"}

# term -> (train losses, validation losses), one entry per epoch
LossHistory = dict[str, tuple[list[float], list[float]]]


def load_losses(checkpoint_path: Path) -> LossHistory:
    """Training and validation loss histories of every term from a checkpoint."""
    logs = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["logs"]
    train_log: list[dict[str, float]] = logs["epoch_train_losses_log"]
    val_log: list[dict[str, float]] = logs["epoch_val_losses_log"]

    if not val_log or len(train_log) != len(val_log):
        raise ValueError(
            f"{checkpoint_path.name}: {len(train_log)} train and {len(val_log)} "
            "validation loss entries"
        )
    missing = [term for term in TERMS if term not in val_log[-1]]
    if missing:
        raise KeyError(
            f"{checkpoint_path.name}: terms {missing} not logged. "
            f"Available: {sorted(val_log[-1])}"
        )

    return {
        term: ([entry[term] for entry in train_log], [entry[term] for entry in val_log])
        for term in TERMS
    }


def running_min_geometric_mean(rows: list[list[float]]) -> np.ndarray:
    """Lowest geometric mean of the per-term losses seen up to each epoch."""
    geomean = np.exp(np.mean(np.log(np.asarray(rows)), axis=0))
    return np.minimum.accumulate(geomean)


def rolling_geometric_mean(values: list[float], window: int) -> np.ndarray:
    """Geometric mean over a centred window, which shrinks at the ends."""
    log_values = np.log(np.asarray(values))
    kernel = np.ones(window)
    total = np.convolve(log_values, kernel, mode="same")
    count = np.convolve(np.ones_like(log_values), kernel, mode="same")
    return np.exp(total / count)


def draw_loss_panel(ax: Axes, history: LossHistory, index: int) -> None:
    """Draw one loss trace per term, index 0 for training and 1 for validation."""
    for i, term in enumerate(TERMS):
        losses = history[term][index]
        # pinninit logs training losses from epoch 0 and validation from epoch 1
        epochs = np.arange(len(losses)) + index
        color = style.TERM_COLORS[term]
        zorder = len(TERMS) - i
        # Rasterised: tens of thousands of points per line bloat the PDF otherwise
        ax.plot(
            epochs,
            losses,
            color=color,
            alpha=RAW_LOSS_ALPHA,
            linewidth=RAW_LOSS_LINEWIDTH,
            zorder=zorder,
            rasterized=True,
        )
        ax.plot(
            epochs,
            rolling_geometric_mean(losses, SMOOTH_WINDOW_EPOCHS),
            color=color,
            label=style.TERM_LABELS[term],
            linewidth=SMOOTH_LOSS_LINEWIDTH,
            zorder=zorder + len(TERMS),
        )


def plot_losses_figure(per_freq: dict[int, LossHistory], out_path: Path) -> None:
    """Plot training and validation losses for all frequency cases."""
    freqs = sorted(per_freq)
    n_cols = len(freqs)
    hgap = 0.10
    # The rows sit far enough apart for the bottom row's panel letters
    vgap = 0.22
    # Room below for the descenders in the x-axis label
    margins = dict(style.MARGINS, top=0.25, left=0.50, bottom=0.38)

    panel_width = (
        style.TEXT_WIDTH - margins["left"] - margins["right"] - (n_cols - 1) * hgap
    ) / n_cols
    fig, axes, _ = style.panel_grid(
        nrows=2,
        ncols=n_cols,
        panel_width=panel_width,
        panel_height=panel_width / 1.4,
        hgap=hgap,
        vgap=vgap,
        margins=margins,
    )

    for j, freq in enumerate(freqs):
        history = per_freq[freq]
        draw_loss_panel(axes[0][j], history, 0)
        draw_loss_panel(axes[1][j], history, 1)

        val_losses = [history[term][1] for term in TERMS]
        epochs = np.arange(1, len(val_losses[0]) + 1)
        axes[1][j].plot(
            epochs, running_min_geometric_mean(val_losses), **GEOMEAN_KWARGS
        )

        axes[0][j].text(
            0.5,
            1.02,
            freq_label(freq),
            transform=axes[0][j].transAxes,
            ha="center",
            va="bottom",
        )

    y_lim = log_ylim(
        loss
        for history in per_freq.values()
        for train, val in history.values()
        for loss in (*train, *val)
    )
    for row in axes:
        for ax in row:
            ax.set_yscale("log")
            ax.set_ylim(*y_lim)

    # Letters run along the top row first, then the bottom row
    for i, ax in enumerate(ax for row in axes for ax in row):
        style.panel_letter(ax, style.PANEL_LABELS[i])

    for j in range(n_cols):
        axes[0][j].tick_params(labelbottom=False)
        if j > 0:
            axes[0][j].tick_params(labelleft=False)
            axes[1][j].tick_params(labelleft=False)

    axes[0][0].set_ylabel("Training loss")
    axes[1][0].set_ylabel("Validation loss")
    for ax in axes[1]:
        ax.set_xlabel(style.EPOCH_LABEL)

    axes[1][-1].legend(
        loc="upper right",
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

    per_freq: dict[int, LossHistory] = {}
    for simulation in suite_values(SUITE, "simulations"):
        freq = simulation_freq(str(simulation))
        _, model_path, name = get_model(param_dir, model_dir, freq, REPEAT)
        print(f"freq={freq} repeat={REPEAT}: {name}")
        per_freq[freq] = load_losses(model_path)

    plot_losses_figure(per_freq, SUITE_DIR / "losses_vs_iteration.pdf")


if __name__ == "__main__":
    main()
