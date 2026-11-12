from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import colorcet as cc
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from pinn_suites.nn_size.select_nn_size import (
    SizeSelection,
    select_default_size,
)
from plotting import style
from plotting.evaluate_test_dataset import (
    TestError,
    load_test_data,
    model_errors,
)
from plotting.plots import draw_error_line
from plotting.plotting_utils import (
    model_int,
    suite_n_repeats,
    suite_values,
)
from simulations.sim_io import sim_data_dir

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name

# Error range shared by the heatmap and the scatter plot
ERROR_MIN = 4.7e-2
ERROR_MAX = 1.2

LAYER_CMAP = style.truncate_colormap(cc.cm.bmy_r, 0.05, 1.00)
ERROR_CMAP = cc.cm.kbgyw

# Marker area range, scaled with the number of neurons
MIN_MARKER_SIZE = 2.0
MAX_MARKER_SIZE = 12.0

MEASURE: Literal["mean", "median"] = "mean"

# Highlighting of the architecture chosen by select_default_size. The guides
# in the scatter plot take the colour of its hidden-layer count
SELECTED_CELL_COLOR = "white"
SELECTED_CELL_LINEWIDTH = 0.8
# Length of each arm of the corner brackets, as a fraction of the cell side
SELECTED_CELL_CORNER_ARM = 0.15


def layer_colors(layers: Iterable[int]) -> dict[int, np.ndarray]:
    """Assign a colour to each hidden-layer count."""
    unique_layers = sorted({int(layer) for layer in layers})
    colors = LAYER_CMAP(np.linspace(0, 0.9, len(unique_layers)))

    return {layer: colors[index] for index, layer in enumerate(unique_layers)}


def neuron_sizes(neurons: Iterable[int]) -> dict[int, float]:
    """Assign marker areas across the available neuron counts."""
    unique_neurons = sorted({int(neuron) for neuron in neurons})

    if unique_neurons[-1] == unique_neurons[0]:
        return dict.fromkeys(unique_neurons, MAX_MARKER_SIZE)

    return {
        neuron: MIN_MARKER_SIZE
        + (MAX_MARKER_SIZE - MIN_MARKER_SIZE)
        * (neuron - unique_neurons[0])
        / (unique_neurons[-1] - unique_neurons[0])
        for neuron in unique_neurons
    }


def group_by_layer_neurons(
    results: dict[str, TestError],
) -> dict[tuple[int, int], list[TestError]]:
    """Group model errors by layer and neuron count."""
    grouped: dict[tuple[int, int], list[TestError]] = {}

    for name, result in results.items():
        layers = model_int(name, "layers")
        neurons = model_int(name, "neurons")
        key = (layers, neurons)
        grouped.setdefault(key, []).append(result)

    return grouped


def error_grid(
    results: dict[str, TestError],
    n_repeats: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Arrange test errors on the layer-by-neuron grid."""
    grouped = group_by_layer_neurons(results)

    layers_vals = sorted({layers for layers, _ in grouped})
    neurons_vals = sorted({neurons for _, neurons in grouped})

    grid = np.full(
        (len(layers_vals), len(neurons_vals)),
        np.nan,
    )

    reducer = np.mean if MEASURE == "mean" else np.median

    for (layers, neurons), cell_results in grouped.items():
        if len(cell_results) != n_repeats:
            raise ValueError(
                f"({layers}, {neurons}) has {len(cell_results)} results, "
                f"expected {n_repeats}."
            )

        layer_index = layers_vals.index(layers)
        neuron_index = neurons_vals.index(neurons)

        grid[layer_index, neuron_index] = float(
            reducer([result.mean_error for result in cell_results])
        )

    return grid, layers_vals, neurons_vals


def draw_error_grid(
    ax: Axes,
    grid: np.ndarray,
    layers_vals: list[int],
    neurons_vals: list[int],
    selected: tuple[int, int],
) -> AxesImage:
    """Draw the test-error grid as a heatmap, outlining the selected cell."""
    image = ax.imshow(
        grid,
        origin="lower",
        cmap=ERROR_CMAP,
        norm=LogNorm(
            vmin=ERROR_MIN,
            vmax=ERROR_MAX,
        ),
        aspect="auto",
    )

    # Corner brackets around the selected cell. Cells are one data unit wide
    selected_layers, selected_neurons = selected
    x0 = neurons_vals.index(selected_neurons) - 0.5
    y0 = layers_vals.index(selected_layers) - 0.5
    arm = SELECTED_CELL_CORNER_ARM
    for x, dx in ((x0, arm), (x0 + 1.0, -arm)):
        for y, dy in ((y0, arm), (y0 + 1.0, -arm)):
            ax.plot(
                [x + dx, x, x],
                [y, y, y + dy],
                color=SELECTED_CELL_COLOR,
                linewidth=SELECTED_CELL_LINEWIDTH,
                solid_capstyle="butt",
                solid_joinstyle="miter",
                clip_on=False,
                zorder=3.0,
            )

    ax.minorticks_off()
    ax.set_xticks(
        np.arange(len(neurons_vals)),
        [str(neurons) for neurons in neurons_vals],
    )
    ax.set_yticks(
        np.arange(len(layers_vals)),
        [str(layers) for layers in layers_vals],
    )

    ax.set_xlabel(style.NEURONS_LABEL)
    ax.set_ylabel(style.LAYERS_LABEL)

    return image


def draw_error_vs_params(
    ax: Axes,
    results: dict[str, TestError],
    n_repeats: int,
    selection: SizeSelection,
) -> tuple[ScalarMappable, np.ndarray]:
    """Plot test error against parameter count.

    Dashed guides run from both axes to the selected architecture's point.
    """
    grouped = group_by_layer_neurons(results)

    unique_layers = np.array(sorted({layers for layers, _ in grouped}))
    unique_neurons = np.array(sorted({neurons for _, neurons in grouped}))

    size_map = neuron_sizes(unique_neurons)
    color_map = layer_colors(unique_layers)
    colors = [color_map[int(layer)] for layer in unique_layers]

    for index, layer in enumerate(unique_layers):
        neurons_here = sorted(neurons for layers, neurons in grouped if layers == layer)

        n_params = [
            grouped[(layer, neurons)][0].n_parameters for neurons in neurons_here
        ]

        errors = [
            [result.mean_error for result in grouped[(layer, neurons)]]
            for neurons in neurons_here
        ]

        order = np.argsort(n_params)

        draw_error_line(
            ax,
            np.array(n_params)[order],
            [errors[i] for i in order],
            n_repeats,
            color=colors[index],
            measure=MEASURE,
            show_range=False,
            marker_size=[
                size_map[neurons] for neurons in np.array(neurons_here)[order]
            ],
            zorder=float(index + 1),
        )

    bounds = np.array(
        [layer - 0.5 for layer in unique_layers] + [unique_layers[-1] + 0.5]
    )

    mappable = ScalarMappable(
        norm=BoundaryNorm(bounds, len(unique_layers)), cmap=ListedColormap(colors)
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ERROR_MIN, ERROR_MAX)

    # Dashed guides from the axes, stopping at the selected point
    selected_layers, _ = selection.chosen
    selected_results = grouped[selection.chosen]
    reducer = np.mean if MEASURE == "mean" else np.median
    x_selected = selected_results[0].n_parameters
    y_selected = reducer([result.mean_error for result in selected_results])
    x_lim = ax.get_xlim()
    guide_style = {
        "color": color_map[selected_layers],
        "linestyle": "--",
        "zorder": 0.5,
    }
    ax.vlines(x_selected, ERROR_MIN, y_selected, **guide_style)
    ax.hlines(y_selected, x_lim[0], x_selected, **guide_style)
    ax.set_xlim(x_lim)

    add_size_legend(
        ax,
        unique_neurons,
        size_map,
        y_bottom=float(y_selected),
    )

    ax.set_xlabel(style.PARAMS_LABEL)
    ax.set_ylabel(rf"{MEASURE.capitalize()} {style.L2_RELATIVE_ERROR_LABEL}")
    ax.yaxis.labelpad = -13.0

    return mappable, unique_layers


def add_size_legend(
    ax: Axes,
    unique_neurons: np.ndarray,
    size_map: dict[int, float],
    y_bottom: float,
    ncol: int = 4,
) -> None:
    """Add a legend showing marker size versus neuron count.

    The legend sits in the left corner, just above the data value y_bottom.
    """
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=style.MARKER_EDGE_WIDTH,
            markersize=np.sqrt(size_map[int(neurons)]),
            linestyle="None",
        )
        for neurons in unique_neurons
    ]

    labels = [f"{neurons}" for neurons in unique_neurons]

    nrow = int(np.ceil(len(handles) / ncol))
    order = [
        row * ncol + column
        for column in range(ncol)
        for row in range(nrow)
        if row * ncol + column < len(handles)
    ]

    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        title=style.NEURONS_LABEL,
        loc="lower left",
        bbox_to_anchor=(0.0, y_bottom),
        bbox_transform=ax.get_yaxis_transform(),
        ncol=ncol,
        handlelength=1,
        handletextpad=0.2,
        columnspacing=0.75,
    )


def plot_grid_figure(
    grid: np.ndarray,
    layers_vals: list[int],
    neurons_vals: list[int],
    selection: SizeSelection,
    out_path: Path,
) -> None:
    """Plot the architecture error grid."""
    panel_width = (
        style.TEXT_WIDTH
        - style.MARGINS["left"]
        - style.MARGINS["right"]
        - style.CBAR_GAP
        - style.CBAR_WIDTH
    )

    fig, axes, caxes = style.panel_grid(
        panel_width=panel_width,
        panel_height=panel_width / 1.4,
        colorbar="column",
    )

    image = draw_error_grid(
        axes[0][0],
        grid,
        layers_vals,
        neurons_vals,
        selection.chosen,
    )

    style.add_colorbar(
        fig,
        image,
        caxes[0],
        label=rf"{MEASURE.capitalize()} {style.L2_RELATIVE_ERROR_LABEL}",
    )

    style.save(fig, out_path)


def plot_scatter_figure(
    results: dict[str, TestError],
    n_repeats: int,
    selection: SizeSelection,
    out_path: Path,
) -> None:
    """Plot test error against parameter count."""
    panel_width = (
        style.TEXT_WIDTH
        - style.MARGINS["left"]
        - style.MARGINS["right"]
        - style.CBAR_GAP
        - style.CBAR_WIDTH
    )

    fig, axes, caxes = style.panel_grid(
        panel_width=panel_width,
        panel_height=panel_width / 1.4,
        colorbar="column",
    )

    mappable, unique_layers = draw_error_vs_params(
        axes[0][0],
        results,
        n_repeats,
        selection,
    )

    style.add_colorbar(
        fig,
        mappable,
        caxes[0],
        label=style.LAYERS_LABEL,
    )
    caxes[0].set_yticks(unique_layers)

    style.save(fig, out_path)


def plot_combined_figure(
    grid: np.ndarray,
    layers_vals: list[int],
    neurons_vals: list[int],
    results: dict[str, TestError],
    n_repeats: int,
    selection: SizeSelection,
    out_path: Path,
) -> None:
    """Plot the error grid and parameter-count plot side by side."""
    aspect_grid = 1.0
    aspect_scatter = 1.2
    hgap = 0.8
    margins = dict(style.MARGINS, top=0.25, right=0.45, bottom=0.45)

    available = (
        style.TEXT_WIDTH
        - margins["left"]
        - margins["right"]
        - hgap
        - 2 * (style.CBAR_GAP + style.CBAR_WIDTH)
    )

    panel_height = available / (aspect_grid + aspect_scatter)

    fig, axes, caxes = style.panel_grid(
        1,
        2,
        panel_width=[
            aspect_grid * panel_height,
            aspect_scatter * panel_height,
        ],
        panel_height=panel_height,
        hgap=hgap,
        margins=margins,
        colorbar="column",
    )

    image = draw_error_grid(
        axes[0][0],
        grid,
        layers_vals,
        neurons_vals,
        selection.chosen,
    )

    style.add_colorbar(
        fig,
        image,
        caxes[0],
    )

    style.panel_letter(
        axes[0][0],
        style.PANEL_LABELS[0],
    )

    mappable, unique_layers = draw_error_vs_params(
        axes[0][1],
        results,
        n_repeats,
        selection,
    )

    # The y label sits between the two tick-labelled axes and serves both
    axes[0][1].yaxis.labelpad = 2.2

    style.add_colorbar(
        fig,
        mappable,
        caxes[1],
        label=style.LAYERS_LABEL,
    )
    caxes[1].set_yticks(unique_layers)

    style.panel_letter(
        axes[0][1],
        style.PANEL_LABELS[1],
    )

    style.save(fig, out_path)


def main() -> None:
    n_batches = 10
    t_min = 21.0
    t_max = 29.0

    out_dir = SUITE_DIR
    param_dir = SUITE_DIR / "param_files"
    model_dir = SUITE_DIR / "models"

    simulation = str(suite_values(SUITE, "simulations")[0])
    test_file = sim_data_dir(simulation) / "test.hdf5"

    n_repeats = suite_n_repeats(SUITE)

    style.use_style()
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test = load_test_data(
        test_file,
        device,
        t_min=t_min,
        t_max=t_max,
    )

    results = model_errors(
        param_dir,
        model_dir,
        test,
        n_batches=n_batches,
    )

    grid, layers_vals, neurons_vals = error_grid(
        results,
        n_repeats,
    )

    selection = select_default_size(results)

    plot_grid_figure(
        grid,
        layers_vals,
        neurons_vals,
        selection,
        out_dir / "test_error_grid.pdf",
    )

    plot_scatter_figure(
        results,
        n_repeats,
        selection,
        out_dir / "test_error_vs_parameters.pdf",
    )

    plot_combined_figure(
        grid,
        layers_vals,
        neurons_vals,
        results,
        n_repeats,
        selection,
        out_dir / "test_error_combined.pdf",
    )

    best, chosen = selection.best, selection.chosen
    print(
        f"Best mean error {selection.best_error:.3e} at "
        f"layers={best[0]}, neurons={best[1]}"
    )
    print(f"Reference mean error {selection.reference_error:.3e}")
    print(
        f"Selected mean error {selection.chosen_error:.3e} at "
        f"layers={chosen[0]}, neurons={chosen[1]} "
        f"(threshold {selection.threshold:.3e})"
    )


if __name__ == "__main__":
    main()
