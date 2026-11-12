import argparse
from pathlib import Path

import numpy as np
import pinninit
import torch
from pinninit import CartesianCoords
from scipy.stats import qmc

from simulations.sampling_configs.sampling_dataclasses import (
    DataDatasets,
    PhysDatasets,
)
from simulations.sampling_io import load_sampling_config
from simulations.sim_io import SnapshotReader, sim_data_dir

Coords = dict[str, np.ndarray]
Fields = dict[str, np.ndarray]
Sample = tuple[str, Coords, Fields]  # (role, coords, fields)


class Sampler:
    def __init__(self, reader: SnapshotReader, rng: np.random.Generator) -> None:
        self.reader = reader
        self.rng = rng

    def sample_dataset(self, dataset: DataDatasets | PhysDatasets) -> list[Sample]:
        """Every (role, coords, fields) that a dataset config asks for."""
        # A dataset can set its own seed, otherwise it draws from the global rng
        rng = self.rng if dataset.seed is None else np.random.default_rng(dataset.seed)
        if isinstance(dataset, DataDatasets):
            return self.sample_data(dataset, rng)
        return self.sample_phys(dataset, rng)

    def collocation_coords(
        self, dataset: PhysDatasets, rng: np.random.Generator
    ) -> Coords:
        """Collocation points (x, t), placed by the dataset's specified method."""
        # Grid method with specified offsets
        if dataset.train_method == "grid":
            assert dataset.N_x is not None and dataset.N_t is not None
            x_lb, x_ub = dataset.x_bounds
            t_lb, t_ub = dataset.t_bounds
            x = (
                x_lb
                + (np.arange(dataset.N_x) + 0.5 + dataset.train_grid_offset_x)
                * (x_ub - x_lb)
                / dataset.N_x
            )
            t = (
                t_lb
                + (np.arange(dataset.N_t) + 0.5 + dataset.train_grid_offset_t)
                * (t_ub - t_lb)
                / dataset.N_t
            )
            t_grid, x_grid = np.meshgrid(t, x, indexing="ij")
            return {"x": x_grid.ravel(), "t": t_grid.ravel()}

        assert dataset.N is not None

        x_lb, x_ub = dataset.x_bounds
        t_lb, t_ub = dataset.t_bounds
        single_x = x_lb == x_ub
        dims = 1 if single_x else 2

        # Sobol sampling
        if dataset.train_method == "sobol":
            # Sobol works in powers of two, so draw the next one up
            sobol = qmc.Sobol(d=dims, scramble=True, seed=rng)
            unit_cube = sobol.random_base2(int(np.ceil(np.log2(dataset.N))))[
                : dataset.N
            ]
        # Latin hypercube sampling
        elif dataset.train_method == "latin_hypercube":
            unit_cube = qmc.LatinHypercube(d=dims, seed=rng).random(dataset.N)
        else:
            raise ValueError(f"Unknown train_method '{dataset.train_method}'.")

        # Scale to requested bounds
        if single_x:
            t = qmc.scale(unit_cube, [t_lb], [t_ub])[:, 0]
            x = np.full_like(t, x_lb)
        else:
            points = qmc.scale(unit_cube, [x_lb, t_lb], [x_ub, t_ub])
            x, t = points[:, 0], points[:, 1]
        return {"x": x, "t": t}

    def get_coords_fields(
        self, dataset: DataDatasets, idx_t: np.ndarray, idx_x: np.ndarray
    ) -> tuple[Coords, Fields]:
        """Returns (coords, fields) of the snapshots at the given indices.

        Both grids are snapshot points, so the values are read with no interpolation.
        """
        # Sample fields at the requested indices
        field_labels = ("b", "u") if dataset.include_u else ("b",)
        fields: Fields = {
            field: np.empty((len(idx_t), len(idx_x))) for field in field_labels
        }
        for row, index in enumerate(idx_t):
            snapshot = self.reader.load_fields(int(index), *field_labels)
            for field in field_labels:
                fields[field][row] = snapshot[field][idx_x]

        # Grid of coordinates at the requested indices
        t_grid, x_grid = np.meshgrid(
            self.reader.times[idx_t], self.reader.x[idx_x], indexing="ij"
        )
        coords: Coords = {"t": t_grid.ravel(), "x": x_grid.ravel()}
        return coords, {label: values.ravel() for label, values in fields.items()}

    def sample_data(
        self, dataset: DataDatasets, rng: np.random.Generator
    ) -> list[Sample]:
        """Sample training and validation points for a data loss term.

        A grid of snapshot points is drawn from which training and validation points
        are randomly selected.
        """
        # Get the snapshot indices of the requested bounds
        idx_t = self.reader.grid_indices("t", dataset.t_bounds, dataset.N_slices)
        idx_x = self.reader.grid_indices("x", dataset.x_bounds, dataset.N_x)

        # Get the coords and fields of the requested snapshots
        coords, fields = self.get_coords_fields(dataset, idx_t, idx_x)

        # Return test dataset
        if dataset.is_test:
            return [("test", coords, fields)]

        # Draw the requested training points. Validation gets what is left
        is_train = np.zeros(len(coords["t"]), dtype=bool)
        is_train[rng.choice(is_train.size, dataset.N_train, replace=False)] = True
        role_masks = [("train", is_train)]
        if dataset.N_val > 0:
            role_masks.append(("val", ~is_train))

        return [
            (
                role,
                {name: values[mask] for name, values in coords.items()},
                {name: values[mask] for name, values in fields.items()},
            )
            for role, mask in role_masks
        ]

    def sample_phys(
        self, dataset: PhysDatasets, rng: np.random.Generator
    ) -> list[Sample]:
        """Sample training and validation points for a physics loss term.

        Training and validation points are drawn as two separate datasets.
        """
        samples: list[Sample] = [("train", self.collocation_coords(dataset, rng), {})]
        if dataset.val_fraction > 0:
            samples.append(
                ("val", self.collocation_coords(dataset.validation(), rng), {})
            )
        return samples


def main() -> None:
    torch.set_default_dtype(torch.float64)

    # Get config files
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sampling_config", type=str, help="Filename of the sampling config file"
    )
    parser.add_argument(
        "sim_config_path",
        type=Path,
        help="Path to the sim_config.py file (absolute or relative to "
        "SIMULATION_BASE_DIR)",
    )
    args = parser.parse_args()

    sampling_cfg = load_sampling_config(args.sampling_config)

    # Snapshot reader
    reader = SnapshotReader(args.sim_config_path)
    sampler = Sampler(reader, sampling_cfg.rng)
    coord_system = CartesianCoords(coord_axes={"x"}, device=torch.device("cpu"))

    # Generate every dataset in the config
    writer = pinninit.DataSetWriter(coord_system, {"x", "t"})
    counts: dict[str, dict[str, int]] = {}
    for dataset in sampling_cfg.datasets:
        for role, coords, fields in sampler.sample_dataset(dataset):
            writer.add(dataset.label, role, coords, fields)
            counts.setdefault(dataset.label, {})[role] = len(coords["t"])

    for label, roles in counts.items():
        breakdown = ", ".join(f"{role}={n}" for role, n in roles.items())
        print(f"{label}: {breakdown}")

    out_path = sim_data_dir(args.sim_config_path) / sampling_cfg.output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
