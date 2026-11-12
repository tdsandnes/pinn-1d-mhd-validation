import runpy
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SIMULATION_BASE_DIR = Path(__file__).resolve().parent


def resolve_sim_config_path(sim_config_relpath: str | Path) -> Path:
    """Resolves a relative config path relative to SIMULATION_BASE_DIR."""
    path = Path(sim_config_relpath)
    if path.is_absolute():
        return path
    return (SIMULATION_BASE_DIR / path).resolve()


def load_sim_config(sim_config_relpath: str | Path) -> Any:
    """Loads Config() from a simulation config file path."""
    config_path = resolve_sim_config_path(sim_config_relpath)
    if not config_path.is_file():
        raise SystemExit(f"No simulation config file found: {config_path}")
    return runpy.run_path(str(config_path))["Config"]()


def sim_data_dir(sim_config_relpath: str | Path) -> Path:
    """Simulation data directory in the sim_config, relative to SIMULATION_BASE_DIR."""
    config_path = resolve_sim_config_path(sim_config_relpath)
    config = load_sim_config(config_path)
    path = Path(config_path.parent / config.data_dir)
    if not path.is_dir():
        raise SystemExit(f"No simulation data directory: {path}")
    return path


def write_snapshot(
    path: Path,
    t: float,
    params: dict[str, Any],
    x: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    """Write a snapshot to an HDF5 file."""
    with h5py.File(path, "w") as f:
        # Write parameters, including the snapshot time t
        group = f.create_group("Parameters")
        for name, value in params.items():
            group.attrs[name] = value
        group.attrs["t"] = t

        # Write coordinates
        f.create_group("Coords").create_dataset("x", data=x)

        # Write fields
        field_group = f.create_group("Fields")
        for name, values in fields.items():
            field_group.create_dataset(name, data=values)


class SnapshotReader:
    """Reads a simulation's snapshots based on its sim_config_*.py."""

    def __init__(self, sim_config_relpath: str | Path) -> None:
        config_path = resolve_sim_config_path(sim_config_relpath)
        config = load_sim_config(config_path)
        snapshot_dir = config_path.parent / config.data_dir / config.snapshot_dir

        self.paths = sorted(snapshot_dir.glob("snapshot_*.hdf5"))
        self.times = np.asarray(config.snapshot_times, dtype=np.float64)

        if not self.paths:
            raise ValueError(f"No snapshots found in {snapshot_dir}")

        if len(self.paths) != len(self.times):
            raise ValueError(
                f"{snapshot_dir} holds {len(self.paths)} snapshots, but its sim "
                f"config describes {len(self.times)}."
            )

        with h5py.File(self.paths[0], "r") as snapshot:
            self.x = snapshot["Coords"]["x"][:]

    def _check_time(self, snapshot: h5py.File, index: int) -> None:
        """Raise if a snapshot's stored t differs from the sim config's."""
        t = float(snapshot["Parameters"].attrs["t"])
        if not np.isclose(t, self.times[index]):
            raise ValueError(
                f"{self.paths[index]} is at t={t:g}, but the sim config puts "
                f"that snapshot at t={self.times[index]:g}."
            )

    def load_fields(self, index: int, *names: str) -> dict[str, np.ndarray]:
        """The fields stored in a given snapshot, on the snapshot x grid."""
        with h5py.File(self.paths[index], "r") as snapshot:
            self._check_time(snapshot, index)
            return {name: snapshot["Fields"][name][:] for name in names}

    def load_fields_at_x(self, x: float, *names: str) -> dict[str, np.ndarray]:
        """Load each requested field at one x over all snapshot times."""
        idx_x = int(np.argmin(np.abs(self.x - x)))
        if not np.isclose(self.x[idx_x], x):
            raise ValueError(f"{x} is not a snapshot point.")

        series = {name: np.empty(len(self.paths), dtype=np.float64) for name in names}
        for index, path in enumerate(self.paths):
            with h5py.File(path, "r") as snapshot:
                self._check_time(snapshot, index)
                for name in names:
                    series[name][index] = snapshot["Fields"][name][idx_x]
        return series

    def grid_indices(
        self,
        label: str,
        bounds: tuple[float, float],
        N: int,
    ) -> np.ndarray:
        """Indices of N snapshot points spanning bounds."""

        if label == "x":
            values = self.x
        elif label == "t":
            values = self.times
        else:
            raise ValueError(f"'{label}' is not a coordinate. Expected 'x' or 't'.")

        # Check that the bounds are exact snapshot points and get their indices
        idx_first = int(np.argmin(np.abs(values - bounds[0])))
        if not np.isclose(values[idx_first], bounds[0]):
            raise ValueError(f"{bounds[0]} is not a snapshot {label} coordinate.")

        idx_last = int(np.argmin(np.abs(values - bounds[1])))
        if not np.isclose(values[idx_last], bounds[1]):
            raise ValueError(f"{bounds[1]} is not a snapshot {label} coordinate.")

        span = idx_last - idx_first

        if N == 1:
            if span != 0:
                raise ValueError(
                    f"N=1 was requested, but {bounds} are not one snapshot point."
                )
            return np.array([idx_first])

        # Check that the span can be evenly divided into N-1 steps
        step, remainder = divmod(span, N - 1)
        if remainder:
            raise ValueError(
                f"{bounds} span {span + 1} snapshot {label} points, which "
                f"{N} samples cannot divide evenly."
            )

        return idx_first + step * np.arange(N)
