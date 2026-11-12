from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

METHODS = ("grid", "sobol", "latin_hypercube")


@dataclass
class DataDatasets:
    """Data sampled from the simulation snapshots."""

    label: str
    x_bounds: tuple[float, float]
    t_bounds: tuple[float, float]
    N_x: int
    val_fraction: float = 0.0
    N_data: int | None = None  # Train + val points. Required unless is_test
    N_t: int | None = None  # Snapshot times to use. Required when is_test
    include_u: bool = True
    is_test: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.is_test and self.N_t is None:
            raise ValueError("'N_t' must be specified when is_test=True.")
        if not self.is_test and self.N_data is None:
            raise ValueError("'N_data' must be specified for training datasets.")
        # Currently make sure training data divides evenly into N_x slices to avoid
        # silent rounding.
        if not self.is_test:
            assert self.N_data is not None
            if self.N_data % self.N_x != 0:
                raise ValueError(
                    f"N_data={self.N_data} must be a multiple of N_x={self.N_x} "
                    f"for dataset '{self.label}'."
                )

    @property
    def N_slices(self) -> int:
        """How many snapshot times the dataset spans."""
        if self.is_test:
            assert self.N_t is not None
            return self.N_t
        assert self.N_data is not None
        return self.N_data // self.N_x

    @property
    def N_train(self) -> int:
        """Number of training points, rounded to the nearest integer."""
        if self.is_test:
            return 0
        assert self.N_data is not None
        return round(self.N_data / (1.0 + self.val_fraction))

    @property
    def N_val(self) -> int:
        """Points left for validation once the training points are drawn."""
        if self.is_test or self.val_fraction <= 0:
            return 0
        assert self.N_data is not None
        return self.N_data - self.N_train


@dataclass
class PhysDatasets:
    """Physics collocation points over (x_bounds, t_bounds). Train and val are
    sampled independently, each can use its own sampling method."""

    label: str
    x_bounds: tuple[float, float]
    t_bounds: tuple[float, float]
    val_fraction: float
    train_method: str = "grid"
    val_method: str = "sobol"
    N: int | None = None  # Used for sobol, latin_hypercube
    N_x: int | None = None  # Used only for grid method
    N_t: int | None = None  # Used only for grid method
    train_grid_offset_x: float = 0.0
    train_grid_offset_t: float = 0.0
    val_grid_offset_x: float = 0.0
    val_grid_offset_t: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.train_method = self.train_method.lower()
        self.val_method = self.val_method.lower()

        if self.train_method == "grid":
            if self.N_x is None or self.N_t is None:
                raise ValueError("train_method is 'grid' but N_x or N_t is missing.")
        elif self.N is None:
            raise ValueError(f"method '{self.train_method}' needs a point count N.")

        # Offsets shift grid points only
        for method, offsets in (
            (self.train_method, (self.train_grid_offset_x, self.train_grid_offset_t)),
            (self.val_method, (self.val_grid_offset_x, self.val_grid_offset_t)),
        ):
            if method not in METHODS:
                raise ValueError(
                    f"Unknown method '{method}'. Use one of {', '.join(METHODS)}."
                )
            if method == "grid":
                if not all(-0.5 <= offset <= 0.5 for offset in offsets):
                    raise ValueError("Grid offsets must lie in [-0.5, 0.5].")
            elif any(offsets):
                raise ValueError(f"Grid offsets cannot be used with method '{method}'.")

        # Sobol should use powers of two
        if self.train_method == "sobol" and self.N_train_points & (
            self.N_train_points - 1
        ):
            warnings.warn(
                f"Sobol '{self.label}' N={self.N_train_points} is not a power of two.",
                stacklevel=2,
            )

    @property
    def N_train_points(self) -> int:
        """How many points."""
        if self.train_method == "grid":
            assert self.N_x is not None and self.N_t is not None
            return self.N_x * self.N_t
        assert self.N is not None
        return self.N

    def validation(self) -> PhysDatasets:
        """The associated validation dataset."""
        points = max(1, round(self.N_train_points * self.val_fraction))
        N: int | None = points
        N_x: int | None = None
        N_t: int | None = None

        if self.val_method == "grid":
            N = None
            if self.train_method == "grid":
                assert self.N_x is not None and self.N_t is not None
                scale = np.sqrt(self.val_fraction)
                N_x = max(1, round(self.N_x * scale))
                N_t = max(1, round(self.N_t * scale))
            else:
                N_x = N_t = max(1, round(np.sqrt(points)))

        return PhysDatasets(
            label=self.label,
            x_bounds=self.x_bounds,
            t_bounds=self.t_bounds,
            val_fraction=0.0,
            train_method=self.val_method,
            N=N,
            N_x=N_x,
            N_t=N_t,
            train_grid_offset_x=self.val_grid_offset_x,
            train_grid_offset_t=self.val_grid_offset_t,
            seed=self.seed,
        )
