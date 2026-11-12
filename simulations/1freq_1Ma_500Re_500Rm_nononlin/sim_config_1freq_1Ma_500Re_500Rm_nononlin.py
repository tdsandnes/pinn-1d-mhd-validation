from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    # Physical parameters
    Re: float = 500.0  # Reynolds
    Rm: float = 500.0  # Magnetic Reynolds
    Ma: float = 1.0  # Alfven Mach
    with_nonlin_term: bool = False

    # Domain parameters
    T: float = 30.0
    L: float = 1.0

    # Numerical parameters
    N: int = 256
    dealias: float = 3 / 2
    dtype: type = np.float64
    dt: float = 1 / 1024

    # Snapshot parameters
    N_snapshots_per_t: int = 1024
    data_dir: str = "data"
    snapshot_dir: str = "snapshots"
    output_dx: float = 1 / 256

    @property
    def N_snapshots(self) -> int:
        return int(np.ceil(self.N_snapshots_per_t * self.T))

    @property
    def snapshot_times(self) -> tuple[float, ...]:
        return tuple(np.linspace(0, self.T, self.N_snapshots + 1))

    # Initial conditions
    @staticmethod
    def zp_initial(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    @staticmethod
    def zm_initial(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    # Forcing
    FREQS: tuple[float, ...] = (np.sqrt(2) / 2,)

    def zm_in_terms(self, t: float) -> tuple[float, ...]:
        amp = np.sqrt(1.0 / len(self.FREQS))
        return tuple(amp * np.sin(2 * np.pi * f * t) for f in self.FREQS)

    def zm_in(self, t: float) -> float:
        return sum(self.zm_in_terms(t))

    def __post_init__(self) -> None:
        steps_per_snapshot = self.T / self.N_snapshots / self.dt
        if not np.isclose(steps_per_snapshot, round(steps_per_snapshot)):
            raise ValueError(
                f"Snapshot interval ({self.T / self.N_snapshots:g}) is not a "
                f"multiple of dt ({self.dt:g}): "
                f"{steps_per_snapshot:g} steps per snapshot"
            )
