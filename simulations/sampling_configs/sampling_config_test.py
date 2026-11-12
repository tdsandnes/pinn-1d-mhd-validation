from dataclasses import dataclass, field

import numpy as np

from simulations.sampling_configs.sampling_dataclasses import DataDatasets


@dataclass
class Config:
    output_filename: str = "test.hdf5"
    t_bounds: tuple[float, float] = (20.0, 30.0)
    x_bounds: tuple[float, float] = (0.0, 1.0)
    seed: int = 0
    N_x_test: int = 257
    N_t_test: int = 257

    datasets: list[DataDatasets] = field(init=False)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.datasets = [
            DataDatasets(
                label="test",
                N_t=self.N_t_test,
                x_bounds=self.x_bounds,
                t_bounds=self.t_bounds,
                N_x=self.N_x_test,
                include_u=True,
                is_test=True,
            ),
        ]
