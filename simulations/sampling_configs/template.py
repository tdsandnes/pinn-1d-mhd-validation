from dataclasses import dataclass, field

import numpy as np

from simulations.sampling_configs.sampling_dataclasses import DataDatasets, PhysDatasets

# Filled in by generate_configs.sh
BASENAME = "template"
REPEAT = 0
SEED = 0
T_MIN = 20.0
T_MAX = 30.0
N_DATA = 1025
N_PHYS = 32768
N_BC = 16384
N_X_PHYS = 1  # only used by the grid method
N_T_PHYS = 1  # only used by the grid method
TRAIN_METHOD = "sobol"
VAL_METHOD = "sobol"
ZP_OFFSET_X = 0.0  # only used by the grid method
ZP_OFFSET_T = 0.0
ZM_OFFSET_X = 0.0
ZM_OFFSET_T = 0.0


@dataclass
class Config:
    output_filename: str = f"train_val_{BASENAME}_repeat{REPEAT:02d}.hdf5"
    repeat: int = REPEAT

    t_bounds: tuple[float, float] = (T_MIN, T_MAX)
    x_bounds: tuple[float, float] = (0.0, 1.0)

    seed: int = SEED

    N_data: int = N_DATA
    N_phys: int = N_PHYS
    N_bc: int = N_BC
    N_x_phys: int = N_X_PHYS
    N_t_phys: int = N_T_PHYS

    train_method: str = TRAIN_METHOD
    val_method: str = VAL_METHOD
    zp_offset_x: float = ZP_OFFSET_X
    zp_offset_t: float = ZP_OFFSET_T
    zm_offset_x: float = ZM_OFFSET_X
    zm_offset_t: float = ZM_OFFSET_T

    data_val_fraction: float = 0.25
    phys_val_fraction: float = 0.25
    bc_val_fraction: float = 0.25

    datasets: list[DataDatasets | PhysDatasets] = field(init=False)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.datasets = [
            DataDatasets(
                label="data_loss",
                N_data=self.N_data,
                val_fraction=self.data_val_fraction,
                x_bounds=(self.x_bounds[1], self.x_bounds[1]),
                t_bounds=self.t_bounds,
                N_x=1,
                include_u=False,
            ),
            PhysDatasets(
                label="dbdx_bc_loss",
                N=self.N_bc,
                x_bounds=(self.x_bounds[1], self.x_bounds[1]),
                t_bounds=self.t_bounds,
                val_fraction=self.bc_val_fraction,
                train_method="sobol",
                val_method="sobol",
            ),
            PhysDatasets(
                label="phys_zp_loss",
                N=self.N_phys,
                N_x=self.N_x_phys,
                N_t=self.N_t_phys,
                x_bounds=self.x_bounds,
                t_bounds=self.t_bounds,
                val_fraction=self.phys_val_fraction,
                train_method=self.train_method,
                val_method=self.val_method,
                train_grid_offset_x=self.zp_offset_x,
                train_grid_offset_t=self.zp_offset_t,
            ),
            PhysDatasets(
                label="phys_zm_loss",
                N=self.N_phys,
                N_x=self.N_x_phys,
                N_t=self.N_t_phys,
                x_bounds=self.x_bounds,
                t_bounds=self.t_bounds,
                val_fraction=self.phys_val_fraction,
                train_method=self.train_method,
                val_method=self.val_method,
                train_grid_offset_x=self.zm_offset_x,
                train_grid_offset_t=self.zm_offset_t,
            ),
        ]
