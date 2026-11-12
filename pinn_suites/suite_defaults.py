from pathlib import Path
from typing import Any

SUITES_DIR = Path(__file__).resolve().parent

CONSTANTS = ("Re", "Rm", "Ma")

# Default values of every quantity a suite can set in a suite.yml
SUITE_DEFAULTS: dict[str, Any] = {
    "template": "default",
    "simulations": "2freq_1Ma_500Re_500Rm/sim_config_2freq_1Ma_500Re_500Rm.py",
    "sampling_configs": "t20to30_eachcol032768sobol_data001025_bc016384",
    "repeats": 3,
    "layers": 7,
    "neurons": 80,
    "methods": ["rba", "w23", "siren", "soap"],
    "methods_config": {},
    "warmup_epochs": 10000,
    "n_batches": 1,
    "log_point_weight_info": False,
    "point_weight_snapshot_freq": 0,
    "checkpoint_freq": 10000,
    "learn": {},
}

# How each quantity appears in a model name, e.g. layers0006
TAG_FORMATS = {
    "freq": "freq{:02d}",
    "t_min": "tmin{:03d}",
    "t_max": "tmax{:03d}",
    "total_data_points": "data{:06d}",
    "collocation_points": "col{:06d}",
    "bc_points": "bc{:06d}",
    "layers": "layers{:04d}",
    "neurons": "neurons{:04d}",
    "repeat": "repeat{:02d}",
}

# How each quantity is read back out of a model name
TAG_PATTERNS = {key: fmt.split("{")[0] + r"(\d+)" for key, fmt in TAG_FORMATS.items()}
TAG_PATTERNS["method"] = r"method(\d+)"
