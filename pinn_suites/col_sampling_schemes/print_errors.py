import re
from pathlib import Path

import numpy as np
import torch

from plotting.evaluate_test_dataset import TestError, load_test_data, model_errors
from plotting.plotting_utils import frequency_filter, simulation_freq, suite_values
from simulations.sim_io import sim_data_dir

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name
SCHEMES = ("sobol", "latinhypercube", "gridaligned", "gridoffset")


def col_scheme(model_name: str) -> str:
    """A model's collocation scheme. Sobol is the default, so has no tag."""
    match = re.search(r"colscheme-([a-z]+)", model_name)
    return match.group(1) if match else "sobol"


def main() -> None:
    n_batches = 10
    t_min = 21.0
    t_max = 29.0

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test errors of every model, grouped by frequency case and scheme
    errors: dict[int, dict[str, list[TestError]]] = {}
    for simulation in suite_values(SUITE, "simulations"):
        freq = simulation_freq(str(simulation))
        test = load_test_data(
            sim_data_dir(str(simulation)) / "test.hdf5",
            device,
            t_min=t_min,
            t_max=t_max,
        )
        results = model_errors(
            SUITE_DIR / "param_files",
            SUITE_DIR / "models",
            test,
            n_batches=n_batches,
            model_filter=frequency_filter(freq),
        )
        for name, result in results.items():
            errors.setdefault(freq, {}).setdefault(col_scheme(name), []).append(result)

    # Mean over the repeats
    print(f"\n{'freq':<6}{'scheme':<16}{'repeats':<9}{'u':<11}{'b':<11}{'mean'}")
    for freq, by_scheme in sorted(errors.items()):
        for scheme in SCHEMES:
            repeats = by_scheme.get(scheme, [])
            if not repeats:
                continue
            u = np.mean([r.error_u for r in repeats])
            b = np.mean([r.error_b for r in repeats])
            mean = np.mean([r.mean_error for r in repeats])
            print(
                f"{freq:<6}{scheme:<16}{len(repeats):<9}{u:<11.3e}{b:<11.3e}{mean:.3e}"
            )


if __name__ == "__main__":
    main()
