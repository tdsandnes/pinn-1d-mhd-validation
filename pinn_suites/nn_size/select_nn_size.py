from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from plotting.evaluate_test_dataset import TestError, load_test_data, model_errors
from plotting.plotting_utils import model_int, suite_values
from simulations.sim_io import sim_data_dir

SUITE_DIR = Path(__file__).resolve().parent

SUITE = SUITE_DIR.name

# Accepted architectures lie within REL_TOL of the reference error, the mean
# over the N_BEST architectures with the lowest mean errors
REL_TOL = 0.1
N_BEST = 10


@dataclass(frozen=True)
class SizeSelection:
    """The chosen and best architectures, as (layers, neurons)."""

    chosen: tuple[int, int]
    chosen_error: float
    best: tuple[int, int]
    best_error: float
    reference_error: float
    threshold: float


def select_default_size(
    results: dict[str, TestError],
    rel_tol: float = REL_TOL,
    n_best: int = N_BEST,
) -> SizeSelection:
    """The smallest architecture whose error is within rel_tol of the reference.

    Errors are averaged over the repeats of each (layers, neurons) pair. The
    reference error is the mean of the n_best lowest of these mean errors. The
    chosen architecture has the fewest parameters among those with a mean
    error at most (1 + rel_tol) times the reference error.
    """
    errors: dict[tuple[int, int], list[float]] = {}
    n_params: dict[tuple[int, int], int] = {}
    for name, result in results.items():
        key = (model_int(name, "layers"), model_int(name, "neurons"))
        errors.setdefault(key, []).append(result.mean_error)
        n_params[key] = result.n_parameters

    mean_errors = {key: float(np.mean(errs)) for key, errs in errors.items()}

    ranked = sorted(mean_errors, key=mean_errors.__getitem__)
    best = ranked[0]
    reference_error = float(np.mean([mean_errors[key] for key in ranked[:n_best]]))
    threshold = (1.0 + rel_tol) * reference_error

    candidates = [key for key, err in mean_errors.items() if err <= threshold]
    chosen = min(candidates, key=lambda key: (n_params[key], mean_errors[key]))

    return SizeSelection(
        chosen=chosen,
        chosen_error=mean_errors[chosen],
        best=best,
        best_error=mean_errors[best],
        reference_error=reference_error,
        threshold=threshold,
    )


def main() -> None:
    n_batches = 10
    t_min = 21.0
    t_max = 29.0

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    simulation = str(suite_values(SUITE, "simulations")[0])
    test = load_test_data(
        sim_data_dir(simulation) / "test.hdf5",
        device,
        t_min=t_min,
        t_max=t_max,
    )
    results = model_errors(
        SUITE_DIR / "param_files",
        SUITE_DIR / "models",
        test,
        n_batches=n_batches,
    )

    selection = select_default_size(results)
    best, chosen = selection.best, selection.chosen

    print(
        f"Best:   layers={best[0]}, neurons={best[1]}, error={selection.best_error:.3e}"
    )
    print(f"Reference error (mean of best {N_BEST}): {selection.reference_error:.3e}")
    print(
        f"Chosen: layers={chosen[0]}, neurons={chosen[1]}, "
        f"error={selection.chosen_error:.3e} "
        f"(smallest within {REL_TOL:.0%} of reference, threshold "
        f"{selection.threshold:.3e})"
    )


if __name__ == "__main__":
    main()
