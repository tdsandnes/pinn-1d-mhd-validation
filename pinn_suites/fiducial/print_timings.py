from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from plotting.plotting_utils import freq_label, model_int

SUITE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunTiming:
    """How long one PINN trained, read from its checkpoint."""

    iterations: int
    best_iteration: int | None
    train_hours: float


def load_timing(model_path: Path) -> RunTiming:
    """Iterations and wall-clock training time of one trained model.

    The wall-clock time is the time spent training, summed over every
    resubmission, so it leaves out time queued between them.
    """
    state = torch.load(model_path, map_location="cpu", weights_only=True)[
        "trainer_state_dict"
    ]
    return RunTiming(
        iterations=state["epoch"],
        best_iteration=state["best_model_epoch"],
        train_hours=state["train_time"] / 3600,
    )


def summary(values: Sequence[float], fmt: str) -> str:
    """Min, mean and max of a list of values."""
    return (
        f"min {np.min(values):{fmt}}, mean {np.mean(values):{fmt}}, "
        f"max {np.max(values):{fmt}}"
    )


def print_stats(label: str, timings: list[RunTiming]) -> None:
    """Statistics over a group of runs."""
    iterations = [t.iterations for t in timings]
    best = [t.best_iteration for t in timings if t.best_iteration is not None]
    hours = [t.train_hours for t in timings]
    seconds_per_1000 = [1000 * 3600 * t.train_hours / t.iterations for t in timings]

    print(f"{label} ({len(timings)} runs)")
    print(f"  Iterations:              {summary(iterations, '.0f')}")
    if best:
        print(f"  Best-model iteration:    {summary(best, '.0f')}")
    print(f"  Wall-clock time (h):     {summary(hours, '.2f')}")
    print(f"  Time per 1000 iters (s): {summary(seconds_per_1000, '.1f')}")


def main() -> None:
    model_dir = SUITE_DIR / "models"

    model_paths = sorted(model_dir.glob("*.pth"))
    if not model_paths:
        raise RuntimeError(f"No trained models found in {model_dir.resolve()}")

    timings_by_freq: dict[int, list[RunTiming]] = {}
    for path in model_paths:
        timing = load_timing(path)
        timings_by_freq.setdefault(model_int(path.stem, "freq"), []).append(timing)
        print(
            f"{path.stem}: {timing.iterations} iterations, "
            f"best model at {timing.best_iteration}, {timing.train_hours:.2f} h"
        )
    print()

    for freq, timings in sorted(timings_by_freq.items()):
        print_stats(freq_label(freq), timings)

    print_stats("All", [t for timings in timings_by_freq.values() for t in timings])


if __name__ == "__main__":
    main()
