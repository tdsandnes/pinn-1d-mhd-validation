from pathlib import Path

import numpy as np
import torch
import yaml

from plotting.pinn_utils import snapshot_field_profiles
from plotting.plotting_utils import model_int
from simulations.sim_io import SnapshotReader

Profile = dict[str, np.ndarray]


def get_model(
    param_dir: Path,
    model_dir: Path,
    freq: int,
    repeat: int,
) -> tuple[Path, Path, str]:
    """The parameter file, model file and model name of one (freq, repeat) run."""
    for yaml_path in sorted(param_dir.glob("*.yml")):
        name = yaml.safe_load(yaml_path.read_text())["model_name"]
        if model_int(name, "freq") != freq or model_int(name, "repeat") != repeat:
            continue

        model_path = model_dir / f"{name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return yaml_path, model_path, name

    raise RuntimeError(f"No model for freq={freq} repeat={repeat} in {param_dir}")


def repeat_profiles(
    param_dir: Path,
    model_dir: Path,
    reader: SnapshotReader,
    freq: int,
    times: list[float],
    repeat: int,
    band_repeats: list[int],
    n_batches: int,
    device: torch.device,
) -> dict[float, Profile]:
    """One repeat's field profiles at each time, with the min/max over band_repeats."""
    by_repeat: dict[int, dict[float, Profile]] = {}
    for r in sorted({repeat, *band_repeats}):
        yaml_path, model_path, name = get_model(param_dir, model_dir, freq, r)
        print(f"freq={freq} repeat={r}: {name}")
        by_repeat[r] = {
            time: snapshot_field_profiles(
                yaml_path, model_path, reader, time, n_batches, device
            )
            for time in times
        }

    profiles: dict[float, Profile] = {}
    for time in times:
        profile = dict(by_repeat[repeat][time])
        if band_repeats:
            for field in ("u", "b"):
                preds = np.stack(
                    [by_repeat[r][time][f"{field}_pred"] for r in band_repeats]
                )
                profile[f"{field}_min"] = preds.min(axis=0)
                profile[f"{field}_max"] = preds.max(axis=0)
        profiles[time] = profile
    return profiles
