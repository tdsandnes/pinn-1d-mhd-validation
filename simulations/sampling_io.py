import runpy
from pathlib import Path
from typing import Any

SAMPLING_CONFIGS_DIR = Path(__file__).resolve().parent / "sampling_configs"


def resolve_sampling_config_path(sampling_config_relpath: str | Path) -> Path:
    """Resolves a relative config path relative to SAMPLING_CONFIGS_DIR."""
    path = Path(sampling_config_relpath)
    if path.is_absolute():
        return path
    return (SAMPLING_CONFIGS_DIR / path).resolve()


def load_sampling_config(sampling_config_relpath: str | Path) -> Any:
    """Loads Config() from a sampling config file path."""
    config_path = resolve_sampling_config_path(sampling_config_relpath)
    if not config_path.is_file():
        raise SystemExit(f"No sampling config file found: {config_path}")
    return runpy.run_path(str(config_path))["Config"]()
