import itertools
import sys
from pathlib import Path
from string import Template
from typing import Any

import yaml

from pinn_methods.method_settings import method_settings, method_str
from pinn_suites.suite_defaults import (
    CONSTANTS,
    SUITE_DEFAULTS,
    SUITES_DIR,
    TAG_FORMATS,
)
from simulations.sampling_io import (
    SAMPLING_CONFIGS_DIR,
    load_sampling_config,
)
from simulations.sim_io import load_sim_config, sim_data_dir

REPO_DIR = SUITES_DIR.parent
TEMPLATES_DIR = REPO_DIR / "pinn_templates"


def collocation_scheme(cfg: Any) -> str:
    """How the collocation points were placed, e.g. latinhypercube or gridoffset."""
    if cfg.train_method != "grid":
        return str(cfg.train_method).replace("_", "")
    offsets = (cfg.zp_offset_x, cfg.zp_offset_t, cfg.zm_offset_x, cfg.zm_offset_t)
    return "gridoffset" if any(offsets) else "gridaligned"


def model_tag(key: str, value: Any) -> str:
    """One quantity as it appears in a model name, e.g. layers0006."""
    if key == "simulations":
        # A simulation names the model by the number of forcing frequencies
        config = load_sim_config(str(value))
        return TAG_FORMATS["freq"].format(len(config.FREQS))
    if key == "sampling_configs":
        # A sampling config names the model by the data set it sampled
        cfg = load_sampling_config(str(value))
        tags = [
            TAG_FORMATS["t_min"].format(int(cfg.t_bounds[0])),
            TAG_FORMATS["t_max"].format(int(cfg.t_bounds[1])),
            TAG_FORMATS["collocation_points"].format(int(cfg.N_phys)),
            TAG_FORMATS["total_data_points"].format(int(cfg.N_data)),
            TAG_FORMATS["bc_points"].format(int(cfg.N_bc)),
            TAG_FORMATS["repeat"].format(int(cfg.repeat)),
        ]
        # Sobol is the default, so only other collocation methods are tagged
        if cfg.train_method != "sobol":
            tags.insert(3, f"colscheme-{collocation_scheme(cfg)}")
        return "_".join(tags)
    if key == "methods":
        return f"method{method_str(value)}"
    if key == "learn":
        # A float, so the decimal point becomes 'p', e.g. {Ma: 0.25} -> ma0p25
        return "_".join(
            f"{name.lower()}{float(start):g}".replace(".", "p")
            for name, start in value.items()
        )
    if key in TAG_FORMATS:
        return TAG_FORMATS[key].format(int(value))
    return f"{key}{str(value).replace('_', '-')}"


def constants(sim_config_path: str, learn: dict[str, float]) -> dict[str, str]:
    """The param file's Constants section."""
    config = load_sim_config(sim_config_path)
    unknown = learn.keys() - set(CONSTANTS)
    if unknown:
        raise SystemExit(
            f"Cannot train {', '.join(sorted(unknown))}. Use {', '.join(CONSTANTS)}"
        )

    fields = {}
    for name in CONSTANTS:
        if not hasattr(config, name):
            raise SystemExit(f"Could not find {name} in the {sim_config_path} config")
        fields[name.upper()] = str(learn.get(name, getattr(config, name)))
        fields[f"{name.upper()}_LEARNABLE"] = (
            "\n    learnable: true" if name in learn else ""
        )
    return fields


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <suite>")

    suite_dir = SUITES_DIR / sys.argv[1]
    out_dir = suite_dir / "param_files"

    # The suite file specifies the parameter sweep
    suite_file = suite_dir / "suite.yml"
    if not suite_file.is_file():
        raise SystemExit(f"No such suite: {suite_file}")
    suite: dict[str, Any] = yaml.safe_load(suite_file.read_text())

    if "base_model_name" not in suite:
        raise SystemExit(f"{suite_file} is missing 'base_model_name'.")
    unknown = suite.keys() - SUITE_DEFAULTS.keys() - {"base_model_name"}
    if unknown:
        raise SystemExit(
            f"{suite_file} sets unknown quantity: {', '.join(sorted(unknown))}. "
            f"Pick from: {', '.join(sorted(SUITE_DEFAULTS))}"
        )

    # Template that every parameter file is filled in from
    template_name = suite.pop("template", SUITE_DEFAULTS["template"])
    template_path = TEMPLATES_DIR / f"{template_name}.yml"
    if not template_path.is_file():
        raise SystemExit(f"No template {template_path}")
    template = Template(template_path.read_text())
    equation_file = REPO_DIR / "equations" / "pinn_equations.py"

    # Start clean
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("param_*.yml"):
        old_file.unlink()

    # Restart checkpoints are only valid for the parameter files they came from
    restart_dir = suite_dir / "restarts"
    for old_file in restart_dir.glob("*.pth"):
        old_file.unlink()

    # A flat list of methods is one combination, not a sweep over single methods
    methods = suite.get("methods")
    if isinstance(methods, list) and all(isinstance(m, str) for m in methods):
        suite["methods"] = [methods]

    # Expand sampling config directories into the requested repeats
    entries = suite.get("sampling_configs", SUITE_DEFAULTS["sampling_configs"])
    if not isinstance(entries, list):
        entries = [entries]

    # repeats: N keeps repeat numbers 1, ..., N, repeats: [...] keeps those listed
    requested_repeats = suite.pop("repeats", SUITE_DEFAULTS["repeats"])
    wanted = (
        requested_repeats
        if isinstance(requested_repeats, list)
        else list(range(1, int(requested_repeats) + 1))
    )

    expanded: list[str] = []
    for entry in entries:
        entry_path = SAMPLING_CONFIGS_DIR / entry

        # Individual config files are kept unchanged
        if not entry_path.is_dir():
            expanded.append(entry)
            continue

        files = sorted(entry_path.glob("*.py"))
        by_repeat = {load_sampling_config(f).repeat: f for f in files}
        missing = [r for r in wanted if r not in by_repeat]
        if missing:
            raise SystemExit(
                f"{entry}: repeat(s) {missing} not found. Have: {sorted(by_repeat)}"
            )
        expanded.extend(
            str(by_repeat[r].relative_to(SAMPLING_CONFIGS_DIR)) for r in wanted
        )
    suite["sampling_configs"] = expanded

    # Sweep every combination. Quantities given as lists define the model grid
    fields = [key for key in suite if key != "base_model_name"]
    values: list[list[Any]] = []
    for key in fields:
        value = suite[key]
        values.append(value if isinstance(value, list) else [value])
    name_fields = [key for key in fields if isinstance(suite[key], list)]

    model_names = set()
    for combination in itertools.product(*values):
        # Fill unspecified quantities from the suite defaults
        run = {**SUITE_DEFAULTS, **dict(zip(fields, combination, strict=True))}
        name_tags = [model_tag(key, run[key]) for key in name_fields]
        model_name = "_".join([str(suite["base_model_name"]), *name_tags])

        # Two combinations sharing a name would silently overwrite each other
        if model_name in model_names:
            raise SystemExit(f"Duplicate model name: {model_name}")
        model_names.add(model_name)

        # Resolve the simulation and sampling configs used for this model
        sim_config_path = str(run["simulations"])
        sampling = load_sampling_config(str(run["sampling_configs"]))

        # The sampling config names the file sample_data.py wrote
        data_file = sim_data_dir(sim_config_path) / sampling.output_filename
        if not data_file.is_file():
            raise SystemExit(f"Missing data file: {data_file}")

        layers = "[" + ", ".join([str(run["neurons"])] * int(run["layers"])) + "]"

        # Fill the template with the resolved settings for this model.
        # Use the sampling config's seed for the torch seed.
        param_file = template.substitute(
            MODEL_NAME=model_name,
            SEED=int(sampling.seed),
            DATA_FILE=str(data_file),
            EQN_FILE=str(equation_file),
            T_MIN=float(sampling.t_bounds[0]),
            T_MAX=float(sampling.t_bounds[1]),
            WARMUP_EPOCHS=int(run["warmup_epochs"]),
            N_BATCHES=int(run["n_batches"]),
            LOG_POINT_WEIGHT_INFO=str(run["log_point_weight_info"]).lower(),
            POINT_WEIGHT_SNAPSHOT_FREQ=int(run["point_weight_snapshot_freq"]),
            RESTART_FILE=str(restart_dir / f"{model_name}.pth"),
            CHECKPOINT_FREQ=int(run["checkpoint_freq"]),
            **constants(sim_config_path, run["learn"]),
            **method_settings(
                run["methods"],
                run["methods_config"],
                hidden_layer_widths=layers,
            ),
        )
        (out_dir / f"param_{model_name}.yml").write_text(param_file)

    print(f"Generated {len(model_names)} parameter files in {out_dir}/")


if __name__ == "__main__":
    main()
