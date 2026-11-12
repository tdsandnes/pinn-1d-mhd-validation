#!/usr/bin/env bash
#
# Run simulations and sample datasets.
#
# Choose simulations and sampling schemes in run_all_config.sh.

set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SIM_DIR/run_all_config.sh"
source "$SIM_DIR/../config.sh"

# Check every sampling group exists before simulating
for group in "${SIM_CONFIGS[@]}"; do
    if ! declare -p "$group" &>/dev/null; then
        echo "Unknown sampling group '$group' in run_all_config.sh" >&2
        exit 1
    fi
done

# Generate sampling configs
bash "$SIM_DIR/sampling_configs/generate_configs.sh"

# Sort for a predictable run order
mapfile -t sim_configs < <(printf '%s\n' "${!SIM_CONFIGS[@]}" | sort)

for sim_config in "${sim_configs[@]}"; do
    cfg="$SIM_DIR/$sim_config"

    echo "simulate $sim_config"
    conda run --no-capture-output -n "$DEDALUS_ENV" python3 -u "$SIM_DIR/simulate.py" "$cfg"

    declare -n sampling_configs="${SIM_CONFIGS[$sim_config]}"
    for entry in "${sampling_configs[@]}"; do
        # An entry is either a .py file or a directory of repeat .py files
        path="$SIM_DIR/sampling_configs/$entry"
        if [[ -d "$path" ]]; then
            files=("$path"/*.py)
        else
            files=("$path")
        fi
        if [[ ! -f "${files[0]}" ]]; then
            echo "No sampling config at $path (run sampling_configs/generate_configs.sh?)" >&2
            exit 1
        fi

        for sampling_cfg in "${files[@]}"; do
            echo "sample $sim_config with ${sampling_cfg#"$SIM_DIR/sampling_configs/"}"
            conda run --no-capture-output -n "$PINNINIT_ENV" python3 -u "$SIM_DIR/sample_data.py" \
                "$sampling_cfg" "$cfg"
        done
    done
done
