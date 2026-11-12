#!/usr/bin/env bash
# Repo-wide settings.

# Conda environments
DEDALUS_ENV=dedalus3
PINNINIT_ENV=pinninit

# Conda install in the home directory, used if present
CONDA_SH="$HOME/software/miniforge3/etc/profile.d/conda.sh"

activate_conda() {
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
    elif ! command -v conda >/dev/null 2>&1; then
        # No version given, so the module system loads its default
        module load miniforge3 2>/dev/null || module load miniforge
    fi
}

# Put the repo and pinninit on the Python path, so absolute imports work from anywhere
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! PINNINIT_DIR="$(cd "$REPO_DIR/../pinninit" 2>/dev/null && pwd)"; then
    echo "pinninit not found at $REPO_DIR/../pinninit. Clone it next to this repo." >&2
    return 1 2>/dev/null || exit 1
fi
export PINNINIT_DIR
export PYTHONPATH="$REPO_DIR:$PINNINIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
