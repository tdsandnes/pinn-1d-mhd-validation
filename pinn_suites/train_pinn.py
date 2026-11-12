import argparse
import os
import sys
from pathlib import Path

# cuBLAS needs a fixed workspace for deterministic results.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pinninit
import torch
import yaml

# Exit code when SIGTERM stops training before it is done, which tells
# train_array_task.sh to resubmit the job
STOPPED_EXIT_CODE = 3


def main() -> None:
    # Error on any operation without a deterministic implementation,
    # rather than silently giving run to run differences
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param_file", type=Path, required=True, help="Path to YAML parameter file"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./models"),
        help="Directory to save models",
    )
    args = parser.parse_args()
    param_file = args.param_file.resolve()
    print(f"Parameter file: {param_file}")

    # The model is named by its parameter file
    model_name = yaml.safe_load(param_file.read_text()).get("model_name")
    if model_name is None:
        raise ValueError(f"No 'model_name' key found in {param_file}")

    # Create output directory and determine device
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pinn = pinninit.PINN.from_yaml(param_file, device)

    # Carry on from the restart checkpoint of a job that ran out of time
    restart_path = pinn.trainer.checkpoint_path
    if restart_path is not None and Path(restart_path).exists():
        print(f"Resuming from restart checkpoint: {restart_path}")
        pinn.load(restart_path)

    pinn.train()

    if pinn.trainer.stopped_by_sigterm:
        print(f"Training stopped before the end. Restart checkpoint: {restart_path}")
        sys.exit(STOPPED_EXIT_CODE)

    pinn.save(model_path)
    print(f"Model saved to: {model_path}")

    # The model file has everything the restart checkpoint had
    if restart_path is not None:
        Path(restart_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
