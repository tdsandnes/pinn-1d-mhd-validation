from pathlib import Path

import numpy as np
import pinninit
import torch
from pinninit import CartesianCoords, Coords, Operators, make_coords_dataclass

from equations.pinn_equations import u_eqn
from simulations.sim_io import SnapshotReader


def load_pinn(yaml_path: Path, model_path: Path, device: torch.device) -> pinninit.PINN:
    """A trained model, ready to evaluate."""
    pinn = pinninit.PINN.from_yaml(yaml_path, device)
    pinn.load(model_path)
    return pinn


def evaluate_pinn_u_b(
    pinn: pinninit.PINN, coords: Coords, n_batches: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the PINN u and b fields at a set of coordinates."""
    pinn.model.eval()
    u_pred, b_pred = [], []
    indices = torch.arange(coords.n_points, device=coords.device)
    for batch_indices in torch.tensor_split(indices, n_batches):
        batch = coords.get_batch(batch_indices)
        out = pinn.model.evaluate(batch)
        u = u_eqn(out[:, 1], batch.get_coord("x"), Operators(batch))
        u_pred.append(u.detach())
        b_pred.append(out[:, 0].detach())
    return torch.cat(u_pred), torch.cat(b_pred)


def slice_coords(x: np.ndarray, time: float, device: torch.device) -> Coords:
    """(x, t) coords along x at a single time."""
    coord_system = CartesianCoords(coord_axes={"x"}, device=device)
    XTCoords = make_coords_dataclass({"x", "t"}, coord_system, device)
    return XTCoords(
        x=torch.tensor(x, device=device, requires_grad=True),
        t=torch.tensor(np.full_like(x, time), device=device, requires_grad=True),
    )


def snapshot_field_profiles(
    yaml_path: Path,
    model_path: Path,
    reader: SnapshotReader,
    time: float,
    n_batches: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """The true and predicted (u, b) profiles at one exact snapshot time."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    idx = int(np.argmin(np.abs(reader.times - time)))
    if not np.isclose(reader.times[idx], time):
        raise ValueError(
            f"No snapshot at t={time}; nearest available is {reader.times[idx]:g}"
        )
    fields = reader.load_fields(idx, "u", "b")

    pinn = load_pinn(yaml_path, model_path, device)
    coords = slice_coords(reader.x, time, device)
    u_pred, b_pred = evaluate_pinn_u_b(pinn, coords, n_batches)

    return {
        "x": reader.x,
        "u_true": fields["u"],
        "b_true": fields["b"],
        "u_pred": u_pred.cpu().numpy(),
        "b_pred": b_pred.cpu().numpy(),
    }
