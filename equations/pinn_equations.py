from __future__ import annotations

import torch
from pinninit import Operators


def zp_eqn(
    b: torch.Tensor,
    u: torch.Tensor,
    Re: torch.Tensor,
    Rm: torch.Tensor,
    Ma: torch.Tensor,
    operators: Operators,
) -> torch.Tensor:
    nu_p = Ma * 0.5 * (1 / Re + 1 / Rm)
    nu_m = Ma * 0.5 * (1 / Re - 1 / Rm)
    zp = u + b
    zm = u - b

    dzp_dt: torch.Tensor = operators.d_dcoord(zp, "t")
    dzp_dx: torch.Tensor = operators.d_dcoord(zp, "x")
    d2zp_dx2: torch.Tensor = operators.d_dcoord(dzp_dx, "x")
    dzm_dx: torch.Tensor = operators.d_dcoord(zm, "x")
    d2zm_dx2: torch.Tensor = operators.d_dcoord(dzm_dx, "x")

    return dzp_dt - dzp_dx + Ma * zm * dzp_dx - nu_p * d2zp_dx2 - nu_m * d2zm_dx2


def zm_eqn(
    b: torch.Tensor,
    u: torch.Tensor,
    Re: torch.Tensor,
    Rm: torch.Tensor,
    Ma: torch.Tensor,
    operators: Operators,
) -> torch.Tensor:
    nu_p = Ma * 0.5 * (1 / Re + 1 / Rm)
    nu_m = Ma * 0.5 * (1 / Re - 1 / Rm)
    zp = u + b
    zm = u - b

    dzm_dt: torch.Tensor = operators.d_dcoord(zm, "t")
    dzm_dx: torch.Tensor = operators.d_dcoord(zm, "x")
    d2zm_dx2: torch.Tensor = operators.d_dcoord(dzm_dx, "x")
    dzp_dx: torch.Tensor = operators.d_dcoord(zp, "x")
    d2zp_dx2: torch.Tensor = operators.d_dcoord(dzp_dx, "x")

    return dzm_dt + dzm_dx + Ma * zp * dzm_dx - nu_p * d2zm_dx2 - nu_m * d2zp_dx2


def u_eqn(nn_out: torch.Tensor, x: torch.Tensor, operators: Operators) -> torch.Tensor:
    return (1.0 - x) * nn_out


def dbdx_bc_eqn(b: torch.Tensor, operators: Operators) -> torch.Tensor:
    db_dx: torch.Tensor = operators.d_dcoord(b, "x")
    return db_dx
