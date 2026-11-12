import argparse
import logging
import shutil
from typing import Any

import dedalus.public as d3
import numpy as np

from simulations import sim_io

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, cfg_path: str) -> None:
        # Config
        cfg = self.load_config(cfg_path)
        self.cfg = cfg

        # Domain
        xcoord = d3.Coordinate("x")
        dist = d3.Distributor(xcoord, dtype=cfg.dtype)
        xbasis = d3.Chebyshev(
            xcoord, size=cfg.N, bounds=(0, cfg.L), dealias=cfg.dealias
        )

        # Fields and taus
        self.zp = dist.Field(name="zp", bases=xbasis)
        self.zm = dist.Field(name="zm", bases=xbasis)
        self.zm_in = dist.Field(name="zm_in")
        self.tau_zp1 = dist.Field(name="tau_zp1")
        self.tau_zp2 = dist.Field(name="tau_zp2")
        self.tau_zm1 = dist.Field(name="tau_zm1")
        self.tau_zm2 = dist.Field(name="tau_zm2")

        # Params
        Re = cfg.Re
        Rm = cfg.Rm
        Ma = cfg.Ma
        nu_p = Ma * 0.5 * (1 / Re + 1 / Rm)
        nu_m = Ma * 0.5 * (1 / Re - 1 / Rm)
        zm_in = self.zm_in

        def lift(A: Any, n: int) -> Any:
            return d3.Lift(A, xbasis, n)

        def dx(A: Any) -> Any:
            return d3.Differentiate(A, xcoord)

        # Problem
        problem = d3.IVP(
            [self.zp, self.zm, self.tau_zp1, self.tau_zp2, self.tau_zm1, self.tau_zm2],
            namespace=locals(),
        )

        if cfg.with_nonlin_term:
            problem.add_equation(
                "dt(zp) - dx(zp) - nu_p*dx(dx(zp)) - nu_m*dx(dx(zm))"
                " + lift(tau_zp1, -1) + lift(tau_zp2, -2)"
                " = -Ma*zm*dx(zp)"
            )

            problem.add_equation(
                "dt(zm) + dx(zm) - nu_p*dx(dx(zm)) - nu_m*dx(dx(zp))"
                " + lift(tau_zm1, -1) + lift(tau_zm2, -2)"
                " = -Ma*zp*dx(zm)"
            )
        else:
            problem.add_equation(
                "dt(zp) - dx(zp) - nu_p*dx(dx(zp)) - nu_m*dx(dx(zm))"
                " + lift(tau_zp1, -1) + lift(tau_zp2, -2)"
                " = 0.0"
            )

            problem.add_equation(
                "dt(zm) + dx(zm) - nu_p*dx(dx(zm)) - nu_m*dx(dx(zp))"
                " + lift(tau_zm1, -1) + lift(tau_zm2, -2)"
                " = 0.0"
            )

        # Forcing at x=0
        problem.add_equation("zm(x=0) = zm_in")
        problem.add_equation("dx(zp)(x=0) = 0")

        # Reflection at x=L
        problem.add_equation(f"zp(x={cfg.L}) + zm(x={cfg.L}) = 0")
        problem.add_equation(f"dx(zp)(x={cfg.L}) - dx(zm)(x={cfg.L}) = 0")

        # Solver
        self.solver = problem.build_solver(d3.SBDF2)
        self.solver.stop_sim_time = cfg.T + cfg.dt

        # ICs
        self.x = dist.local_grid(xbasis)
        self.zp["g"] = cfg.zp_initial(self.x)
        self.zm["g"] = cfg.zm_initial(self.x)

        # Output grid
        N_out = round(cfg.L / cfg.output_dx)
        if not np.isclose(cfg.L, N_out * cfg.output_dx):
            raise ValueError("output_dx must divide L exactly.")
        self.x_out = np.linspace(0.0, cfg.L, N_out + 1)

    def load_config(self, cfg_path: str) -> Any:
        config_path = sim_io.resolve_sim_config_path(cfg_path)
        cfg = sim_io.load_sim_config(config_path)
        cfg.root_dir = config_path.parent
        return cfg

    def run(self) -> None:
        cfg = self.cfg
        step = 0
        snapshot_num = 0
        snapshot_steps = {int(round(t / cfg.dt)) for t in cfg.snapshot_times}

        # Start clean, so a re-run cannot mix snapshots from different configs
        data_path = (cfg.root_dir / cfg.data_dir).resolve()
        if cfg.root_dir.resolve() not in data_path.parents:
            raise ValueError(f"data_dir must be a subdirectory of {cfg.root_dir}")
        shutil.rmtree(data_path, ignore_errors=True)

        while self.solver.proceed:
            # Write snapshots before stepping
            if step in snapshot_steps:
                self.write_snapshot(step * cfg.dt, snapshot_num)
                snapshot_num += 1

            # Update zm_in
            self.zm_in["g"] = cfg.zm_in(t=self.solver.sim_time)

            # Step
            self.solver.step(cfg.dt)

            if self.solver.iteration % 100 == 0:
                logger.info(f"t={self.solver.sim_time:.10f}")

            step += 1

    def write_snapshot(self, t: float, snapshot_num: int) -> None:
        cfg = self.cfg
        path = cfg.root_dir / cfg.data_dir / cfg.snapshot_dir
        path.mkdir(parents=True, exist_ok=True)

        x = self.x_out
        zp = np.array([self.zp(x=xi).evaluate()["g"][0] for xi in self.x_out])
        zm = np.array([self.zm(x=xi).evaluate()["g"][0] for xi in self.x_out])
        params = {
            "L": cfg.L,
            "Re": cfg.Re,
            "Rm": cfg.Rm,
            "Ma": cfg.Ma,
            "N": cfg.N,
            "dealias": cfg.dealias,
            "dtype": np.dtype(cfg.dtype).name,
            "dt": cfg.dt,
            "T": cfg.T,
            "output_dx": cfg.output_dx,
            "output_N": len(x),
            "with_nonlin_term": cfg.with_nonlin_term,
            "freqs": np.array(cfg.FREQS),
        }
        fields = {"zp": zp, "zm": zm, "u": 0.5 * (zp + zm), "b": 0.5 * (zp - zm)}

        sim_io.write_snapshot(
            path / f"snapshot_{snapshot_num:06d}.hdf5", t, params, x, fields
        )


if __name__ == "__main__":
    # Get path to config file
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the sim_config.py file")
    args = parser.parse_args()

    # Simulate
    Simulation(args.config).run()
