"""
Parallel multi-run utility: run multiple independent simulations concurrently.

Use case: parameter sweeps (e.g., scan over gamma values), ensemble runs
with different random seeds, or comparison of models.

Each simulation is fully independent → embarrassingly parallel.
Uses concurrent.futures.ProcessPoolExecutor (fork-safe, stdlib).
"""

from __future__ import annotations
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from pattern_formation.core.grid import CartesianGrid
from pattern_formation.solvers.explicit import ExplicitSolver


@dataclass
class RunConfig:
    """Configuration for a single simulation run."""
    model_cls:   type
    model_kwargs: dict
    N:           int   = 64
    delta1:      float = 0.01
    delta2:      float = 1.0
    dt:          float | None = None   # None → auto (90 % stability limit)
    n_steps:     int   = 5000
    seed:        int   = 0
    label:       str   = ""


def _run_one(cfg: RunConfig) -> dict[str, Any]:
    """Worker: run one simulation. Returns result dict."""
    import warnings
    warnings.filterwarnings("ignore")

    grid  = CartesianGrid(N=cfg.N)
    model = cfg.model_cls(**cfg.model_kwargs)
    u0, v0 = model.initial_conditions(grid, seed=cfg.seed)

    dt = cfg.dt
    if dt is None:
        dt = grid.dx**2 / (4.0 * max(cfg.delta1, cfg.delta2)) * 0.9

    solver = ExplicitSolver(model, grid, cfg.delta1, cfg.delta2, dt)
    u, v   = solver.run(u0, v0, cfg.n_steps)

    return {
        "label":  cfg.label or str(cfg.model_kwargs),
        "u":      u,
        "v":      v,
        "model_kwargs": cfg.model_kwargs,
        "n_steps": cfg.n_steps,
        "dt":      dt,
    }


def parallel_multi_run(
    configs: list[RunConfig],
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run a list of RunConfigs in parallel using ProcessPoolExecutor.

    Parameters
    ----------
    configs     : list of RunConfig instances
    max_workers : number of parallel processes (None → os.cpu_count())

    Returns
    -------
    List of result dicts in the same order as *configs*.
    """
    results = [None] * len(configs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_run_one, cfg): i
            for i, cfg in enumerate(configs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results
