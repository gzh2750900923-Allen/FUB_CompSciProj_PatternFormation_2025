"""
Explicit (Forward Euler) solver for the reaction-diffusion system.

Update rule:
    u^{n+1} = u^n + dt * [δ₁ · Δu^n + f(u^n, v^n)]
    v^{n+1} = v^n + dt * [δ₂ · Δv^n + g(u^n, v^n)]

Stability constraint (von Neumann, 2-D diffusion):
    dt ≤ dx² / (4 · max(δ₁, δ₂))      [conservative, 2-D factor]
"""

import warnings
import numpy as np
from pattern_formation.core.interface import BaseSolver, BaseModel
from pattern_formation.core.grid import CartesianGrid
from pattern_formation.core.laplacian import laplacian


class ExplicitSolver(BaseSolver):
    """
    Forward-Euler time integrator for the PDE system:

        ∂ₜu = δ₁ Δu + f(u, v)
        ∂ₜv = δ₂ Δv + g(u, v)

    Parameters
    ----------
    model  : BaseModel      — provides f(u,v) and g(u,v)
    grid   : CartesianGrid  — spatial discretisation
    delta1 : float          — diffusion coefficient for u
    delta2 : float          — diffusion coefficient for v
    dt     : float          — time step (checked against stability limit)
    """

    def __init__(self, model: BaseModel, grid: CartesianGrid,
                 delta1: float, delta2: float, dt: float):
        super().__init__(model, grid, delta1, delta2, dt)
        self._check_stability()

    # ── stability ──────────────────────────────────────────────────────────────
    def _check_stability(self) -> None:
        """Warn if dt violates the von Neumann stability condition."""
        d_max = max(self.delta1, self.delta2)
        if d_max == 0.0:
            return
        dx2 = self.grid.dx ** 2
        limit = dx2 / (4.0 * d_max)
        if self.dt > limit:
            warnings.warn(
                f"[ExplicitSolver] dt={self.dt:.3e} exceeds stability limit "
                f"{limit:.3e}. Solution may blow up. "
                f"Consider dt <= {limit:.3e}.",
                RuntimeWarning, stacklevel=2)

    @property
    def stability_limit(self) -> float:
        """Maximum stable time step for the current grid and diffusion."""
        d_max = max(self.delta1, self.delta2)
        if d_max == 0.0:
            return float("inf")
        return self.grid.dx**2 / (4.0 * d_max)

    # ── core step ──────────────────────────────────────────────────────────────
    def step(self, u: np.ndarray, v: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
        """
        Advance solution by one Forward-Euler step.

        Parameters
        ----------
        u, v : ndarray, shape (N, N) — current concentrations

        Returns
        -------
        (u_new, v_new) : ndarrays, shape (N, N)
        """
        lap_u = laplacian(u, self.grid)
        lap_v = laplacian(v, self.grid)

        u_new = u + self.dt * (self.delta1 * lap_u + self.model.f(u, v))
        v_new = v + self.dt * (self.delta2 * lap_v + self.model.g(u, v))

        return u_new, v_new
