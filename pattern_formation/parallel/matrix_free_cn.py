"""
Matrix-free Crank-Nicolson solver.

Motivation
----------
In the standard CrankNicolsonSolver the LHS matrix A = I - dt/2·δ·L is
pre-factored with SuperLU.  For large N the factorisation itself is cheap
(done once), but the *solve* step dominates (94 % of wall time per step).

The matrix-free approach replaces the direct solve with an **iterative**
Krylov solver (scipy GMRES / CG) that never forms L explicitly.  Instead it
evaluates matrix-vector products

    A · x  =  x  -  (dt/2·δ) · L · x

where L·x is computed via the fast finite-difference roll stencil — the
same O(N²) operation used in the explicit solver — rather than a sparse
matrix-vector multiply.

Advantages
----------
· O(N²) memory  (no N²×N² sparse matrix stored)
· The MVP is implemented in pure numpy (no scipy.sparse overhead)
· Works naturally with domain decomposition — each process owns a slab

Disadvantages
-------------
· Iterative solver requires a good preconditioner for fast convergence
· Per-step cost is O(k·N²) where k = number of Krylov iterations (typically
  k = 5–20 for the diffusion operator with a Jacobi preconditioner)

Implementation
--------------
We use scipy.sparse.linalg.LinearOperator to wrap the matrix-free MVP and
pass it to scipy.sparse.linalg.cg (conjugate gradient — A is symmetric PD
after suitable shifting).
"""

import numpy as np
import scipy.sparse.linalg as spla

from pattern_formation.core.interface import BaseSolver, BaseModel
from pattern_formation.core.grid import CartesianGrid
from pattern_formation.core.laplacian import laplacian as lap_op


def _laplacian_matvec(x: np.ndarray, grid: CartesianGrid) -> np.ndarray:
    """Apply discrete Laplacian L to a flat vector x → L·x (flat)."""
    N  = grid.N
    u  = x.reshape(N, N)
    Lu = lap_op(u, grid)
    return Lu.ravel()


class MatrixFreeCNSolver(BaseSolver):
    """
    IMEX Crank-Nicolson solver with a matrix-free iterative linear solve.

    Each step solves:
        (I - dt/2·δ·L) · u_new = rhs_u

    using scipy's conjugate-gradient solver with a Jacobi preconditioner.
    The operator (I - dt/2·δ·L) is represented as a LinearOperator.

    Parameters
    ----------
    model, grid, delta1, delta2, dt : same as BaseSolver
    tol     : iterative solver tolerance (default 1e-8)
    maxiter : max CG iterations (default 100)
    """

    def __init__(self, model: BaseModel, grid: CartesianGrid,
                 delta1: float, delta2: float, dt: float,
                 tol: float = 1e-8, maxiter: int = 100):
        super().__init__(model, grid, delta1, delta2, dt)
        self.tol     = tol
        self.maxiter = maxiter
        self._build_operators()

    def _build_operators(self):
        N  = self.grid.N
        N2 = N * N
        dt = self.dt

        # ── Matrix-free A·x for u  ───────────────────────────────────────────
        def _matvec_u(x):
            return x - 0.5 * dt * self.delta1 * _laplacian_matvec(x, self.grid)

        def _matvec_v(x):
            return x - 0.5 * dt * self.delta2 * _laplacian_matvec(x, self.grid)

        self._A_u_op = spla.LinearOperator((N2, N2), matvec=_matvec_u, dtype=np.float64)
        self._A_v_op = spla.LinearOperator((N2, N2), matvec=_matvec_v, dtype=np.float64)

        # ── Jacobi (diagonal) preconditioner ─────────────────────────────────
        # Diagonal of (I - dt/2·δ·L) = 1 - dt/2·δ·(-4/dx²) = 1 + 2·dt·δ/dx²
        diag_u = 1.0 + 2.0 * dt * self.delta1 / self.grid.dx**2
        diag_v = 1.0 + 2.0 * dt * self.delta2 / self.grid.dx**2

        self._M_u = spla.LinearOperator(
            (N2, N2), matvec=lambda x: x / diag_u, dtype=np.float64)
        self._M_v = spla.LinearOperator(
            (N2, N2), matvec=lambda x: x / diag_v, dtype=np.float64)

        # ── Explicit RHS operator  (I + dt/2·δ·L) ────────────────────────────
        def _rhs_u(x):
            return x + 0.5 * dt * self.delta1 * _laplacian_matvec(x, self.grid)

        def _rhs_v(x):
            return x + 0.5 * dt * self.delta2 * _laplacian_matvec(x, self.grid)

        self._rhs_u_op = _rhs_u
        self._rhs_v_op = _rhs_v

        self._N2 = N2

    def step(self, u: np.ndarray, v: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
        N  = self.grid.N
        dt = self.dt

        u_flat = u.ravel()
        v_flat = v.ravel()

        rhs_u = self._rhs_u_op(u_flat) + dt * self.model.f(u, v).ravel()
        rhs_v = self._rhs_v_op(v_flat) + dt * self.model.g(u, v).ravel()

        u_new_flat, info_u = spla.cg(
            self._A_u_op, rhs_u, x0=u_flat,
            M=self._M_u, rtol=self.tol, maxiter=self.maxiter)
        v_new_flat, info_v = spla.cg(
            self._A_v_op, rhs_v, x0=v_flat,
            M=self._M_v, rtol=self.tol, maxiter=self.maxiter)

        if info_u != 0 or info_v != 0:
            import warnings
            warnings.warn(f"CG did not converge: info_u={info_u}, info_v={info_v}",
                          RuntimeWarning)

        return u_new_flat.reshape(N, N), v_new_flat.reshape(N, N)

    @property
    def iterations_last_step(self):
        """Not tracked per step in this implementation."""
        return None
