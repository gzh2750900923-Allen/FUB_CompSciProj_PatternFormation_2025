"""
Crank-Nicolson solver for the reaction-diffusion system.

The diffusion part is treated implicitly (unconditionally stable).
The reaction part is treated explicitly (IMEX operator splitting).

Scheme per step  (u only, v is symmetric):
    (I - dt/2 · δ₁ L) u^{n+1} = (I + dt/2 · δ₁ L) u^n + dt · f(u^n, v^n)

where L is the 2-D finite-difference Laplacian with periodic BCs,
built as a Kronecker sum of two 1-D tridiagonal periodic matrices.

The LHS matrices are pre-factored (SuperLU) at construction time so that
each call to `step` only requires two cheap triangular solves.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pattern_formation.core.interface import BaseSolver, BaseModel
from pattern_formation.core.grid import CartesianGrid


def _build_1d_laplacian(N: int, dx: float) -> sp.csr_matrix:
    """
    1-D second-order central-difference matrix on N points with
    periodic boundary conditions.

        L[i,i]   = -2/dx²
        L[i,i±1] =  1/dx²
        L[0,N-1] = L[N-1,0] = 1/dx²   (periodic wrap)
    """
    dx2 = dx * dx
    d   = np.full(N, -2.0 / dx2)
    off = np.ones(N - 1) / dx2
    L   = sp.diags([d, off, off], [0, 1, -1], shape=(N, N), format="lil")
    L[0, N - 1] = 1.0 / dx2
    L[N - 1, 0] = 1.0 / dx2
    return L.tocsr()


def _build_laplacian_matrix(N: int, dx: float) -> sp.csr_matrix:
    """
    2-D Laplacian matrix (N²×N²) for an N×N grid with periodic BCs.
    Built via Kronecker sum:  Δ = L_x ⊗ I + I ⊗ L_y.

    The flat index ordering is: k = i*N + j  (row-major / C order).
    """
    I  = sp.eye(N, format="csr")
    L1 = _build_1d_laplacian(N, dx)
    return (sp.kron(L1, I, format="csr")
            + sp.kron(I, L1, format="csr"))


class CrankNicolsonSolver(BaseSolver):
    """
    IMEX Crank-Nicolson integrator for the reaction-diffusion PDE.

    Diffusion: Crank-Nicolson (unconditionally stable).
    Reaction : Forward-Euler (explicit).

    Parameters
    ----------
    model  : BaseModel
    grid   : CartesianGrid
    delta1 : float  — diffusion coefficient for u
    delta2 : float  — diffusion coefficient for v
    dt     : float  — time step
    """

    def __init__(self, model: BaseModel, grid: CartesianGrid,
                 delta1: float, delta2: float, dt: float):
        super().__init__(model, grid, delta1, delta2, dt)
        self._build_operators()

    # ── operator construction ────────────────────────────────────────────────
    def _build_operators(self) -> None:
        N  = self.grid.N
        dx = self.grid.dx
        dt = self.dt

        L = _build_laplacian_matrix(N, dx)
        I = sp.eye(N * N, format="csr")

        # LHS  (I - dt/2 · δ · L)
        self._A_u = (I - 0.5 * dt * self.delta1 * L).tocsr()
        self._A_v = (I - 0.5 * dt * self.delta2 * L).tocsr()

        # RHS operator  (I + dt/2 · δ · L)
        self._B_u = (I + 0.5 * dt * self.delta1 * L).tocsr()
        self._B_v = (I + 0.5 * dt * self.delta2 * L).tocsr()

        # Pre-factorise (SuperLU) — paid once, cheap to solve
        self._lu_u = spla.splu(self._A_u.tocsc())
        self._lu_v = spla.splu(self._A_v.tocsc())

    # ── core step ─────────────────────────────────────────────────────────────
    def step(self, u: np.ndarray, v: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
        """
        Advance solution by one IMEX Crank-Nicolson step.

        Parameters
        ----------
        u, v : ndarray, shape (N, N)

        Returns
        -------
        (u_new, v_new) : ndarray, shape (N, N)
        """
        N  = self.grid.N
        dt = self.dt

        u_flat = u.ravel()
        v_flat = v.ravel()

        rhs_u = self._B_u @ u_flat + dt * self.model.f(u, v).ravel()
        rhs_v = self._B_v @ v_flat + dt * self.model.g(u, v).ravel()

        u_new = self._lu_u.solve(rhs_u).reshape(N, N)
        v_new = self._lu_v.solve(rhs_v).reshape(N, N)

        return u_new, v_new
