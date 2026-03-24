"""
Unit tests for the Crank-Nicolson solver.

Tests
-----
1. Step output shape
2. Constant field: stays constant (pure diffusion, zero reaction)
3. Mass conservation under pure diffusion
4. Laplacian matrix: correct stencil values and row-sums = 0 (periodic)
5. Agreement with Explicit solver at small dt (both O(dt) accurate)
"""

import numpy as np
from pattern_formation.core.grid import CartesianGrid
from pattern_formation.core.interface import BaseModel
from pattern_formation.solvers.crank_nicolson import CrankNicolsonSolver, _build_laplacian_matrix
from pattern_formation.solvers.explicit import ExplicitSolver


class ZeroReaction(BaseModel):
    def f(self, u, v): return np.zeros_like(u)
    def g(self, u, v): return np.zeros_like(v)


def test_laplacian_matrix_row_sums():
    """Row sums of the periodic Laplacian must be exactly zero."""
    N  = 8
    dx = 1.0 / N
    L  = _build_laplacian_matrix(N, dx)
    row_sums = np.array(L.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 0.0, atol=1e-12), \
        f"Max |row_sum| = {np.max(np.abs(row_sums)):.2e}"
    print("  [PASS] Laplacian matrix row sums = 0")


def test_laplacian_matrix_diagonal():
    """Diagonal entries = -4/dx²."""
    N  = 8
    dx = 1.0 / N
    L  = _build_laplacian_matrix(N, dx)
    diag = L.diagonal()
    expected = -4.0 / dx**2
    assert np.allclose(diag, expected)
    print("  [PASS] Laplacian matrix diagonal = -4/dx²")


def test_cn_step_output_shape():
    grid   = CartesianGrid(N=16)
    model  = ZeroReaction()
    solver = CrankNicolsonSolver(model, grid, delta1=1e-5, delta2=1e-5, dt=0.1)
    u = np.ones((16, 16))
    v = np.zeros((16, 16))
    u2, v2 = solver.step(u, v)
    assert u2.shape == (16, 16) and v2.shape == (16, 16)
    print("  [PASS] CN step output shape correct")


def test_cn_constant_field_stays_constant():
    """A uniform field with zero reaction must stay constant (Δconst = 0)."""
    grid   = CartesianGrid(N=16)
    model  = ZeroReaction()
    solver = CrankNicolsonSolver(model, grid, delta1=1e-3, delta2=1e-3, dt=0.1)
    u = np.full((16, 16), 0.6)
    v = np.full((16, 16), 0.4)
    for _ in range(10):
        u, v = solver.step(u, v)
    assert np.allclose(u, 0.6, atol=1e-12)
    assert np.allclose(v, 0.4, atol=1e-12)
    print("  [PASS] Constant field stays constant under CN")


def test_cn_mass_conservation():
    """Pure diffusion (no reaction) must conserve total mass."""
    grid   = CartesianGrid(N=32)
    model  = ZeroReaction()
    solver = CrankNicolsonSolver(model, grid, delta1=1e-4, delta2=1e-4, dt=0.5)

    rng = np.random.default_rng(7)
    u0  = rng.random((32, 32))
    v0  = rng.random((32, 32))
    mass_u = u0.sum()
    mass_v = v0.sum()

    u, v = u0.copy(), v0.copy()
    for _ in range(20):
        u, v = solver.step(u, v)

    assert np.isclose(u.sum(), mass_u, rtol=1e-10), \
        f"u mass drift: {abs(u.sum()-mass_u):.2e}"
    assert np.isclose(v.sum(), mass_v, rtol=1e-10)
    print("  [PASS] CN mass conservation under pure diffusion")


def test_cn_vs_explicit_small_dt():
    """
    At very small dt both solvers should agree closely (both first-order in
    the reaction part, second-order in diffusion for CN).
    Compare single-step results on a smooth IC.
    """
    grid  = CartesianGrid(N=32)
    model = ZeroReaction()
    dt    = 1e-6

    u0 = np.sin(2 * np.pi * grid.X) * np.cos(2 * np.pi * grid.Y) * 0.1 + 0.5
    v0 = np.zeros_like(u0)

    exp_solver = ExplicitSolver(model, grid, 1e-4, 1e-4, dt)
    cn_solver  = CrankNicolsonSolver(model, grid, 1e-4, 1e-4, dt)

    u_exp, _ = exp_solver.step(u0.copy(), v0.copy())
    u_cn,  _ = cn_solver.step(u0.copy(), v0.copy())

    diff = np.max(np.abs(u_exp - u_cn))
    assert diff < 1e-8, f"Difference too large: {diff:.2e}"
    print(f"  [PASS] CN vs Explicit max diff at dt=1e-6: {diff:.2e}")


if __name__ == "__main__":
    print("=== Crank-Nicolson Solver Tests ===")
    test_laplacian_matrix_row_sums()
    test_laplacian_matrix_diagonal()
    test_cn_step_output_shape()
    test_cn_constant_field_stays_constant()
    test_cn_mass_conservation()
    test_cn_vs_explicit_small_dt()
    print("All Crank-Nicolson tests PASSED ✓")
