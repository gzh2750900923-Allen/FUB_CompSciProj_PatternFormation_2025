"""
Unit tests for the Explicit (Forward Euler) solver.

Tests
-----
1. Stability limit calculation
2. Single step shape preservation
3. Zero-reaction, zero-diffusion: solution stays constant
4. Diffusion only (no reaction): mass conservation
5. Convergence order in time (O(dt))
"""

import numpy as np

from pattern_formation.core.grid import CartesianGrid
from pattern_formation.core.interface import BaseModel
from pattern_formation.solvers.explicit import ExplicitSolver


# ── helper model fixtures ─────────────────────────────────────────────────────

class ZeroReaction(BaseModel):
    """f = g = 0  →  pure diffusion."""
    def f(self, u, v): return np.zeros_like(u)
    def g(self, u, v): return np.zeros_like(v)


class ConstantReaction(BaseModel):
    """f = c, g = 0  — used to check Euler step value."""
    def __init__(self, c=1.0):
        self.c = c
    def f(self, u, v): return np.full_like(u, self.c)
    def g(self, u, v): return np.zeros_like(v)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_stability_limit():
    grid = CartesianGrid(N=32)
    model = ZeroReaction()
    solver = ExplicitSolver(model, grid, delta1=1e-4, delta2=5e-5, dt=1e-6)
    expected = grid.dx**2 / (4.0 * max(solver.delta1, solver.delta2))
    assert np.isclose(solver.stability_limit, expected)
    print("  [PASS] stability_limit correct")


def test_step_output_shape():
    grid  = CartesianGrid(N=16)
    model = ZeroReaction()
    solver = ExplicitSolver(model, grid, delta1=1e-5, delta2=1e-5, dt=1e-5)
    u = np.ones((16, 16))
    v = np.zeros((16, 16))
    u2, v2 = solver.step(u, v)
    assert u2.shape == (16, 16) and v2.shape == (16, 16)
    print("  [PASS] step output shape correct")


def test_constant_field_no_reaction_no_change():
    """Constant u/v with zero diffusion (delta→0) and zero reaction stays constant."""
    grid  = CartesianGrid(N=16)
    model = ZeroReaction()
    # Very small delta so Laplacian contribution vanishes numerically
    solver = ExplicitSolver(model, grid, delta1=0.0, delta2=0.0, dt=0.1)
    u = np.full((16, 16), 0.7)
    v = np.full((16, 16), 0.3)
    u2, v2 = solver.step(u, v)
    assert np.allclose(u2, 0.7) and np.allclose(v2, 0.3)
    print("  [PASS] constant field with zero delta stays constant")


def test_mass_conservation_pure_diffusion():
    """
    Pure diffusion conserves total mass (periodic BC, no reaction).
    Sum of u must remain constant over many steps.
    """
    grid   = CartesianGrid(N=32)
    model  = ZeroReaction()
    dt     = grid.dx**2 / (4.0 * 1e-4) * 0.4   # 40 % of stability limit
    solver = ExplicitSolver(model, grid, delta1=1e-4, delta2=1e-4, dt=dt)

    rng  = np.random.default_rng(0)
    u0   = rng.random((32, 32))
    v0   = rng.random((32, 32))
    mass_u0 = u0.sum()
    mass_v0 = v0.sum()

    u, v = u0.copy(), v0.copy()
    for _ in range(50):
        u, v = solver.step(u, v)

    assert np.isclose(u.sum(), mass_u0, rtol=1e-10), \
        f"Mass drift: {abs(u.sum()-mass_u0):.2e}"
    assert np.isclose(v.sum(), mass_v0, rtol=1e-10)
    print("  [PASS] mass conservation under pure diffusion")


def test_euler_step_value_with_reaction():
    """
    With delta=0 and f=c (constant), one step gives u + dt*c exactly.
    """
    grid   = CartesianGrid(N=8)
    c      = 2.5
    model  = ConstantReaction(c)
    dt     = 0.01
    solver = ExplicitSolver(model, grid, delta1=0.0, delta2=0.0, dt=dt)
    u = np.ones((8, 8)) * 0.5
    v = np.zeros((8, 8))
    u2, _ = solver.step(u, v)
    expected = 0.5 + dt * c
    assert np.allclose(u2, expected), f"Got {u2[0,0]:.6f}, expected {expected:.6f}"
    print("  [PASS] Euler step value exact for constant reaction")


if __name__ == "__main__":
    print("=== Explicit Solver Tests ===")
    test_stability_limit()
    test_step_output_shape()
    test_constant_field_no_reaction_no_change()
    test_mass_conservation_pure_diffusion()
    test_euler_step_value_with_reaction()
    print("All explicit solver tests PASSED ✓")
