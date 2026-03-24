"""
Unit tests for GiraffeModel and LeopardModel.

Tests
-----
1. Steady-state values correct
2. Reaction terms at steady state = 0
3. Jacobian signs satisfy Turing conditions
4. Initial conditions shape and positivity
5. Short simulation remains finite (smoke test)
"""
import numpy as np
from pattern_formation.core.grid import CartesianGrid
from pattern_formation.models.giraffe import GiraffeModel
from pattern_formation.models.leopard import LeopardModel
from pattern_formation.solvers.explicit import ExplicitSolver


# ── Giraffe (Schnakenberg) ────────────────────────────────────────────────────
def test_giraffe_steady_state():
    """At (u*, v*), f=g=0."""
    m = GiraffeModel(a=0.1, b=0.9, gamma=100)
    u_ss, v_ss = m.steady_state
    u = np.full((8,8), u_ss)
    v = np.full((8,8), v_ss)
    assert np.allclose(m.f(u, v), 0.0, atol=1e-10), "f ≠ 0 at SS"
    assert np.allclose(m.g(u, v), 0.0, atol=1e-10), "g ≠ 0 at SS"
    print("  [PASS] Giraffe steady state: f=g=0")


def test_giraffe_turing_conditions():
    """
    Jacobian at SS must satisfy:
        trace(J) < 0, det(J) > 0, and Turing discriminant > 0.
    """
    a, b, gamma = 0.1, 0.9, 100
    DELTA1, DELTA2 = 0.01, 1.0
    u_ss = a + b;  v_ss = b / u_ss**2
    f_u = gamma * (-1 + 2*u_ss*v_ss)
    f_v = gamma * u_ss**2
    g_u = gamma * (-2*u_ss*v_ss)
    g_v = gamma * (-u_ss**2)

    trace = f_u + g_v
    det   = f_u*g_v - f_v*g_u
    h     = DELTA2*f_u + DELTA1*g_v
    discr = h**2 - 4*DELTA1*DELTA2*det

    assert trace < 0,  f"trace={trace:.2f} should be <0"
    assert det   > 0,  f"det={det:.2f} should be >0"
    assert h     > 0,  f"h={h:.2f} should be >0"
    assert discr > 0,  f"discr={discr:.2f} should be >0 for Turing"
    print("  [PASS] Giraffe Turing instability conditions satisfied")


def test_giraffe_ic_shape_positive():
    grid = CartesianGrid(N=16)
    m = GiraffeModel()
    u0, v0 = m.initial_conditions(grid, seed=42)
    assert u0.shape == (16, 16)
    assert np.all(u0 >= 0) and np.all(v0 >= 0), "ICs must be non-negative"
    print("  [PASS] Giraffe ICs: correct shape and non-negative")


def test_giraffe_simulation_finite():
    """100 explicit steps should stay finite."""
    grid = CartesianGrid(N=16)
    m  = GiraffeModel(a=0.1, b=0.9, gamma=100)
    u0, v0 = m.initial_conditions(grid, seed=0)
    dt = grid.dx**2 / (4 * 1.0) * 0.9
    import warnings
    with warnings.catch_warnings(): warnings.simplefilter('ignore')
    sol = ExplicitSolver(m, grid, 0.01, 1.0, dt)
    u, v = sol.run(u0, v0, 100)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    print("  [PASS] Giraffe simulation remains finite after 100 steps")


# ── Leopard (Gierer-Meinhardt) ────────────────────────────────────────────────
def test_leopard_steady_state_approx():
    """Approximate SS: g(u*,v*)≈0 when κ is small."""
    m = LeopardModel(gamma=300, mu=0.5, nu=1.0, kappa=0.01)
    u_ss, v_ss = m.steady_state
    u = np.full((8,8), u_ss)
    v = np.full((8,8), v_ss)
    # g(u*,v*) = gamma*(u*² - nu*v*) should be ~0
    g_val = m.g(u, v)
    assert np.allclose(g_val, 0.0, atol=1e-6), f"g at SS ≠ 0: {g_val.mean():.2e}"
    print("  [PASS] Leopard g=0 at steady state")


def test_leopard_turing_conditions():
    """Turing instability conditions for Gierer-Meinhardt."""
    mu, nu, kappa, gamma = 0.5, 1.0, 0.1, 300
    DELTA1, DELTA2 = 0.01, 1.0
    u_ss = np.sqrt(nu/mu); v_ss = u_ss**2/nu
    sat  = 1 + kappa*u_ss**2
    f_u  = gamma*(2*u_ss/(v_ss*sat) - 2*kappa*u_ss**3/(v_ss*sat**2) - mu)
    f_v  = gamma*(-u_ss**2/(v_ss**2*sat))
    g_u  = gamma*2*u_ss
    g_v  = gamma*(-nu)
    trace = f_u + g_v
    det   = f_u*g_v - f_v*g_u
    h     = DELTA2*f_u + DELTA1*g_v
    discr = h**2 - 4*DELTA1*DELTA2*det
    assert trace < 0, f"trace={trace:.2f} must be <0"
    assert det   > 0, f"det={det:.2f} must be >0"
    assert h     > 0, f"h={h:.2f} must be >0"
    assert discr > 0, f"discr={discr:.2f} must be >0"
    print("  [PASS] Leopard Turing instability conditions satisfied")


def test_leopard_ic_shape_positive():
    grid = CartesianGrid(N=16)
    m = LeopardModel()
    u0, v0 = m.initial_conditions(grid, seed=99)
    assert u0.shape == (16, 16)
    assert np.all(u0 >= 0) and np.all(v0 >= 0)
    print("  [PASS] Leopard ICs: correct shape and non-negative")


def test_leopard_simulation_finite():
    """100 explicit steps stay finite."""
    grid = CartesianGrid(N=16)
    m  = LeopardModel(gamma=300, mu=0.5, nu=1.0, kappa=0.1)
    u0, v0 = m.initial_conditions(grid, seed=0)
    dt = grid.dx**2 / (4 * 1.0) * 0.9
    import warnings
    with warnings.catch_warnings(): warnings.simplefilter('ignore')
    sol = ExplicitSolver(m, grid, 0.01, 1.0, dt)
    u, v = sol.run(u0, v0, 100)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    print("  [PASS] Leopard simulation remains finite after 100 steps")


if __name__ == "__main__":
    print("=== Giraffe Model Tests ===")
    test_giraffe_steady_state()
    test_giraffe_turing_conditions()
    test_giraffe_ic_shape_positive()
    test_giraffe_simulation_finite()
    print()
    print("=== Leopard Model Tests ===")
    test_leopard_steady_state_approx()
    test_leopard_turing_conditions()
    test_leopard_ic_shape_positive()
    test_leopard_simulation_finite()
    print()
    print("All biological model tests PASSED ✓")
