"""
Unit tests for parallel modules.

Tests
-----
1. MatrixFreeCNSolver: correctness vs standard CN
2. MatrixFreeCNSolver: mass conservation
3. MatrixFreeCNSolver: constant field stays constant
4. ParallelExplicitSolver: correctness vs serial
5. parallel_multi_run: returns correct number of results
"""
import numpy as np
import warnings

from pattern_formation.core.grid import CartesianGrid
from pattern_formation.models.giraffe import GiraffeModel
from pattern_formation.core.interface import BaseModel
from pattern_formation.solvers.crank_nicolson import CrankNicolsonSolver
from pattern_formation.solvers.explicit import ExplicitSolver
from pattern_formation.parallel.matrix_free_cn import MatrixFreeCNSolver
from pattern_formation.parallel.parallel_explicit import ParallelExplicitSolver
from pattern_formation.parallel.multi_run import RunConfig, parallel_multi_run


class ZeroReaction(BaseModel):
    def f(self, u, v): return np.zeros_like(u)
    def g(self, u, v): return np.zeros_like(v)


DELTA1, DELTA2 = 0.01, 1.0
DT_CN = 5e-4


def test_mf_cn_correctness():
    """MF-CN must agree with standard CN to 1e-5."""
    grid  = CartesianGrid(N=32)
    model = GiraffeModel(a=0.1, b=0.9, gamma=100)
    u0, v0 = model.initial_conditions(grid, seed=0)

    cn = CrankNicolsonSolver(model, grid, DELTA1, DELTA2, DT_CN)
    mf = MatrixFreeCNSolver(model,  grid, DELTA1, DELTA2, DT_CN, tol=1e-10)

    u_cn, _ = cn.run(u0.copy(), v0.copy(), 50)
    u_mf, _ = mf.run(u0.copy(), v0.copy(), 50)

    diff = np.max(np.abs(u_cn - u_mf))
    assert diff < 1e-5, f"MF-CN diff too large: {diff:.2e}"
    print(f"  [PASS] MF-CN correctness: max diff = {diff:.2e}")


def test_mf_cn_constant_field():
    """Constant field with zero reaction must stay constant."""
    grid  = CartesianGrid(N=16)
    model = ZeroReaction()
    mf    = MatrixFreeCNSolver(model, grid, 1e-3, 1e-3, 0.1, tol=1e-12)
    u  = np.full((16, 16), 0.6)
    v  = np.full((16, 16), 0.4)
    for _ in range(10):
        u, v = mf.step(u, v)
    assert np.allclose(u, 0.6, atol=1e-10)
    assert np.allclose(v, 0.4, atol=1e-10)
    print("  [PASS] MF-CN constant field stays constant")


def test_mf_cn_mass_conservation():
    """Pure diffusion conserves mass under MF-CN."""
    grid  = CartesianGrid(N=32)
    model = ZeroReaction()
    mf    = MatrixFreeCNSolver(model, grid, DELTA1, DELTA2, DT_CN, tol=1e-12)
    rng   = np.random.default_rng(5)
    u0    = rng.random((32, 32))
    v0    = rng.random((32, 32))
    mass_u = u0.sum()
    u, v   = u0.copy(), v0.copy()
    for _ in range(20):
        u, v = mf.step(u, v)
    assert np.isclose(u.sum(), mass_u, rtol=1e-8), \
        f"Mass drift: {abs(u.sum()-mass_u):.2e}"
    print("  [PASS] MF-CN mass conservation")


def test_parallel_explicit_correctness():
    """ParallelExplicit must match serial Explicit exactly."""
    grid  = CartesianGrid(N=32)
    model = GiraffeModel(a=0.1, b=0.9, gamma=100)
    u0, v0 = model.initial_conditions(grid, seed=0)
    dt = grid.dx**2 / (4*DELTA2) * 0.9

    with warnings.catch_warnings(): warnings.simplefilter('ignore')
    ser = ExplicitSolver(model, grid, DELTA1, DELTA2, dt)
    par = ParallelExplicitSolver(model, grid, DELTA1, DELTA2, dt, n_workers=2)

    u_s, _ = ser.run(u0.copy(), v0.copy(), 20)
    u_p, _ = par.run(u0.copy(), v0.copy(), 20)

    diff = np.max(np.abs(u_s - u_p))
    assert diff < 1e-12, f"Parallel explicit diff: {diff:.2e}"
    print(f"  [PASS] Parallel explicit correctness: max diff = {diff:.2e}")


def test_multi_run_count():
    """parallel_multi_run returns one result per config."""
    configs = [
        RunConfig(model_cls=GiraffeModel,
                  model_kwargs=dict(a=0.1, b=0.9, gamma=g),
                  N=16, n_steps=50, seed=0, label=f"g{g}")
        for g in [50, 100, 150]
    ]
    results = parallel_multi_run(configs, max_workers=2)
    assert len(results) == 3
    for r in results:
        assert r['u'].shape == (16, 16)
        assert np.isfinite(r['u']).all()
    print("  [PASS] multi_run returns 3 correct results")


if __name__ == "__main__":
    print("=== Parallel Module Tests ===")
    test_mf_cn_correctness()
    test_mf_cn_constant_field()
    test_mf_cn_mass_conservation()
    test_parallel_explicit_correctness()
    test_multi_run_count()
    print("\nAll parallel tests PASSED ✓")
