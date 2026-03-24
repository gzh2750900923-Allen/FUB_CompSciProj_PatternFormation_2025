"""
Gray-Scott validation: convergence order & benchmark.

1. Pattern simulation  — run both solvers, compare visually
2. Convergence test    — measure error vs reference at decreasing dt
3. Performance bench   — wall-clock time for N=64 and N=128

All figures are saved to outputs/ via pattern_formation.visualization.plot.
"""

import time, os, warnings

import numpy as np

from pattern_formation.core.grid import CartesianGrid
from pattern_formation.models.gray_scott import GrayScottModel
from pattern_formation.solvers.explicit import ExplicitSolver
from pattern_formation.solvers.crank_nicolson import CrankNicolsonSolver
from pattern_formation.visualization.plot import (
    plot_comparison,
    plot_convergence,
    plot_benchmark,
)

os.makedirs("outputs", exist_ok=True)
warnings.filterwarnings("ignore")

ALPHA,  BETA   = 0.035, 0.065
DELTA1, DELTA2 = 2e-5, 1e-5
N_SIM          = 64
N_STEPS        = 2000


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pattern simulation & comparison
# ─────────────────────────────────────────────────────────────────────────────
def run_and_plot():
    print("\n=== Gray-Scott Pattern Simulation ===")
    model  = GrayScottModel(alpha=ALPHA, beta=BETA)
    grid   = CartesianGrid(N=N_SIM)
    u0, v0 = model.initial_conditions(grid, seed=0)

    # Explicit
    dt_exp = grid.dx**2 / (4 * max(DELTA1, DELTA2)) * 0.9
    exp    = ExplicitSolver(model, grid, DELTA1, DELTA2, dt_exp)
    t0     = time.perf_counter()
    u_e, v_e = exp.run(u0.copy(), v0.copy(), N_STEPS)
    t_exp  = time.perf_counter() - t0
    print(f"  Explicit       : {N_STEPS} steps  {t_exp:.2f}s  (dt={dt_exp:.2e})")

    # Crank-Nicolson
    dt_cn = dt_exp * 5.0
    cn    = CrankNicolsonSolver(model, grid, DELTA1, DELTA2, dt_cn)
    t0    = time.perf_counter()
    u_c, v_c = cn.run(u0.copy(), v0.copy(), N_STEPS)
    t_cn  = time.perf_counter() - t0
    print(f"  Crank-Nicolson : {N_STEPS} steps  {t_cn:.2f}s  (dt={dt_cn:.2e})")

    # ── plot via plot.py ──────────────────────────────────────────────────────
    plot_comparison(
        u_e, v_e, u_c, v_c,
        label1=f"Explicit  (dt={dt_exp:.2e}, {t_exp:.1f}s)",
        label2=f"Crank-Nicolson  (dt={dt_cn:.2e}, {t_cn:.1f}s)",
        title=f"Gray-Scott  (alpha={ALPHA}, beta={BETA},  N={N_SIM},  {N_STEPS} steps)",
        save_path="outputs/gray_scott_patterns.png",
    )
    return t_exp, t_cn, dt_exp, dt_cn


# ─────────────────────────────────────────────────────────────────────────────
# 2. Convergence order
# ─────────────────────────────────────────────────────────────────────────────
def convergence_test():
    print("\n=== Convergence Order Test ===")
    model  = GrayScottModel(alpha=ALPHA, beta=BETA)
    grid   = CartesianGrid(N=32)
    u0, v0 = model.initial_conditions(grid, seed=1)
    T_END  = 5.0

    # Reference solution (very small dt)
    dt_ref = grid.dx**2 / (4 * max(DELTA1, DELTA2)) * 0.05
    n_ref  = int(T_END / dt_ref)
    u_ref, _ = ExplicitSolver(model, grid, DELTA1, DELTA2, dt_ref) \
                   .run(u0.copy(), v0.copy(), n_ref)
    print(f"  Reference: dt={dt_ref:.2e}, {n_ref} steps")

    dt_base = grid.dx**2 / (4 * max(DELTA1, DELTA2)) * 0.4
    dts     = [dt_base / k for k in (1, 2, 4, 8)]
    errs_exp, errs_cn = [], []

    for dt in dts:
        n = int(T_END / dt)
        u_e, _ = ExplicitSolver(model, grid, DELTA1, DELTA2, dt) \
                     .run(u0.copy(), v0.copy(), n)
        u_c, _ = CrankNicolsonSolver(model, grid, DELTA1, DELTA2, dt) \
                     .run(u0.copy(), v0.copy(), n)
        errs_exp.append(np.max(np.abs(u_e - u_ref)))
        errs_cn.append( np.max(np.abs(u_c - u_ref)))

    orders_exp = [np.log2(errs_exp[i]/errs_exp[i+1]) for i in range(len(dts)-1)]
    orders_cn  = [np.log2(errs_cn[i] /errs_cn[i+1])  for i in range(len(dts)-1)]
    print(f"  Explicit  orders: {[f'{o:.2f}' for o in orders_exp]}")
    print(f"  CN        orders: {[f'{o:.2f}' for o in orders_cn]}")

    # ── plot via plot.py ──────────────────────────────────────────────────────
    plot_convergence(
        dts, errs_exp, errs_cn,
        title="Time-step convergence — Gray-Scott (N=32)",
        save_path="outputs/gray_scott_convergence.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Performance benchmark
# ─────────────────────────────────────────────────────────────────────────────
def benchmark():
    print("\n=== Performance Benchmark ===")
    model       = GrayScottModel(alpha=ALPHA, beta=BETA)
    BENCH_STEPS = 200
    labels, t_exps, t_cns = [], [], []

    for N in (64, 128):
        grid   = CartesianGrid(N=N)
        u0, v0 = model.initial_conditions(grid, seed=2)
        dt_exp = grid.dx**2 / (4 * max(DELTA1, DELTA2)) * 0.9
        dt_cn  = dt_exp * 5

        t0 = time.perf_counter()
        ExplicitSolver(model, grid, DELTA1, DELTA2, dt_exp) \
            .run(u0.copy(), v0.copy(), BENCH_STEPS)
        t_exp = time.perf_counter() - t0

        t0 = time.perf_counter()
        CrankNicolsonSolver(model, grid, DELTA1, DELTA2, dt_cn) \
            .run(u0.copy(), v0.copy(), BENCH_STEPS)
        t_cn = time.perf_counter() - t0

        print(f"  N={N:3d} | Explicit {t_exp:.3f}s | CN {t_cn:.3f}s "
              f"| ratio {t_cn/t_exp:.2f}x")
        labels.append(f"N={N}")
        t_exps.append(t_exp)
        t_cns.append(t_cn)

    # ── plot via plot.py ──────────────────────────────────────────────────────
    plot_benchmark(
        labels, t_exps, t_cns,
        title=f"Benchmark — {BENCH_STEPS} steps",
        save_path="outputs/gray_scott_benchmark.png",
    )


if __name__ == "__main__":
    run_and_plot()
    convergence_test()
    benchmark()
    print("\nAll Gray-Scott validation complete.")
