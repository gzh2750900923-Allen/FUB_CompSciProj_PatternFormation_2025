# Pattern Formation — Turing Reaction-Diffusion Simulator

---

## Overview

This package simulates **Turing pattern formation** via two-component reaction-diffusion systems:

$$\partial_t u = \delta_1 \Delta u + f(u,v), \qquad \partial_t v = \delta_2 \Delta v + g(u,v)$$

on the periodic unit square $\Omega = [0,1]^2$.

Key features:
- **Two numerical solvers**: Explicit (Forward Euler) and Crank-Nicolson (IMEX)
- **Three reaction models**: Gray-Scott, Giraffe (Schnakenberg), Leopard (Gierer-Meinhardt)
- **Validated**: convergence tests, mass conservation, Turing instability conditions
- **Parallelisation**: domain decomposition, matrix-free CG solver, parallel parameter sweeps

---

## Installation

```bash
git clone https://github.com/gzh2750900923-Allen/FUB_CompSciProj_PatternFormation_2025.git
cd pattern_formation
python -m venv venv #(Windows)Optional
venv\Scripts\activate #(Windows)
python3 -m venv venv #(Linux)
source venv/bin/activate #(Linux)
pip install -e ".[dev]"
```

**Dependencies:** `numpy >= 1.24`, `scipy >= 1.10`, `matplotlib >= 3.7`

---

## Quick Start

### Gray-Scott spots

```python
from pattern_formation.core.grid import CartesianGrid
from pattern_formation.models.gray_scott import GrayScottModel
from pattern_formation.solvers.explicit import ExplicitSolver
from pattern_formation.visualization.plot import plot_state

model = GrayScottModel(alpha=0.035, beta=0.065)
grid  = CartesianGrid(N=128)
u0, v0 = model.initial_conditions(grid, seed=42)

dt = grid.dx**2 / (4 * 1e-5) * 0.9          # 90% of stability limit
solver = ExplicitSolver(model, grid, delta1=2e-5, delta2=1e-5, dt=dt)
u, v = solver.run(u0, v0, n_steps=5000)

plot_state(u, v, title="Gray-Scott Spots")
```

### Giraffe pattern (Schnakenberg)

```python
from pattern_formation.models.giraffe import GiraffeModel
from pattern_formation.solvers.crank_nicolson import CrankNicolsonSolver

model = GiraffeModel(a=0.1, b=0.9, gamma=100)
grid  = CartesianGrid(N=96)
u0, v0 = model.initial_conditions(grid, seed=7)

solver = CrankNicolsonSolver(model, grid, delta1=0.01, delta2=1.0, dt=5e-4)
u, v = solver.run(u0, v0, n_steps=10000)
plot_state(u, v, title="Giraffe Pattern")
```

### Leopard pattern (Gierer-Meinhardt)

```python
from pattern_formation.models.leopard import LeopardModel

model = LeopardModel(gamma=300, mu=0.5, nu=1.0, kappa=0.1)
grid  = CartesianGrid(N=96)
u0, v0 = model.initial_conditions(grid, seed=3)

solver = CrankNicolsonSolver(model, grid, delta1=0.01, delta2=1.0, dt=5e-4)
u, v = solver.run(u0, v0, n_steps=10000)
plot_state(u, v, title="Leopard Pattern")
```

### Parallel parameter sweep

```python
from pattern_formation.parallel.multi_run import RunConfig, parallel_multi_run

configs = [
    RunConfig(model_cls=GiraffeModel, model_kwargs=dict(a=0.1, b=0.9, gamma=g),
              N=64, n_steps=5000, seed=0, label=f"gamma={g}")
    for g in [50, 100, 200, 400]
]
results = parallel_multi_run(configs, max_workers=4)
```

---

## Package Structure

```
pattern_formation/
├── setup.py                    # pip-installable package
├── requirements.txt            # dependency list
├── README.md                   # this file
├── docs/
│   └── DEVELOPMENT_PROCESS.md
│
├── pattern_formation/
│   ├── core/
│   │   ├── grid.py             # CartesianGrid: N, dx, X, Y meshgrid
│   │   ├── laplacian.py        # 2nd-order FD Laplacian (periodic, O(dx²))
│   │   └── interface.py        # BaseModel, BaseSolver (abstract base classes)
│   │
│   ├── models/
│   │   ├── gray_scott.py       # GrayScottModel — validation & benchmarks
│   │   ├── giraffe.py          # GiraffeModel — Schnakenberg (large patches)
│   │   └── leopard.py          # LeopardModel — Gierer-Meinhardt (spots)
│   │
│   ├── solvers/
│   │   ├── explicit.py         # ExplicitSolver — Forward Euler, stability check
│   │   └── crank_nicolson.py   # CrankNicolsonSolver — IMEX, Kronecker Laplacian
│   │
│   ├── parallel/
│   │   ├── parallel_explicit.py # Domain decomposition via SharedMemory + Pool
│   │   ├── matrix_free_cn.py    # Matrix-free CG solver (LinearOperator + CG)
│   │   └── multi_run.py         # Embarrassingly parallel parameter sweeps
│   │
│   └── visualization/
│       └── plot.py              # plot_state(), animate_history()
│
└── tests/                       # 26 unit tests across all modules
    ├── test_grid.py
    ├── test_laplacian.py
    ├── test_gray_scott.py
    ├── test_explicit.py
    ├── test_crank_nicolson.py
    ├── test_bio_models.py
    └── test_parallel.py
```

---

## Numerical Methods

### Spatial discretisation

Second-order central differences on a uniform N×N grid with **periodic boundary conditions**:

```
Δu[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4·u[i,j]) / dx²
```

Periodic wrapping implemented via `numpy.roll` — O(1) per axis, cache-friendly.  
Spatial convergence: **O(dx²)**, verified with sin/cos exact solution (ratio ≈ 4.00).

### Explicit solver

```
u^{n+1} = u^n + dt · [δ₁ Δu^n + f(u^n, v^n)]
```

**Stability:** `dt ≤ dx² / (4·max(δ₁,δ₂))` — checked at construction time.  
**Cost:** O(N²) per step. Fast for small dt; dominant method for stiff reactions.

### Crank-Nicolson (IMEX)

```
(I − dt/2·δ₁·L) u^{n+1} = (I + dt/2·δ₁·L) u^n + dt·f(u^n,v^n)
```

**Stability:** unconditional for the diffusion part.  
**Implementation:** Kronecker-sum sparse Laplacian, SuperLU pre-factorisation (paid once).  
**Cost:** O(N²) matvec + O(N³) at construction; O(N²) solve per step.

---

## Models

| Model | Class | Reference | Pattern |
|-------|-------|-----------|---------|
| Gray-Scott | `GrayScottModel` | Pearson 1993 | Spots / stripes / maze |
| Giraffe (Schnakenberg) | `GiraffeModel` | Murray 2003 | Large irregular polygons |
| Leopard (Gierer-Meinhardt) | `LeopardModel` | Liu et al. 2006 | Uniform circular spots |

### Gray-Scott presets

```python
model = GrayScottModel.from_preset('spots')    # alpha=0.035, beta=0.065
model = GrayScottModel.from_preset('stripes')  # alpha=0.060, beta=0.062
model = GrayScottModel.from_preset('maze')     # alpha=0.029, beta=0.057
model = GrayScottModel.from_preset('worms')    # alpha=0.039, beta=0.058
```

---

## Adding a Custom Model

```python
from pattern_formation.core.interface import BaseModel
import numpy as np

class MyModel(BaseModel):
    def __init__(self, param1=1.0):
        self.param1 = param1

    def f(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return ...   # return array same shape as u

    def g(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return ...

    def initial_conditions(self, grid, seed=0):
        rng = np.random.default_rng(seed)
        u0 = ... + 0.01 * rng.standard_normal((grid.N, grid.N))
        v0 = ...
        return u0, v0
```

Any model subclassing `BaseModel` works with both `ExplicitSolver` and
`CrankNicolsonSolver` automatically.

---

## Running Tests

```bash
# All 26 tests
python3 -m pytest tests/ -v --cov=pattern_formation

# Single module
python3 tests/test_explicit.py
python3 tests/test_bio_models.py
```

Test coverage spans: grid, Laplacian convergence, solver correctness,
mass conservation, Turing conditions, biological model stability,
matrix-free CG correctness, and parallel multi-run.

---

## Performance Guide

| Grid N | Solver | ms/step | Recommendation |
|--------|--------|---------|----------------|
| ≤ 64   | Explicit | 0.14 ms | ✅ Fastest overall |
| ≤ 64   | CN       | 1.09 ms | Use for stiff-free, long T |
| 128    | Explicit | 0.42 ms | ✅ Preferred for stiff models |
| 128    | CN       | 5.45 ms | Use when dt can be 10× larger |
| ≥ 256  | Explicit | 3.4 ms  | ✅ Better efficiency/phys-time |
| ≥ 256  | CN       | 26+ ms  | Avoid — O(N³) cost dominates |

**Tip:** For stiff reaction terms (large γ), the explicit solver is usually
faster per unit of physical time because CN's dt advantage is limited by the
reaction timescale `1/γ`, not diffusion.

---

## Parallelisation

Three strategies are implemented (see `docs/parallelization_report.md`):

| Strategy | Module | Speedup | Effective when |
|----------|--------|---------|----------------|
| Domain decomposition | `ParallelExplicitSolver` | <1× (Python) | N≥512 + Numba workers |
| Matrix-free CG | `MatrixFreeCNSolver` | 0.2–0.5× | Needs multigrid precon |
| Parameter sweep | `parallel_multi_run` | ~1.5× | ≥10 s per run |

---

## Development Process

See `DEVELOPMENT_PROCESS.md` for the full Git-Flow workflow, commit
message conventions, and code-style guidelines.

```bash
# Install dev tools
pip install -e ".[dev]"

# Format
black pattern_formation tests

# Lint
flake8 pattern_formation tests
```

---

## References

1. A. M. Turing (1952). "The chemical basis of morphogenesis." *Phil. Trans. R. Soc. B*, 237, 37–72.
2. P. Ball (2015). "Forging patterns and making waves." *Phil. Trans. R. Soc. B*, 370, 20140218.
3. S. Kondo & T. Miura (2010). "Reaction-Diffusion Model as a Framework for Understanding Biological Pattern Formation." *Science*, 329, 1616–1620.
4. J. E. Pearson (1993). "Complex Patterns in a Simple System." *Science*, 261, 189–192.
5. J. Murray (2003). *Mathematical Biology II*. Springer.
6. R. T. Liu, S. S. Liaw & P. K. Maini (2006). "Two-stage Turing model for generating pigment patterns on the leopard and the jaguar." *Phys. Rev. E*, 74, 011914.
