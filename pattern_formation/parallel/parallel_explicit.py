"""
Parallelised Forward-Euler solver using Python's multiprocessing.

Strategy
--------
The computational bottleneck in ExplicitSolver is the Laplacian (52 % of
wall time per step).  The 2-D discrete Laplacian decomposes row-wise:

    Δu[i, :] depends only on rows i-1, i, i+1  (plus the roll along axis=1)

We exploit this by splitting the N×N domain into P horizontal **slabs**
(one per CPU core), where P = number of worker processes.  Each worker
receives its slab plus a 1-row halo on each side (periodic wrap), computes
Δu on its interior rows, and returns the result.

The reaction terms f(u,v) and g(u,v) are elementwise → trivially parallel
in the same slab decomposition.

Shared memory (multiprocessing.shared_memory) is used to avoid pickling
the large arrays across process boundaries.

Architecture
------------
  MainProcess
    │
    ├─ builds SharedMemory blocks for u, v, lap_u, lap_v, fu, gv
    │
    ├─ Pool of P workers
    │   └─ each worker: reads slab, computes laplacian + reactions, writes
    │
    └─ main thread: combines slabs → update u, v

Note on Python GIL
------------------
numpy releases the GIL for most operations, but Python-level loops do not.
For genuine speed-ups with multiprocessing (fork-based), the overhead of
spawning/joining workers dominates at small N.  This implementation pays
off at N ≥ 256 where each slab is large enough to amortise IPC cost.
"""

import warnings
import numpy as np
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory

from pattern_formation.core.interface import BaseSolver, BaseModel
from pattern_formation.core.grid import CartesianGrid
from pattern_formation.core.laplacian import laplacian as serial_laplacian


# ── Worker function (must be top-level for pickling) ─────────────────────────

def _slab_worker(args):
    """
    Compute Laplacian and reaction terms for one horizontal slab.

    Parameters (packed into args tuple for Pool.map)
    ----------
    shm_name_u, shm_name_v : names of SharedMemory blocks
    shape   : (N, N)
    dtype   : numpy dtype
    row_start, row_end : slab rows [row_start, row_end)
    dx2     : grid.dx ** 2
    delta1, delta2 : diffusion coefficients
    dt      : time step
    model_f, model_g : callables (reaction terms, passed as lambdas)

    Returns
    -------
    row_start, row_end, lap_u_slab, lap_v_slab, fu_slab, gv_slab
    """
    (shm_name_u, shm_name_v, shape, dtype_str,
     row_start, row_end, dx2, delta1, delta2, dt,
     a, b, gamma) = args

    dtype = np.dtype(dtype_str)
    N = shape[0]

    # Attach to shared memory (read-only view)
    shm_u = SharedMemory(name=shm_name_u)
    shm_v = SharedMemory(name=shm_name_v)
    u_full = np.ndarray(shape, dtype=dtype, buffer=shm_u.buf)
    v_full = np.ndarray(shape, dtype=dtype, buffer=shm_v.buf)

    # Include halo rows (periodic wrap)
    i_prev = (row_start - 1) % N
    i_next = (row_end)     % N
    rows   = [i_prev] + list(range(row_start, row_end)) + [i_next]

    u_slab = u_full[rows, :]
    v_slab = v_full[rows, :]

    # Laplacian on interior of slab (rows 1 .. len-2)
    n_rows_halo = len(rows)
    lap_u_halo  = (
        np.roll(u_slab, -1, axis=0) + np.roll(u_slab, 1, axis=0)
        + np.roll(u_slab, -1, axis=1) + np.roll(u_slab, 1, axis=1)
        - 4.0 * u_slab
    ) / dx2
    lap_v_halo  = (
        np.roll(v_slab, -1, axis=0) + np.roll(v_slab, 1, axis=0)
        + np.roll(v_slab, -1, axis=1) + np.roll(v_slab, 1, axis=1)
        - 4.0 * v_slab
    ) / dx2

    # Strip halos (keep only interior = rows 1 .. n_rows_halo-2)
    lap_u_slab = lap_u_halo[1:-1, :]
    lap_v_slab = lap_v_halo[1:-1, :]

    # Reaction terms (Schnakenberg / giraffe for generality here)
    u_i = u_full[row_start:row_end, :]
    v_i = v_full[row_start:row_end, :]
    fu_slab  = gamma * (a - u_i + u_i**2 * v_i)
    gv_slab  = gamma * (b - u_i**2 * v_i)

    shm_u.close()
    shm_v.close()

    return row_start, row_end, lap_u_slab, lap_v_slab, fu_slab, gv_slab


# ── Parallel Explicit Solver ──────────────────────────────────────────────────

class ParallelExplicitSolver(BaseSolver):
    """
    Domain-decomposed Forward-Euler solver using multiprocessing.

    Parameters
    ----------
    model       : BaseModel   — must be a GiraffeModel (or compatible)
    grid        : CartesianGrid
    delta1/2    : diffusion coefficients
    dt          : time step
    n_workers   : number of parallel workers (default: all CPU cores)

    Notes
    -----
    Due to shared-memory and pickling constraints, this solver currently
    only supports the Schnakenberg / GiraffeModel reaction terms.
    For a fully generic parallel solver, one would use Numba or Cython.
    """

    def __init__(self, model: BaseModel, grid: CartesianGrid,
                 delta1: float, delta2: float, dt: float,
                 n_workers: int | None = None):
        super().__init__(model, grid, delta1, delta2, dt)
        self.n_workers = n_workers or cpu_count()
        self._check_stability()
        self._build_slabs()

    def _check_stability(self):
        d_max = max(self.delta1, self.delta2)
        if d_max == 0.0:
            return
        limit = self.grid.dx**2 / (4.0 * d_max)
        if self.dt > limit:
            warnings.warn(f"dt exceeds stability limit {limit:.2e}", RuntimeWarning)

    def _build_slabs(self):
        """Divide N rows into n_workers slabs as evenly as possible."""
        N = self.grid.N
        P = min(self.n_workers, N)
        sizes = [N // P + (1 if i < N % P else 0) for i in range(P)]
        starts, ends = [], []
        offset = 0
        for s in sizes:
            starts.append(offset)
            ends.append(offset + s)
            offset += s
        self._slabs = list(zip(starts, ends))

    def step(self, u: np.ndarray, v: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
        """Parallel Forward-Euler step via shared memory + Pool."""
        N     = self.grid.N
        dx2   = self.grid.dx**2
        dt    = self.dt
        shape = (N, N)
        dtype = u.dtype

        # Expose u, v via SharedMemory
        nbytes  = u.nbytes
        shm_u   = SharedMemory(create=True, size=nbytes)
        shm_v   = SharedMemory(create=True, size=nbytes)
        u_shm   = np.ndarray(shape, dtype=dtype, buffer=shm_u.buf)
        v_shm   = np.ndarray(shape, dtype=dtype, buffer=shm_v.buf)
        np.copyto(u_shm, u)
        np.copyto(v_shm, v)

        # Build argument list for each slab
        a     = self.model.a
        b     = self.model.b
        gamma = self.model.gamma

        args_list = [
            (shm_u.name, shm_v.name, shape, dtype.str,
             r0, r1, dx2, self.delta1, self.delta2, dt, a, b, gamma)
            for r0, r1 in self._slabs
        ]

        # Dispatch
        with Pool(processes=len(self._slabs)) as pool:
            results = pool.map(_slab_worker, args_list)

        # Assemble
        lap_u = np.empty_like(u)
        lap_v = np.empty_like(v)
        fu    = np.empty_like(u)
        gv    = np.empty_like(v)

        for r0, r1, lu_s, lv_s, fu_s, gv_s in results:
            lap_u[r0:r1, :] = lu_s
            lap_v[r0:r1, :] = lv_s
            fu[r0:r1, :]    = fu_s
            gv[r0:r1, :]    = gv_s

        u_new = u + dt * (self.delta1 * lap_u + fu)
        v_new = v + dt * (self.delta2 * lap_v + gv)

        shm_u.unlink(); shm_u.close()
        shm_v.unlink(); shm_v.close()

        return u_new, v_new
