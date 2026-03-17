import numpy as np
import dask.array as da
import scipy.sparse.linalg as splinalg
from .dask_explicit_solver import dask_laplacian_2d

def dask_crank_nicolson_step(u_da, v_da, dx, dt, D_u, D_v, f_func, g_func):
    """
    Performs one Crank-Nicolson time step using Dask-accelerated Matrix-Free 
    linear solvers[cite: 16, 30, 31].
    
    The diffusion term is treated implicitly using the CN scheme, 
    while the reaction term is treated explicitly (IMEX).
    """
    # Get grid metadata for reshaping
    shape = u_da.shape
    chunks = u_da.chunks
    N2 = shape[0] * shape[1]

    # --- 1. Compute Right-Hand Side (RHS) in Parallel ---
    # Formula: RHS = u + (dt*D/2)*Laplacian(u) + dt*f(u,v) [cite: 16]
    lap_u_curr = dask_laplacian_2d(u_da, dx)
    lap_v_curr = dask_laplacian_2d(v_da, dx)
    
    rhs_u_da = u_da + (dt * D_u / 2.0) * lap_u_curr + dt * f_func(u_da, v_da)
    rhs_v_da = v_da + (dt * D_v / 2.0) * lap_v_curr + dt * g_func(u_da, v_da)
    
    # Materialize RHS as NumPy for the Scipy solver
    rhs_u = rhs_u_da.compute().flatten()
    rhs_v = rhs_v_da.compute().flatten()

    # --- 2. Define the Parallel LHS Operator (Matrix-Free) ---
    # Formula: LHS * u_next = u_next - (dt*D/2)*Laplacian(u_next) [cite: 16, 31]
    
    def create_lhs_op(D):
        def matvec(v_np):
            # 1. Convert NumPy vector back to Dask Array
            v_da = da.from_array(v_np.reshape(shape), chunks=chunks)
            # 2. Compute Laplacian in parallel across cores
            lap = dask_laplacian_2d(v_da, dx)
            # 3. Apply the LHS formula
            res_da = v_da - (dt * D / 2.0) * lap
            # 4. Compute and return to the CG solver as a NumPy vector
            return res_da.compute().flatten()
        return matvec

    # Wrap as Scipy LinearOperators 
    A_u = splinalg.LinearOperator((N2, N2), matvec=create_lhs_op(D_u))
    A_v = splinalg.LinearOperator((N2, N2), matvec=create_lhs_op(D_v))

    # --- 3. Solve the Linear System (Ax = b) ---
    # Initial guess is the current state (warm start)
    u_next_flat, info_u = splinalg.cg(A_u, rhs_u, x0=u_da.compute().flatten())
    v_next_flat, info_v = splinalg.cg(A_v, rhs_v, x0=v_da.compute().flatten())

    if info_u != 0 or info_v != 0:
        print("Warning: Dask-CG solver failed to reach full convergence.")

    # Convert results back to Dask Arrays for the next step
    u_next = da.from_array(u_next_flat.reshape(shape), chunks=chunks)
    v_next = da.from_array(v_next_flat.reshape(shape), chunks=chunks)

    return u_next, v_next
