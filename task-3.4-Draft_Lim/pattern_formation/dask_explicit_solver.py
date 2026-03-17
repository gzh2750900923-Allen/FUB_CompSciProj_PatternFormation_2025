import numpy as np
import dask.array as da

def dask_laplacian_2d(u_dist, dx, depth=1):
    """
    Calculates the Laplacian using Dask's map_overlap to handle 
    periodic boundary conditions across chunks. [cite: 4, 15]
    """
    def local_laplacian(u_chunk):
        # Shift operators for the 5-point stencil
        u_up = np.roll(u_chunk, 1, axis=0)
        u_down = np.roll(u_chunk, -1, axis=0)
        u_left = np.roll(u_chunk, 1, axis=1)
        u_right = np.roll(u_chunk, -1, axis=1)
        return (u_up + u_down + u_left + u_right - 4.0 * u_chunk) / (dx ** 2)

    # map_overlap handles the 'ghost cells' between dask chunks.
    # boundary='periodic' ensures our BCs are respected. [cite: 4]
    return u_dist.map_overlap(local_laplacian, depth=depth, boundary='periodic')

def dask_explicit_euler_step(u_da, v_da, dx, dt, D_u, D_v, f_func, g_func):
    """
    Performs one explicit Euler step using Dask parallel arrays. [cite: 15, 30]
    """
    # 1. Compute Diffusion terms in parallel
    lap_u = dask_laplacian_2d(u_da, dx)
    lap_v = dask_laplacian_2d(v_da, dx)
    
    # 2. Compute Reaction terms
    # f_func and g_func must be written to support dask/numpy broadcasting
    react_u = f_func(u_da, v_da)
    react_v = g_func(u_da, v_da)
    
    # 3. Update step: u_{t+\Delta t} = u_t + \Delta t(D\Delta u + f) [cite: 4]
    u_next = u_da + dt * (D_u * lap_u + react_u)
    v_next = v_da + dt * (D_v * lap_v + react_v)
    
    return u_next, v_next
