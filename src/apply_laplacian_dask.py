import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from dask.array import map_overlap

def apply_laplacian_dask(block, dx=1.0):
    """
    Parallel Laplacian calculation using 2nd order Finite Difference.
    The 'block' includes ghost cells from neighboring chunks.

    """
    # 5-point stencil: (up + down + left + right - 4*center) / dx^2
    # Boundary padding is handled by Dask's map_overlap depth
    lap = (block[2:, 1:-1] + block[:-2, 1:-1] +
           block[1:-1, 2:] + block[1:-1, :-2] - 4*block[1:-1, 1:-1]) / dx**2
    return lap

def simulate_giraffe_parallel():
    # --- 1. Configuration ---
    N = 1000                # Large scale grid for high-resolution fur
    chunks = (500, 500)     # Divide grid into 4 chunks for parallel processing
    dx, dt = 1.0, 0.2
    steps = 5000

    # --- 2. Giraffe Parameters (Grey-Scott) ---
    Du, Dv = 0.208, 0.105
    f, k = 0.030, 0.062

    # --- 3. Initialize Dask Arrays ---
    u_np = np.ones((N, N))
    v_np = np.zeros((N, N))
    # Seed the pattern with noise in the center
    v_np[N//2-20:N//2+20, N//2-20:N//2+20] = 0.25 + np.random.randn(40, 40) * 0.01

    u = da.from_array(u_np, chunks=chunks)
    v = da.from_array(v_np, chunks=chunks)

    # --- 4. Simulation Loop ---
    for i in range(steps):
        # Calculate Laplacian in parallel with 1-pixel overlap (depth=1)
        lu = u.map_overlap(apply_laplacian_dask, depth=1, boundary='periodic', dx=dx)
        lv = v.map_overlap(apply_laplacian_dask, depth=1, boundary='periodic', dx=dx)

        # Element-wise reaction terms (Embarrassingly parallel)
        uv2 = u * v**2
        du = Du * lu - uv2 + f * (1 - u)
        dv = Dv * lv + uv2 - k * v

        # Update fields
        u = u + dt * du
        v = v + dt * dv

        if i % 1000 == 0:
            print(f"Step {i}: Parallel computation in progress...")

    # Compute the final graph and bring results to memory
    return v.compute()

# --- 5. Visualize Results ---
giraffe_fur = simulate_giraffe_parallel()
plt.figure(figsize=(10, 10))
plt.imshow(giraffe_fur, cmap='YlOrBr')
plt.title("High-Resolution Giraffe Pattern (DASK Parallelized)")
plt.axis('off')
plt.show()
