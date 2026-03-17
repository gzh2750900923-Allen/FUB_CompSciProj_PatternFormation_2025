import os
import sys
import time
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the parallel solvers we've engineered
from pattern_formation.dask_explicit_solver import dask_explicit_euler_step
from pattern_formation.dask_implicit_solver import dask_crank_nicolson_step

#from dask_explicit_solver import dask_explicit_euler_step
#from dask_implicit_solver import dask_crank_nicolson_step

def run_dask_benchmark(N=256, chunks=(128, 128)):
    """
    Comprehensive parallel benchmark for Task 3.4.
    Compares Dask Explicit vs. Dask Implicit (CN) performance.
    """
    # --- Configuration ---
    dx, dt = 1.0, 0.2
    steps = 500  # Number of iterations for benchmarking
    
    # Parameters for Leopard Rosettes [cite: 22, 40]
    D_u, D_v = 0.160, 0.080
    alpha_base, beta_base = 0.024, 0.036
    
    # --- Initialization ---
    u_np = np.ones((N, N))
    v_np = np.zeros((N, N))
    center = N // 2
    v_np[center-5:center+5, center-5:center+5] = 0.25 + np.random.randn(10, 10) * 0.01
    
    def get_dask_init():
        return da.from_array(u_np, chunks=chunks), da.from_array(v_np, chunks=chunks)

    # ======================================================
    # 1. Benchmark: Parallel Explicit Euler
    # ======================================================
    print(f"\n🚀 Starting Dask Explicit Solver (Grid: {N}x{N})")
    u_da, v_da = get_dask_init()
    start_time = time.time()
    
    for i in range(steps):
        # Two-stage logic for leopard patterns 
        if i == steps // 2:
            alpha_curr = da.where(v_da > 0.1, alpha_base * 1.5, alpha_base)
        else:
            alpha_curr = alpha_base
            
        f_gs = lambda u, v: -u * (v**2) + alpha_curr * (1.0 - u)
        g_gs = lambda u, v: u * (v**2) - (alpha_curr + beta_base) * v
        
        u_da, v_da = dask_explicit_euler_step(u_da, v_da, dx, dt, D_u, D_v, f_gs, g_gs)
    
    v_exp_final = v_da.compute() # Synchronization point
    time_exp = time.time() - start_time
    print(f"✅ Explicit Runtime: {time_exp:.3f} seconds")

    # ======================================================
    # 2. Benchmark: Parallel Crank-Nicolson (Implicit)
    # ======================================================
    print(f"\n🚀 Starting Dask Implicit Solver (Matrix-Free CG)")
    u_da, v_da = get_dask_init()
    start_time = time.time()
    
    for i in range(steps):
        # Using the same reaction terms for consistency
        if i == steps // 2:
            alpha_curr = da.where(v_da > 0.1, alpha_base * 1.5, alpha_base)
        else:
            alpha_curr = alpha_base
            
        f_gs = lambda u, v: -u * (v**2) + alpha_curr * (1.0 - u)
        g_gs = lambda u, v: u * (v**2) - (alpha_curr + beta_base) * v
        
        # Parallel CN solver [cite: 31]
        u_da, v_da = dask_crank_nicolson_step(u_da, v_da, dx, dt, D_u, D_v, f_gs, g_gs)
        
        if i % 100 == 0:
            print(f"⏳ Step {i}/{steps}...")

    v_imp_final = v_da.compute()
    time_imp = time.time() - start_time
    print(f"✅ Implicit Runtime: {time_imp:.3f} seconds")

    # --- Visualization & Comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(v_exp_final, cmap='YlOrBr', interpolation='bilinear')
    ax1.set_title(f"Dask Explicit\nTime: {time_exp:.2f}s")
    ax1.axis('off')
    
    ax2.imshow(v_imp_final, cmap='YlOrBr', interpolation='bilinear')
    ax2.set_title(f"Dask Implicit (Matrix-Free)\nTime: {time_imp:.2f}s")
    ax2.axis('off')
    
    plt.suptitle(f"Task 3.4 Benchmark: Parallel Performance Comparison (N={N})")
    plt.show()

if __name__ == "__main__":
    # Initialize Dask Client for local parallelization [cite: 30]
    client = Client()
    print(f"📊 Dashboard Link: {client.dashboard_link}")
    
    try:
        # Note: Increase N to 1024 to see Dask's true power vs single-thread!
        run_dask_benchmark(N=256, chunks=(128, 128))
    finally:
        client.close()
