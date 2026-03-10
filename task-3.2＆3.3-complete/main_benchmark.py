# Import the NumPy library for efficient matrix mathematical operations
import numpy as np
# Import Matplotlib for the final side-by-side image rendering
import matplotlib.pyplot as plt
# Import the time module, which is our stopwatch for "performance benchmarking"
import time

# Import the respective core step functions from our two independent engine files
from explicit_solver import explicit_euler_step
from implicit_CN_solver import crank_nicolson_step

# ==========================================
# 1. Define the Grey-Scott chemical reaction equations (fuel)
# ==========================================

def f_GS(u, v, alpha):
    """Calculate the reaction generation/consumption rate of substance u"""
    return -u * (v**2) + alpha * (1.0 - u)

def g_GS(u, v, alpha, beta):
    """Calculate the reaction generation/consumption rate of substance v"""
    return u * (v**2) - (alpha + beta) * v

# ==========================================
# 2. Core benchmarking and side-by-side visualization function
# ==========================================

def run_benchmark():
    """Run a performance comparison of explicit and implicit methods and display the results side-by-side"""
    
    # --- Unified physical and grid parameter configuration (to ensure a fair comparison) ---
    N = 100               # Set the side length of the square grid (100x100)
    dx = 1.0              # Set the spatial step size
    dt = 0.2              # Set the time step size
    steps = 8000          # Increased to 8000 steps to ensure the animal patterns mature sufficiently
    
    # ==========================================
    # --- Grey-Scott model parameter configuration ---
    # ==========================================
    
    # [Currently Active]: Giraffe Pattern parameters
    #D_u = 0.208           # Diffusion coefficient of substance u
    #D_v = 0.105           # Diffusion coefficient of substance v
    #alpha = 0.030         # Feed rate
    #beta = 0.032          # Kill rate

    # [Cold Switch Backup]: Leopard Pattern approximate parameters
    # 💡 Usage instructions: If you want to test leopard spots, please comment out the four lines above by adding # at the beginning,
    # and remove the # at the beginning of the four lines below.
    D_u = 0.160
    D_v = 0.080
    alpha = 0.024
    beta = 0.036
    
    # ==========================================
    
    # Wrap the reaction functions so they only receive the two variables u and v
    current_f = lambda u_mat, v_mat: f_GS(u_mat, v_mat, alpha)
    current_g = lambda u_mat, v_mat: g_GS(u_mat, v_mat, alpha, beta)

    # --- Prepare the initial canvas (generate an identical initial state) ---
    def get_initial_state():
        """Generate and return a brand new initial concentration matrix to ensure both engines start from the exact same point"""
        u_init = np.ones((N, N))
        v_init = np.zeros((N, N))
        center = N // 2
        # Add high concentration of v and random noise in the center area to trigger the reaction
        v_init[center-5:center+5, center-5:center+5] = 0.25 + np.random.standard_normal((10, 10)) * 0.01
        return u_init, v_init

    print(f"🏁 Benchmark test starting (Total steps: {steps}) 🏁\n")

    # ==========================================
    # First Test: Explicit Euler method
    # ==========================================
    print("👉 Running Explicit Euler method...")
    u_exp, v_exp = get_initial_state()
    start_time_exp = time.time()
    
    for i in range(steps):
        u_exp, v_exp = explicit_euler_step(u_exp, v_exp, dx, dt, D_u, D_v, current_f, current_g)
        
    time_exp = time.time() - start_time_exp
    print(f"✅ Explicit method complete! Time taken: {time_exp:.3f} seconds\n")

    # ==========================================
    # Second Test: Crank-Nicolson implicit method
    # ==========================================
    print("👉 Running Crank-Nicolson implicit method...")
    u_imp, v_imp = get_initial_state()
    start_time_imp = time.time()
    
    for i in range(steps):
        u_imp, v_imp = crank_nicolson_step(u_imp, v_imp, dx, dt, D_u, D_v, current_f, current_g)
        
    time_imp = time.time() - start_time_imp
    print(f"✅ Implicit method complete! Time taken: {time_imp:.3f} seconds\n")

    # ==========================================
    # 3. Quality and result side-by-side visualization (Quality Comparison)
    # ==========================================
    print("📊 Generating comparison images...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(v_exp, cmap='YlOrBr', interpolation='bilinear')
    ax1.set_title(f"Explicit Method\nRuntime: {time_exp:.2f} s")
    ax1.axis('off')
    
    im2 = ax2.imshow(v_imp, cmap='YlOrBr', interpolation='bilinear')
    ax2.set_title(f"Crank-Nicolson (Implicit) Method\nRuntime: {time_imp:.2f} s")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_benchmark()