import numpy as np
import matplotlib.pyplot as plt
import time
from explicit_solver import explicit_euler_step
from implicit_CN_solver import crank_nicolson_step

def get_l2_error(u_approx, u_ref):
    """
    Calculates the relative L2 norm error between the numerical approximation 
    and the high-precision reference solution.
    """
    return np.sqrt(np.sum((u_approx - u_ref)**2) / np.sum(u_ref**2))

def run_convergence_analysis():
    """
    Performs temporal convergence analysis to validate the order of accuracy 
    for both Explicit Euler and Crank-Nicolson schemes.
    """
    # --- Configuration ---
    N = 64                # Grid resolution
    dx = 1.0              # Spatial step size
    T_final = 5.0         # Total simulation time for validation
    
    # Grey-Scott parameters (Stable regime for convergence testing)
    D_u, D_v = 0.16, 0.08
    alpha, beta = 0.035, 0.060
    f_func = lambda u, v: -u * (v**2) + alpha * (1.0 - u)
    g_func = lambda u, v: u * (v**2) - (alpha + beta) * v

    # --- 1. Generate High-Precision Reference Solution ---
    # We use a very small dt to treat this result as the "exact" solution
    dt_ref = 0.0005
    steps_ref = int(T_final / dt_ref)
    
    # Initial Conditions (Gaussian perturbation at center)
    u_init = np.ones((N, N))
    v_init = np.zeros((N, N))
    center = N // 2
    v_init[center-3:center+3, center-3:center+3] = 0.25
    
    print(f"Generating reference solution (dt={dt_ref})...")
    u_curr, v_curr = u_init.copy(), v_init.copy()
    for _ in range(steps_ref):
        u_curr, v_curr = explicit_euler_step(u_curr, v_curr, dx, dt_ref, D_u, D_v, f_func, g_func)
    
    u_ref_final = u_curr
    print("Reference solution generated successfully.\n")

    # --- 2. Temporal Convergence Test ---
    # Testing different time steps to observe error decay rates
    dt_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    errors_explicit = []
    errors_implicit = []

    for dt in dt_list:
        steps = int(T_final / dt)
        print(f"Testing dt = {dt} ({steps} steps)...")
        
        # Test Explicit Euler (Expected O(dt^1))
        u_exp, v_exp = u_init.copy(), v_init.copy()
        for _ in range(steps):
            u_exp, v_exp = explicit_euler_step(u_exp, v_exp, dx, dt, D_u, D_v, f_func, g_func)
        errors_explicit.append(get_l2_error(u_exp, u_ref_final))

        # Test Crank-Nicolson (Expected O(dt^2))
        u_imp, v_imp = u_init.copy(), v_init.copy()
        for _ in range(steps):
            u_imp, v_imp = crank_nicolson_step(u_imp, v_imp, dx, dt, D_u, D_v, f_func, g_func)
        errors_implicit.append(get_l2_error(u_imp, u_ref_final))

    # --- 3. Visualization and Slope Calculation ---
    plt.figure(figsize=(10, 7))
    
    # Plot numerical results
    plt.loglog(dt_list, errors_explicit, 'ro-', label='Explicit Euler (O(dt))')
    plt.loglog(dt_list, errors_implicit, 'bs-', label='Crank-Nicolson (O(dt^2))')
    
    # Plot theoretical slopes for comparison
    plt.loglog(dt_list, [dt * (errors_explicit[0]/dt_list[0]) for dt in dt_list], 'k--', alpha=0.5, label='Theoretical Order 1')
    plt.loglog(dt_list, [(dt**2) * (errors_implicit[0]/dt_list[0]**2) for dt in dt_list], 'k:', alpha=0.5, label='Theoretical Order 2')

    plt.xlabel('Time Step (dt)', fontsize=12)
    plt.ylabel('L2 Relative Error', fontsize=12)
    plt.title('Temporal Convergence Analysis: Explicit vs Implicit', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    print("\nConvergence analysis complete. Displaying plot...")
    plt.show()

if __name__ == "__main__":
    run_convergence_analysis()
