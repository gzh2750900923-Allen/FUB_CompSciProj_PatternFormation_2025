import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
# [Key Fix] Add the root directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# [Key Fix] Update imports to use the package namespace
from pattern_formation.explicit_solver import explicit_euler_step
from pattern_formation.implicit_CN_solver import crank_nicolson_step


#from explicit_solver import explicit_euler_step
#from implicit_CN_solver import crank_nicolson_step

# ==========================================
# 1. Reaction Models [cite: 18, 40]
# ==========================================

def grey_scott_reaction(u, v, alpha, beta):
    """Standard Grey-Scott equations """
    f = -u * (v**2) + alpha * (1.0 - u)
    g = u * (v**2) - (alpha + beta) * v
    return f, g

# ==========================================
# 2. Pattern Generation Engine
# ==========================================

def run_animal_simulation(animal_type, use_implicit=False):
    """
    Runs simulation for Task 3.3.
    For Leopard, it implements a Two-stage model.
    """
    # Common Grid Parameters 
    N, dx, dt = 100, 1.0, 0.2
    steps = 8000
    
    # Initial State
    u = np.ones((N, N))
    v = np.zeros((N, N))
    center = N // 2
    v[center-5:center+5, center-5:center+5] = 0.25 + np.random.randn(10, 10) * 0.01

    # Parameter Selection [cite: 20, 22]
    if animal_type == "giraffe":
        D_u, D_v = 0.208, 0.105
        alpha_base, beta_base = 0.030, 0.032
    else: # leopard
        D_u, D_v = 0.160, 0.080
        alpha_base, beta_base = 0.024, 0.036

    start_time = time.time()
    
    for i in range(steps):
        # Two-stage Logic for Leopard 
        # Stage 2: Modulate parameters halfway to create Rosettes
        if animal_type == "leopard" and i == steps // 2:
            # Inference: Local feed rate is increased where spots already exist
            alpha_current = np.where(v > 0.1, alpha_base * 1.5, alpha_base)
            beta_current = beta_base
        else:
            alpha_current = alpha_base
            beta_current = beta_base

        # Wrapper functions for the solvers
        f_func = lambda u_m, v_m: -u_m * (v_m**2) + alpha_current * (1.0 - u_m)
        g_func = lambda u_m, v_m: u_m * (v_m**2) - (alpha_current + beta_current) * v_m

        if use_implicit:
            u, v = crank_nicolson_step(u, v, dx, dt, D_u, D_v, f_func, g_func)
        else:
            u, v = explicit_euler_step(u, v, dx, dt, D_u, D_v, f_func, g_func)

    runtime = time.time() - start_time
    return v, runtime

# ==========================================
# 3. Main Comparison & Execution 
# ==========================================

def run_task_3_3():
    animals = ["giraffe", "leopard"]
    
    for animal in animals:
        print(f"\n🐾 Simulating markings of a {animal.upper()}...")
        
        # Run Explicit
        v_exp, time_exp = run_animal_simulation(animal, use_implicit=False)
        print(f"   - Explicit Euler: {time_exp:.2f}s")
        
        # Run Implicit (CN)
        v_imp, time_imp = run_animal_simulation(animal, use_implicit=True)
        print(f"   - Crank-Nicolson: {time_imp:.2f}s")

        # Visual Comparison 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(v_exp, cmap='YlOrBr', interpolation='bilinear')
        ax1.set_title(f"Explicit {animal.capitalize()}\nTime: {time_exp:.2f}s")
        ax1.axis('off')
        
        ax2.imshow(v_imp, cmap='YlOrBr', interpolation='bilinear')
        ax2.set_title(f"Crank-Nicolson {animal.capitalize()}\nTime: {time_imp:.2f}s")
        ax2.axis('off')
        
        plt.suptitle(f"Task 3.3: {animal.capitalize()} Pattern Formation Analysis")
        plt.show()

if __name__ == "__main__":
    run_task_3_3()
