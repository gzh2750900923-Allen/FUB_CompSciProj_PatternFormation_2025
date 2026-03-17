import numpy as np

def f_leopard_stage2(u, v, alpha_map, beta_map):
    """
    [Inference] Spatially varying Feed rate (alpha) to create Rosettes.
    alpha_map and beta_map are matrices of the same shape as u and v.
    """
    return -u * (v**2) + alpha_map * (1.0 - u)

def g_leopard_stage2(u, v, alpha_map, beta_map):
    """
    [Inference] Spatially varying Kill rate (beta).
    """
    return u * (v**2) - (alpha_map + beta_map) * v

def generate_leopard_pattern(N, steps_stage1=4000, steps_stage2=4000):
    """
    [Confirmed] Two-stage simulation for Leopard Rosettes.
    """
    # Standard Stage 1 parameters for spots
    D_u, D_v = 0.16, 0.08
    alpha_1, beta_1 = 0.024, 0.036
    
    # Initialize
    u = np.ones((N, N))
    v = np.zeros((N, N))
    center = N // 2
    v[center-5:center+5, center-5:center+5] = 0.25 + np.random.randn(10, 10) * 0.01
    
    # --- STAGE 1: Formation of initial spots ---
    # (Existing explicit_euler_step loop with constant alpha_1, beta_1)
    # ... 

    # --- STAGE 2: Modulation based on Stage 1 result ---
    # [Inference] We create a parameter map where alpha is higher 
    # only where substance 'v' already exists.
    alpha_map = np.full((N, N), 0.024)
    alpha_map[v > 0.1] = 0.040  # Increase feed rate inside existing spots
    
    beta_map = np.full((N, N), 0.036)
    # Modulation creates the "hollow" effect in the center
    
    # --- Continue simulation with alpha_map ---
    # (The solver will now use f_leopard_stage2 with alpha_map)
