import numpy as np
import matplotlib.pyplot as plt

def generate_giraffe_pattern():
    # --- Grid and Time Parameters ---
    N = 100          # Grid size (NxN)
    dx = 1.0         # Spatial step
    dt = 0.2         # Time step (Must satisfy CFL condition for stability)
    steps = 15000    # Total iterations to reach steady pattern

    # --- Grey-Scott Model Parameters for Giraffe (Plates) ---
    # These parameters are tuned to produce large polygonal domains
    Du, Dv = 0.208, 0.105   # Diffusion rates for U and V
    f = 0.030               # Feed rate (alpha)
    k = 0.062               # Kill rate (alpha + beta)

    # --- Initialization ---
    # Start with a homogeneous state of U=1.0, V=0.0
    u = np.ones((N, N))
    v = np.zeros((N, N))

    # Introduce random noise in the center to trigger Turing Instability
    center_slice = slice(N//2 - 5, N//2 + 5)
    v[center_slice, center_slice] = 0.25 + np.random.standard_normal((10, 10)) * 0.01

    # --- Main Simulation Loop ---
    for i in range(steps):
        # Calculate Laplacian using 2nd order Finite Difference
        # We use np.roll for periodic boundary conditions
        lu = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
              np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u) / dx**2
        lv = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
              np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v) / dx**2

        # Reaction terms f(u,v) and g(u,v)
        uv2 = u * v**2
        reaction_u = -uv2 + f * (1 - u)
        reaction_v = uv2 - k * v

        # Time integration (Forward Euler)
        u += dt * (Du * lu + reaction_u)
        v += dt * (Dv * lv + reaction_v)

        # Optional: Print progress
        if i % 5000 == 0:
            print(f"Simulation Progress: {i}/{steps} steps")

    return v

# --- Visualization ---
pattern = generate_giraffe_pattern()
plt.figure(figsize=(8, 8))
# 'YlOrBr' colormap simulates the natural colors of a giraffe fur
plt.imshow(pattern, cmap='YlOrBr', interpolation='bilinear')
plt.title("Giraffe Fur Pattern Simulation (Grey-Scott Model)")
plt.axis('off')
plt.show()
