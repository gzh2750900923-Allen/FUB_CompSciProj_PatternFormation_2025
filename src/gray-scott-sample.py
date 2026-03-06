import numpy as np
import matplotlib.pyplot as plt

def solve_gray_scott(size=256, iterations=10000, dt=1.0, du=0.16, dv=0.08, f=0.035, k=0.06):
    """
    Simulates the Gray-Scott reaction-diffusion system.
    
    Parameters:
    size : Grid size (size x size)
    iterations : Number of time steps
    dt : Time step increment
    du, dv : Diffusion coefficients for U and V
    f : Feed rate
    k : Kill rate
    """
    
    # Initialize U with 1.0 and V with 0.0
    u = np.ones((size, size))
    v = np.zeros((size, size))

    # Seed the center with V to start the reaction
    r = 10
    u[size//2-r:size//2+r, size//2-r:size//2+r] = 0.5
    v[size//2-r:size//2+r, size//2-r:size//2+r] = 0.25

    # Add some noise to break symmetry
    u += np.random.normal(scale=0.01, size=(size, size))
    v += np.random.normal(scale=0.01, size=(size, size))

    def laplacian(z):
        """
        Calculates the discrete Laplacian using a five-point stencil
        with periodic boundary conditions.
        """
        return (np.roll(z, 1, axis=0) + np.roll(z, -1, axis=0) +
                np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1) - 4*z)

    # Time integration loop
    for i in range(iterations):
        lu = laplacian(u)
        lv = laplacian(v)
        
        # Calculate the reaction term UV^2
        uvv = u * v**2
        
        # Update U and V based on the Gray-Scott equations
        u += (du * lu - uvv + f * (1 - u)) * dt
        v += (dv * lv + uvv - (f + k) * v) * dt
        
        # Keep values within physical bounds [0, 1]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

    return v

# Run simulation
# Parameters for 'Coral' or 'Mitosis' like patterns
result = solve_gray_scott(f=0.0367, k=0.0649)

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(result, cmap='magma')
plt.title("Gray-Scott Reaction-Diffusion Pattern")
plt.axis('off')
plt.show()
