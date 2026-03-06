import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def solve_crank_nicolson(L=1.0, T=0.1, nx=100, nt=200, D=1.0):
    """
    Solves the 1D Heat Equation using the Crank-Nicolson method.
    
    Parameters:
    L  : Length of the domain
    T  : Total simulation time
    nx : Number of spatial grid points
    nt : Number of time steps
    D  : Diffusion coefficient
    """
    
    dx = L / (nx - 1)
    dt = T / nt
    alpha = (D * dt) / (2 * dx**2)

    # Spatial grid
    x = np.linspace(0, L, nx)
    
    # Initial condition: A Gaussian pulse in the middle
    u = np.exp(-100 * (x - 0.5)**2)
    
    # Boundary conditions (Dirichlet)
    u[0] = 0
    u[-1] = 0

    # Matrix construction for Au(n+1) = Bu(n)
    # Since A is tridiagonal, we use solve_banded format (3, nx-2)
    # A_banded contains [upper diagonal, main diagonal, lower diagonal]
    main_diag = (1 + 2 * alpha) * np.ones(nx - 2)
    off_diag = -alpha * np.ones(nx - 3)
    
    # For solve_banded, we need to pad the diagonals
    ab = np.zeros((3, nx - 2))
    ab[0, 1:] = off_diag  # Upper diagonal
    ab[1, :] = main_diag  # Main diagonal
    ab[2, :-1] = off_diag # Lower diagonal

    # Simulation loop
    for n in range(nt):
        # Construct the RHS vector (B * u_n)
        # Internal points only
        rhs = (alpha * u[:-2] + 
               (1 - 2 * alpha) * u[1:-1] + 
               alpha * u[2:])
        
        # Solve the tridiagonal system for the next time step
        u[1:-1] = solve_banded((1, 1), ab, rhs)

    return x, u

# Parameters
x, final_u = solve_crank_nicolson()

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(x, np.exp(-100 * (x - 0.5)**2), '--', label='Initial (t=0)')
plt.plot(x, final_u, label='Final (t=0.1)')
plt.title("1D Diffusion using Crank-Nicolson Method")
plt.xlabel("Position (x)")
plt.ylabel("Concentration (u)")
plt.legend()
plt.grid(True)
plt.show()
