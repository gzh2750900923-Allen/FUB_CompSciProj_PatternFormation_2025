import numpy as np
# Import the spatial difference interface from our previously written and tested file
from UnitTest_discrete_laplacian_2d import discrete_laplacian_2d

def explicit_euler_step(u, v, dx, dt, D_u, D_v, f_func, g_func):
    """
    Performs a single explicit Euler time step advancement.
    
    Args:
        u (np.ndarray): The concentration matrix of substance u at the current time step.
        v (np.ndarray): The concentration matrix of substance v at the current time step.
        dx (float): Spatial grid step size.
        dt (float): Time step size.
        D_u (float): Diffusion coefficient for substance u (corresponds to delta_1 in the documentation).
        D_v (float): Diffusion coefficient for substance v (corresponds to delta_2 in the documentation).
        f_func (callable): Reaction term function for substance u, takes (u, v) and returns the computed matrix.
        g_func (callable): Reaction term function for substance v, takes (u, v) and returns the computed matrix.
        
    Returns:
        tuple: A tuple containing the next time step's concentration matrices (u_next, v_next).
    """
    
    # 1. Compute spatial diffusion terms (calling our previously written and tested interface)
    # Compute the spatial second derivative of substance u (Δu)
    laplacian_u = discrete_laplacian_2d(u, dx)
    # Compute the spatial second derivative of substance v (Δv)
    laplacian_v = discrete_laplacian_2d(v, dx)
    
    # 2. Compute chemical reaction terms (calling the specific reaction functions passed as arguments)
    # Compute the reaction rate of substance u at current concentrations f(u, v)
    reaction_u = f_func(u, v)
    # Compute the reaction rate of substance v at current concentrations g(u, v)
    reaction_v = g_func(u, v)
    
    # 3. Explicit Euler time advancement (direct implementation of the core physical formula)
    # Next u = Current u + Time step * (Diffusion rate * Spatial difference + Reaction term)
    u_next = u + dt * (D_u * laplacian_u + reaction_u)
    
    # Next v = Current v + Time step * (Diffusion rate * Spatial difference + Reaction term)
    v_next = v + dt * (D_v * laplacian_v + reaction_v)
    
    # Return the pair of computed new matrices for use in the next loop iteration
    return u_next, v_next