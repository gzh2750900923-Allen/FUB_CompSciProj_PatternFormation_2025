# Import the NumPy library, our core tool for high-performance matrix operations in Python
import numpy as np
# Import the pyplot module from Matplotlib, used to draw our calculated matrix data as beautiful color images
import matplotlib.pyplot as plt

# [Key Modification 1] Import the Crank-Nicolson core engine function from our written implicit solver file
from implicit_CN_solver import crank_nicolson_step

# ==========================================
# 1. Define the specific chemical reaction equations for the Grey-Scott model
# ==========================================

def f_GS(u, v, alpha):
    """
    Calculate the reaction generation/consumption rate of substance u.
    Corresponds to the project documentation formula: f(u,v) = -u*v^2 + alpha*(1-u)
    """
    # Strictly write out the matrix operation according to the mathematical formula, returning the calculated result matrix
    return -u * (v**2) + alpha * (1.0 - u)

def g_GS(u, v, alpha, beta):
    """
    Calculate the reaction generation/consumption rate of substance v.
    Corresponds to the project documentation formula: g(u,v) = u*v^2 - (alpha+beta)*v
    """
    # Also strictly return the calculation result according to the mathematical formula
    return u * (v**2) - (alpha + beta) * v

# ==========================================
# 2. Main simulation and visualization function
# ==========================================

def simulate_and_visualize():
    """Configure parameters, run the loop, and display the final image"""
    
    # --- Physical and grid parameter configuration ---
    N = 100               # Set the side length of the square grid, a resolution of 100x100
    dx = 1.0              # Set the spatial grid step size, which is the physical distance between two adjacent points
    dt = 0.2              # Set the time step size
    steps = 8000          # Set the total number of time steps to simulate (since it's implicit, 8000 steps will require a longer waiting time)
    
    # --- Grey-Scott model parameters (Example: generating the classic "maze" pattern) ---
    # D_u = 0.160         # Diffusion coefficient of substance u
    # D_v = 0.080         # Diffusion coefficient of substance v
    # alpha = 0.040       # Feed rate parameter (corresponds to alpha in the equation)
    # beta = 0.020        # Kill rate parameter (changes the consumption rate, thereby changing the pattern shape)

    # --- Grey-Scott model parameters (Example: generating giraffe-like spots) ---
    D_u = 0.208           # Diffusion coefficient of substance u (it runs relatively fast)
    D_v = 0.105           # Diffusion coefficient of substance v (it runs relatively slow)
    alpha = 0.030         # Feed rate parameter (corresponds to alpha in the documentation)
    beta = 0.032          # Kill rate parameter (documentation mentions alpha+beta, so beta = 0.062 - 0.030)

    # --- Grey-Scott model parameters (Example: generating independent cheetah-like spots) ---
    #D_u = 0.160          # Diffusion coefficient of substance u
    #D_v = 0.080          # Diffusion coefficient of substance v
    #alpha = 0.024        # Feed rate parameter (corresponds to alpha in the equation): a lower feed rate helps form isolated spots
    #beta = 0.036         # Kill rate parameter (making alpha+beta = 0.060): a higher kill rate limits the expansion of the spots

    # --- Initialize the canvas (initial state) ---
    u = np.ones((N, N))   # Initially, the entire field is covered with substance u (concentration is all 1)
    v = np.zeros((N, N))  # Initially, there is absolutely no substance v on the field (concentration is all 0)
    
    # Find the center position of the grid
    center = N // 2
    # Drop a little bit of substance v (base concentration 0.25) into a 10x10 area at the center, adding tiny random perturbation (noise)
    v[center-5:center+5, center-5:center+5] = 0.25 + np.random.standard_normal((10, 10)) * 0.01
    
    print("🚀 Crank-Nicolson implicit simulation starting!")
    print("💡 Hint: The implicit method needs to solve large systems of equations for each frame. Calculation is slower, please wait patiently...")
    
    # --- Start the time engine (main loop) ---
    for i in range(steps):
        # Use lambda anonymous functions to wrap the GS functions with specific alpha and beta parameters into a form that only takes (u, v)
        current_f = lambda u_mat, v_mat: f_GS(u_mat, v_mat, alpha)
        current_g = lambda u_mat, v_mat: g_GS(u_mat, v_mat, alpha, beta)
        
        # [Key Modification 2] Call our Crank-Nicolson implicit engine to calculate the next frame of u and v
        u_next, v_next = crank_nicolson_step(u, v, dx, dt, D_u, D_v, current_f, current_g)
        
        # Update the "future" state calculated to the "current" state, ready to enter the next loop
        u = u_next
        v = v_next
        
        # Because implicit calculation is slower, print the progress every 500 steps to relieve waiting anxiety
        if i % 500 == 0:
            print(f"⏳ Calculated {i}/{steps} steps...")
            
    print("✅ Simulation complete! Generating image...")

    # --- Visual output ---
    plt.figure(figsize=(6, 6))                 # Create a 6x6 inch blank canvas
    # Draw the concentration matrix of substance v using imshow; cmap='YlOrBr' is the yellow-brown palette commonly used for giraffes
    plt.imshow(v, cmap='YlOrBr', interpolation='bilinear') 
    
    # [Key Modification 3] Change the picture title to clearly indicate that this is the result generated by the implicit method
    plt.title("Grey-Scott Turing Pattern (Crank-Nicolson Method)") 
    plt.axis('off')                            # Turn off axis ticks to make the picture cleaner
    plt.show()                                 # Pop up a window to show the final pattern

# If this script is run directly, execute the main function above
if __name__ == '__main__':
    simulate_and_visualize()