import numpy as np
# Import scipy's sparse linear algebra library for solving implicit systems of equations
import scipy.sparse.linalg as splinalg
# Perfect reuse: Import the spatial difference interface we previously wrote and tested
from .UnitTest_discrete_laplacian_2d import discrete_laplacian_2d

def crank_nicolson_step(u, v, dx, dt, D_u, D_v, f_func, g_func):
    """
    Perform one Crank-Nicolson time step advancement (Semi-implicit IMEX scheme).
    The diffusion term uses C-N (implicit + explicit average), while the reaction term remains explicit to avoid solving nonlinear equations.
    
    Args:
        u, v (np.ndarray): Concentration matrices at the current time step.
        dx, dt (float): Spatial and time step sizes.
        D_u, D_v (float): Diffusion coefficients.
        f_func, g_func (callable): Chemical reaction functions.
        
    Returns:
        tuple: The concentration matrices at the next time step (u_next, v_next).
    """
    # Get the grid dimensions (N x N)
    shape = u.shape
    N2 = shape[0] * shape[1]
    
    # ==========================================
    # 1. Calculate the known part on the right side of the equation (RHS - Right Hand Side)
    # Formula: RHS = u + (dt * D / 2) * Δu + dt * f(u, v)
    # ==========================================
    
    # Calculate the spatial second derivative at the current time step (reuse interface)
    lap_u_current = discrete_laplacian_2d(u, dx)
    lap_v_current = discrete_laplacian_2d(v, dx)
    
    # Calculate the chemical reaction rate at the current time step
    reaction_u = f_func(u, v)
    reaction_v = g_func(u, v)
    
    # Assemble the right-hand side constant matrix
    rhs_u = u + (dt * D_u / 2.0) * lap_u_current + dt * reaction_u
    rhs_v = v + (dt * D_v / 2.0) * lap_v_current + dt * reaction_v
    
    # ==========================================
    # 2. Define the linear operator on the left side of the equation (LHS - Left Hand Side)
    # Formula: LHS(u_next) = u_next - (dt * D / 2) * Δ(u_next)
    # ==========================================
    
    def lhs_operator_u(u_vec):
        """Define the operation rules on the left side of the equation when solving for u_next (Matrix-Free)"""
        # Reshape the 1D vector passed in by the solver back into a 2D matrix
        u_mat = u_vec.reshape(shape)
        # Reuse interface again: calculate the Laplacian term of the unknown matrix
        lap_next = discrete_laplacian_2d(u_mat, dx)
        # Combine according to the left side formula
        result_mat = u_mat - (dt * D_u / 2.0) * lap_next
        # Flatten the result back to a 1D vector and return it to the solver
        return result_mat.flatten()

    def lhs_operator_v(v_vec):
        """Define the operation rules on the left side of the equation when solving for v_next"""
        v_mat = v_vec.reshape(shape)
        lap_next = discrete_laplacian_2d(v_mat, dx)
        result_mat = v_mat - (dt * D_v / 2.0) * lap_next
        return result_mat.flatten()

    # Wrap our defined Python functions into a linear operator (LinearOperator) recognized by Scipy
    # This is equivalent to telling the computer: "I didn't give you a matrix, but if you give me an x, I can use this operator to tell you what Ax is"
    A_u = splinalg.LinearOperator((N2, N2), matvec=lhs_operator_u)
    A_v = splinalg.LinearOperator((N2, N2), matvec=lhs_operator_v)
    
    # ==========================================
    # 3. Solve the system of linear equations A * x = b
    # ==========================================
    
    # Iterate and solve using the Conjugate Gradient (cg) method.
    # Flatten rhs as the target vector b, and set the initial guess to the current u and v
    u_next_vec, info_u = splinalg.cg(A_u, rhs_u.flatten(), x0=u.flatten())
    v_next_vec, info_v = splinalg.cg(A_v, rhs_v.flatten(), x0=v.flatten())
    
    # If solving fails (info > 0), print a warning
    if info_u != 0 or info_v != 0:
        print("Warning: The implicit solver conjugate gradient method fails to converge completely!")
        
    # Reshape the calculated 1D result vectors back into 2D matrices and return them
    u_next = u_next_vec.reshape(shape)
    v_next = v_next_vec.reshape(shape)
    
    # Return the next time step matrices
    return u_next, v_next
