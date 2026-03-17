import numpy as np
import unittest

# ==========================================
# 1. Core Interface
# ==========================================

def discrete_laplacian_2d(u, dx):
    """
    Calculates the discrete Laplacian operator on a 2D Cartesian grid.
    Uses a second-order central finite difference approximation and strictly includes periodic boundary conditions.
    
    Args:
        u (np.ndarray): The 2D concentration matrix at the current time step.
        dx (float): Spatial grid step size (assuming dx = dy).
        
    Returns:
        np.ndarray: The computed spatial second derivative matrix (Δu).
    """
    # Use np.roll to implement overall matrix shifting, naturally fitting "periodic boundary conditions".
    # np.roll(matrix, shift_amount, axis) 
    # axis=0 shifts vertically (up/down), axis=1 shifts horizontally (left/right).
    
    # Shift the entire matrix down by one step; the bottom row wraps around to the top. Equivalent to getting the "upper" neighbor of each point.
    u_up = np.roll(u, 1, axis=0)
    
    # Shift the entire matrix up by one step. Equivalent to getting the "lower" neighbor of each point.
    u_down = np.roll(u, -1, axis=0)
    
    # Shift the entire matrix right by one step. Equivalent to getting the "left" neighbor of each point.
    u_left = np.roll(u, 1, axis=1)
    
    # Shift the entire matrix left by one step. Equivalent to getting the "right" neighbor of each point.
    u_right = np.roll(u, -1, axis=1)
    
    # Apply the second-order central difference formula: (up + down + left + right - 4*center) / dx^2
    laplacian = (u_up + u_down + u_left + u_right - 4.0 * u) / (dx ** 2)
    
    return laplacian


# ==========================================
# 2. Unit Tests
# ==========================================

class TestFiniteDifference(unittest.TestCase):
    """
    Test class to verify the accuracy of the discrete_laplacian_2d interface.
    Uses known, simple mathematical edge cases to verify the logic.
    """
    
    def test_constant_matrix(self):
        """
        Test Case 1: Completely flat concentration field (constant matrix).
        Physical meaning: If the concentration is perfectly uniform everywhere, no diffusion occurs, and the result must be all 0s.
        """
        dx = 1.0
        # Create a 5x5 matrix of all 1s
        u_constant = np.ones((5, 5)) 
        
        # Call our interface to compute the finite difference
        result = discrete_laplacian_2d(u_constant, dx)
        
        # Create the theoretically correct answer: an all 0 matrix
        expected_result = np.zeros((5, 5))
        
        # Assert: Strictly compare the computed result with the expected answer, allowing for tiny floating-point errors.
        np.testing.assert_allclose(result, expected_result, atol=1e-10)

    def test_impulse_matrix(self):
        """
        Test Case 2: Center single-point impulse matrix.
        Physical meaning: Tests the diffusion rate of an isolated high-concentration point to its surroundings, verifying boundary conditions and the coefficients of the difference formula.
        """
        dx = 0.5
        # Create a 3x3 matrix of all 0s
        u_impulse = np.zeros((3, 3))
        # Place a single "drop of high-concentration material" right in the center
        u_impulse[1, 1] = 1.0 
        
        result = discrete_laplacian_2d(u_impulse, dx)
        
        # Theoretical calculation 1: The material at the center point (1,1) is flowing outwards.
        # Apply formula: (0 + 0 + 0 + 0 - 4*1) / (0.5)^2 = -4 / 0.25 = -16.0
        self.assertAlmostEqual(result[1, 1], -16.0)
        
        # Theoretical calculation 2: The point directly above the center (0,1) is receiving material from below (1,1).
        # Its neighbors: up (wraps to bottom edge 0), down (center point 1), left (0), right (0).
        # Apply formula: (0 + 1 + 0 + 0 - 4*0) / (0.5)^2 = 1 / 0.25 = 4.0
        self.assertAlmostEqual(result[0, 1], 4.0)

# Entry point to run this script directly in VS Code
if __name__ == '__main__':
    # Execute all unit test functions starting with test_
    unittest.main(verbosity=2)