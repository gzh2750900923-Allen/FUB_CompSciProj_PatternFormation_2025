import numpy as np
import unittest
#from explicit_solver import explicit_euler_step
#from implicit_CN_solver import crank_nicolson_step
from pattern_formation.explicit_solver import explicit_euler_step
from pattern_formation.implicit_CN_solver import crank_nicolson_step


class TestTimeSteppers(unittest.TestCase):
    def setUp(self):
        """Set up a basic grid and zero-reaction functions for testing."""
        self.N = 32
        self.dx = 1.0
        self.D_u = 0.5
        self.D_v = 0.5
        # f, g = 0 (Pure Diffusion)
        self.f_zero = lambda u, v: np.zeros_like(u)
        self.g_zero = lambda u, v: np.zeros_like(v)

    def test_pure_diffusion_decay(self):
        """
        [Confirmed] Test pure diffusion decay against analytical solution.
        On a periodic grid, a sine wave u(x,0) = sin(2*pi*x/L) decays as 
        u(x,t) = exp(-D * k^2 * t) * sin(2*pi*x/L).
        """
        dt = 0.01
        T = 1.0
        steps = int(T / dt)
        
        # Initial condition: Sine wave in x-direction
        x = np.linspace(0, self.N * self.dx, self.N, endpoint=False)
        u = np.zeros((self.N, self.N))
        for i in range(self.N):
            u[i, :] = np.sin(2 * np.pi * x[i] / (self.N * self.dx))
        v = np.zeros_like(u)
        
        # Run solvers for T seconds
        u_exp, v_exp = u.copy(), v.copy()
        u_imp, v_imp = u.copy(), v.copy()
        
        for _ in range(steps):
            u_exp, v_exp = explicit_euler_step(u_exp, v_exp, self.dx, dt, self.D_u, self.D_v, self.f_zero, self.g_zero)
            u_imp, v_imp = crank_nicolson_step(u_imp, v_imp, self.dx, dt, self.D_u, self.D_v, self.f_zero, self.g_zero)
            
        # Analytical solution
        k = 2 * np.pi / (self.N * self.dx)
        expected_decay = np.exp(-self.D_u * (k**2) * T)
        u_analytical = u * expected_decay
        
        # Check if numerical solution matches analytical within tolerance
        # Explicit: O(dt), Implicit: O(dt^2)
        np.testing.assert_allclose(u_exp, u_analytical, atol=1e-2, err_msg="Explicit Euler pure diffusion failure")
        np.testing.assert_allclose(u_imp, u_analytical, atol=1e-3, err_msg="Crank-Nicolson pure diffusion failure")

    def test_cn_unconditional_stability(self):
    	"""
    	[Confirmed] Test Crank-Nicolson stability with a very large dt.
    	Crank-Nicolson is L2-stable but can violate the maximum principle 
    	(oscillate) at very large dt. We verify the solution remains finite.
    	"""
    	large_dt = 10.0 # Way beyond the explicit CFL limit
    	u = np.random.rand(self.N, self.N)
    	v = np.random.rand(self.N, self.N)
    
    # Calculate initial L2 norm for reference
    	initial_l2_u = np.linalg.norm(u)
    
    	u_next, v_next = crank_nicolson_step(u, v, self.dx, large_dt, self.D_u, self.D_v, self.f_zero, self.g_zero)
    
    # 1. Check if the solution is still finite (Not NaN or Inf)
    # This proves the 'Unconditional Stability' of the implicit scheme.
    	self.assertTrue(np.all(np.isfinite(u_next)), "CN solution blew up (non-finite)")
    
    # 2. Check L2 stability (The energy should not explode)
    # In pure diffusion, the L2 norm of the solution should not increase significantly.
    	final_l2_u = np.linalg.norm(u_next)
    	self.assertLessEqual(final_l2_u, initial_l2_u + 1e-10, "CN failed L2 stability test")
if __name__ == '__main__':
    unittest.main(verbosity=2)
