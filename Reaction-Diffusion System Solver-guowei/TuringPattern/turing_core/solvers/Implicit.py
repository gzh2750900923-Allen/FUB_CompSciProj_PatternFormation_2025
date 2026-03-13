import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import time
from ..models import Grey_Scott

class CrankNicolsonScheme:
    """combination 2: Crank-Nicolson time stepping + second-order finite difference"""
    def __init__(self, N, dx, dt, Du, Dv, steps,F,K,u,v):
        self.N = N
        self.dt = dt
        self.steps = steps
        self.Du = Du
        self.Dv = Dv    
        self.F = F
        self.K = K
        self.u = u
        self.v = v
        
        # construct the sparse Laplacian matrix L for 2D grid with periodic boundary conditions
        I_1D = sparse.eye(N, format='csc')
        e = np.ones(N)
        D_1D = sparse.spdiags([e, e, -2*e, e, e], [-N+1, -1, 0, 1, N-1], N, N, format='csc')
        L = (sparse.kron(I_1D, D_1D) + sparse.kron(D_1D, I_1D)) / (dx ** 2)
        
        # construct the CN left-hand side matrix A and right-hand side matrix B
        I_2D = sparse.eye(N**2, format='csc')
        A_u = I_2D - (dt / 2.0) * Du * L
        B_u = I_2D + (dt / 2.0) * Du * L
        A_v = I_2D - (dt / 2.0) * Dv * L
        B_v = I_2D + (dt / 2.0) * Dv * L
        
        # pre-factorize the sparse matrices A_u and A_v for efficient solving during time stepping
        self.B_u = B_u
        self.B_v = B_v
        self.solve_u = splinalg.factorized(A_u)
        self.solve_v = splinalg.factorized(A_v)

    def step(self, u, v, f, g):
        """CN time stepping"""
        dt = self.dt
        
        u_vec = u.flatten()
        v_vec = v.flatten()
        f_vec = f.flatten()
        g_vec = g.flatten()
        
        # The RHS includes the diffusion term from the previous time step and the nonlinear reaction term from the current time step
        rhs_u = self.B_u.dot(u_vec) + dt * f_vec
        rhs_v = self.B_v.dot(v_vec) + dt * g_vec
        
        u_next_vec = self.solve_u(rhs_u)
        v_next_vec = self.solve_v(rhs_v)
        
        return u_next_vec.reshape((self.N, self.N)), v_next_vec.reshape((self.N, self.N))

    def run(self):
        start_time = time.perf_counter()#record the start time for performance measurement
        
        for i in range(self.steps):
            f, g = Grey_Scott(self.F, self.K, self.u, self.v)#compute the reaction terms based on the current state of u and v
            self.u, self.v = self.step(self.u, self.v, f, g)#update u and v using the explicit time stepping scheme
            
            if i % 1000 == 0:
                print(f"[{CrankNicolsonScheme}] Progress: step {i}/{self.steps}")#periodically print progress every 1000 steps to monitor the simulation
                
        end_time = time.perf_counter()#record the end time and calculate the total generation time for the simulation
        gen_time = end_time - start_time
        return self.v, gen_time