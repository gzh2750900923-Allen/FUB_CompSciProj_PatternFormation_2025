import numpy as np
import time
from ..models import Grey_Scott

class ExplicitScheme:
    """combination 1: Explicit time stepping + second-order finite difference"""
    def __init__(self, dx, dt, Du, Dv, steps,F,K,u,v):
        #dx is the spatial step size, dt is the time step size, Du and Dv are diffusion coefficients δ
        self.dx2 = dx ** 2
        self.dt = dt
        self.Du = Du
        self.Dv = Dv
        self.steps = steps
        self.F = F
        self.K = K
        self.u = u
        self.v = v

    def laplacian_2d(self, field):
        """Second order finite difference approximation for the Laplacian"""
        dx2 = self.dx2
        #lap=(u_i+1,j + u_i-1,j+u_i,j+1 + u_i,j-1 - 4u_i,j) / dx^2；
        #np.roll can also used to implement the periodic boundary conditions
        lap = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
               np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
               4.0 * field) / dx2
        return lap

    def step(self, u, v, f, g):
        """Explicit time stepping"""
       
        dt = self.dt
        Du = self.Du
        Dv = self.Dv
        
        lap_u = self.laplacian_2d(u)
        lap_v = self.laplacian_2d(v)
        
        # 完美对应白板推导形式：u^{k+1} = u^k + dt * (D * Laplacian(u) + f(u))
        # This perfectly matches our derivation: u^{k+1} = u^k + dt * (D * Laplacian(u) + f(u))
        u_next = u + dt * (Du * lap_u + f)
        v_next = v + dt * (Dv * lap_v + g)
        
        return u_next, v_next
    
    def run(self):
        start_time = time.perf_counter()#record the start time for performance measurement
        print(type(self.u), type(self.v))
        for i in range(self.steps):
            f, g = Grey_Scott(self.F, self.K, self.u, self.v)#compute the reaction terms based on the current state of u and v
            self.u, self.v = self.step(self.u, self.v, f, g)#update u and v using the explicit time stepping scheme
            
            if i % 1000 == 0:
                print(f"[{ExplicitScheme}] Progress: step {i}/{self.steps}")#periodically print progress every 1000 steps to monitor the simulation
                
        end_time = time.perf_counter()#record the end time and calculate the total generation time for the simulation
        gen_time = end_time - start_time
        return self.v, gen_time