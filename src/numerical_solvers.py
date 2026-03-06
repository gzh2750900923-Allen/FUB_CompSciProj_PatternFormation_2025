# numerical_solvers.py
import numpy as np
from scipy.optimize import fsolve

def explicit_euler(f, y0, t):
    """Explicit Euler for First-Order ODEs."""
    y = np.zeros(len(t))
    y[0] = y0
    h = t[1] - t[0]
    for n in range(len(t) - 1):
        y[n+1] = y[n] + h * f(t[n], y[n])
    return y

def implicit_euler(f, y0, t):
    """Implicit Euler for Stiff ODEs."""
    y = np.zeros(len(t))
    y[0] = y0
    h = t[1] - t[0]
    for n in range(len(t) - 1):
        func = lambda next_y: next_y - y[n] - h * f(t[n+1], next_y)
        y[n+1] = fsolve(func, y[n])
    return y

def solve_bvp_2nd_fd(f_func, a, b, ya, yb, N):
    """2nd-order Finite Difference Method for BVPs."""
    x = np.linspace(a, b, N+1)
    h = x[1] - x[0]
    A = np.zeros((N-1, N-1))
    rhs = np.zeros(N-1)
    
    for i in range(N-1):
        A[i, i] = -2
        if i > 0: A[i, i-1] = 1
        if i < N-2: A[i, i+1] = 1
        rhs[i] = (h**2) * f_func(x[i+1])
        
    rhs[0] -= ya
    rhs[-1] -= yb
    y_inner = np.linalg.solve(A, rhs)
    return x, np.concatenate(([ya], y_inner, [yb]))
