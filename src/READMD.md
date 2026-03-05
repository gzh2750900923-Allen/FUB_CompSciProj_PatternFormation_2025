# Numerical Solvers for Differential Equations

This repository provides robust implementations of fundamental numerical methods for Computational Science.

## File Descriptions
- `numerical_solvers.py`: Contains implementations of Explicit/Implicit Euler and Finite Difference schemes.
- `main.py`: Script to demonstrate accuracy and stability comparison.

## Technical Details

### 1. Explicit Euler Method
The most basic explicit method for initial value problems (IVPs).
$$y_{n+1} = y_n + h f(t_n, y_n)$$
* **Note**: Numerically unstable for large step sizes ($h$) in stiff systems.

### 2. Implicit Euler Method
A numerically stable alternative for stiff differential equations.
$$y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})$$
* **Note**: Employs `scipy.optimize.fsolve` for solving the implicit equation at each step.

### 3. 2nd-Order Finite Difference (FD)
Used for Boundary Value Problems (BVPs).
$$y''(x) \approx \frac{y_{i-1} - 2y_i + y_{i+1}}{h^2}$$
* **Note**: Transforms the BVP into a linear system $A\mathbf{y} = \mathbf{b}$.
