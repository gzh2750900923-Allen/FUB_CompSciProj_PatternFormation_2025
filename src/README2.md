# Numerical Methods for 2D Reaction-Diffusion Systems

This document outlines the mathematical derivation and discretization of a 2-dimensional **Reaction-Diffusion Equation** using the **Finite Difference Method (FDM)** and various temporal integration schemes.

## 1. The Governing Equation (PDE)

A system of reaction-diffusion equations in a domain $\Omega = [0, L]^2 \subset \mathbb{R}^2$ is given by:

$$\begin{cases}
u_t = \delta_1 \Delta u + f(u, v) \\
v_t = \delta_2 \Delta v + g(u, v)
\end{cases}$$

Where:

* $u, v$: Concentrations of species.
* $\delta_1, \delta_2$: Diffusion coefficients.
* $\Delta = \partial_{xx} + \partial_{yy}$: The Laplacian operator.
* $f, g$: Non-linear reaction terms.

---

## 2. Spatial Discretization

We discretize the spatial domain into a grid $G$ with spacing $\Delta x$ and $\Delta y$. Assuming $\Delta x = \Delta y$, the 2nd-order central difference for the Laplacian at grid point $(i, j)$ is:

$$\Delta u \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2}$$

The discrete spatial operator (RHS) becomes:

$$\text{RHS}(u_{i,j}) = \frac{\delta_1}{\Delta x^2} \left( u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j} \right) + f(u_{i,j}, v_{i,j})$$

---

## 3. Temporal Discretization Schemes

Let $u^k$ denotes the solution at time $t_k = k \Delta t$.

### A. Explicit Euler (eE)

Evaluates the RHS at the current time step $t_k$. It is easy to implement but conditionally stable.

$$u_{i,j}^{k+1} = u_{i,j}^k + \Delta t \cdot \text{RHS}(u_{i,j}^k, v_{i,j}^k)$$

### B. Implicit Euler (iE)

Evaluates the RHS at the next time step $t_{k+1}$. This requires solving a system of equations but is unconditionally stable.

$$u_{i,j}^{k+1} = u_{i,j}^k + \Delta t \cdot \text{RHS}(u_{i,j}^{k+1}, v_{i,j}^{k+1})$$

### C. Crank-Nicolson (CN)

The "Swiss Army Knife" of integration. It averages the explicit and implicit steps, providing 2nd-order accuracy in time.

$$\frac{u^{k+1} - u^k}{\Delta t} = \frac{1}{2} \left( \text{RHS}|_{t_k} + \text{RHS}|_{t_{k+1}} \right)$$

---

## 4. Stability Analysis (CFL Condition)

For the **Explicit Euler** scheme to remain stable and avoid divergent oscillations, the time step $\Delta t$ must satisfy:

$$\Delta t \le \frac{\Delta x^2}{4 \cdot \max(\delta_1, \delta_2)}$$

---

## 5. Matrix Representation & Vectorization

To compute the solution efficiently (e.g., using NumPy or SciPy), we flatten the 2D grid $u_{i,j}$ into a 1D vector $\mathbf{u} \in \mathbb{R}^{n^2}$.

### Grid Mapping

The vector $\mathbf{u}$ is ordered column-wise (or row-wise):

$$\mathbf{u} = [u_{0,0}, \dots, u_{n,0}, u_{0,1}, \dots, u_{n,1}, \dots, u_{n,n}]^T$$

### Linear System Form

The discretized equation can be written in matrix form using **Shift Matrices** ($A, B$) and the **Identity Matrix** ($E$):

$$\mathbf{u}^{k+1} = \left[ \frac{\Delta t \delta_1}{\Delta x^2} (A + B - 4E) + E \right] \mathbf{u}^k + \Delta t \mathbf{F}^k$$

* $A$: Interaction matrix for the $x$-direction ($i \pm 1$).
* $B$: Interaction matrix for the $y$-direction ($j \pm 1$).
* $E$: Identity matrix.
* $\mathbf{F}^k$: Vectorized reaction terms $f(u^k, v^k)$.
