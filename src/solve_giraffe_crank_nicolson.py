import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu

def solve_giraffe_crank_nicolson():
    # --- 1. Simulation Setup ---
    N = 100
    N2 = N * N
    dx = 1.0
    dt = 1.0  # Crank-Nicolson allows for a larger dt than Explicit
    steps = 2000

    # --- 2. Parameters for Giraffe Pattern ---
    Du, Dv = 0.208, 0.105
    f, k = 0.030, 0.062  # f=alpha, k=alpha+beta

    # --- 3. Build Sparse Laplacian Matrix (L) ---
    # Creating 1D periodic Laplacian
    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    L1D = sparse.diags([main_diag, off_diag, off_diag, [1], [1]], [0, -1, 1, N-1, -(N-1)])

    # 2D Laplacian using Kronecker sum
    I_1d = sparse.eye(N)
    L = (sparse.kron(L1D, I_1d) + sparse.kron(I_1d, L1D)) / (dx**2)
    I_sp = sparse.eye(N2)

    # --- 4. Pre-factorize Matrices for Implicit Step ---
    # (I - alpha*L)u_next = (I + alpha*L)u_curr + dt*f(u,v)
    alpha_u = (Du * dt) / 2
    alpha_v = (Dv * dt) / 2

    A_u = (I_sp - alpha_u * L).tocsc()
    B_u = (I_sp + alpha_u * L).tocsc()
    A_v = (I_sp - alpha_v * L).tocsc()
    B_v = (I_sp + alpha_v * L).tocsc()

    # Pre-factorization for speed
    solve_u = splu(A_u).solve
    solve_v = splu(A_v).solve

    # --- 5. Initialize Fields ---
    u = np.ones((N, N))
    v = np.zeros((N, N))
    # Seed noise
    v[45:55, 45:55] = 0.25 + np.random.standard_normal((10, 10)) * 0.01

    u_vec = u.flatten()
    v_vec = v.flatten()

    # --- 6. Main Simulation Loop ---
    for i in range(steps):
        # Reaction terms (computed explicitly)
        uv2 = u_vec * (v_vec**2)
        reac_u = -uv2 + f * (1 - u_vec)
        reac_v = uv2 - k * v_vec

        # Solve for next time step
        rhs_u = B_u.dot(u_vec) + dt * reac_u
        u_vec = solve_u(rhs_u)

        rhs_v = B_v.dot(v_vec) + dt * reac_v
        v_vec = solve_v(rhs_v)

        if i % 500 == 0:
            print(f"Step {i} completed...")

    return v_vec.reshape((N, N))

# --- 7. Execution and Visualization ---
result_v = solve_giraffe_crank_nicolson()
plt.figure(figsize=(6,6))
plt.imshow(result_v, cmap='YlOrBr', interpolation='lanczos')
plt.title("Giraffe Pattern: Crank-Nicolson Method")
plt.axis('off')
plt.show()
