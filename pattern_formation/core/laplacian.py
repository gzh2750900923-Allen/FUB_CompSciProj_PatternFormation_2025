"""
Second-order finite-difference Laplacian on a periodic Cartesian grid.

The discrete Laplacian at interior point (i,j) is:

    Δu[i,j] ≈ (u[i+1,j] - 2u[i,j] + u[i-1,j]) / dx²
             + (u[i,j+1] - 2u[i,j] + u[i,j-1]) / dx²

Periodic boundary conditions are enforced with np.roll (O(1) indexing).
"""

import numpy as np
from .grid import CartesianGrid


def laplacian(u: np.ndarray, grid: CartesianGrid) -> np.ndarray:
    """
    Compute the 2-D Laplacian of *u* using second-order central differences
    with periodic boundary conditions.

    Parameters
    ----------
    u    : ndarray, shape (N, N)
    grid : CartesianGrid

    Returns
    -------
    ndarray, shape (N, N)
    """
    dx2 = grid.dx ** 2
    # Shifts along axis=0 (x-direction) and axis=1 (y-direction)
    lap = (
        np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0)   # u_{i±1, j}
        + np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1)  # u_{i, j±1}
        - 4.0 * u
    ) / dx2
    return lap
