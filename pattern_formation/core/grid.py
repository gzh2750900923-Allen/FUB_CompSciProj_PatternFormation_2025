"""Cartesian grid definition for the reaction-diffusion domain Ω = [0,1]²."""

import numpy as np


class CartesianGrid:
    """
    Uniform Cartesian grid on Ω = [0, L]² with N×N interior points
    and periodic boundary conditions.

    Parameters
    ----------
    N : int
        Number of grid points along each axis.
    L : float, optional
        Domain side length (default 1.0).
    """

    def __init__(self, N: int, L: float = 1.0):
        if N < 2:
            raise ValueError("N must be at least 2.")
        self.N = N
        self.L = L
        self.dx = L / N          # grid spacing
        # Cell-centre coordinates
        self.x = np.linspace(0, L - self.dx, N)
        self.y = np.linspace(0, L - self.dx, N)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

    def zeros(self) -> np.ndarray:
        """Return an N×N array of zeros (useful for initial conditions)."""
        return np.zeros((self.N, self.N))

    def random_perturbation(self, center: float = 0.0,
                            scale: float = 0.01,
                            seed: int | None = None) -> np.ndarray:
        """Small random field around *center* for initial conditions."""
        rng = np.random.default_rng(seed)
        return center + scale * rng.standard_normal((self.N, self.N))

    def __repr__(self) -> str:
        return f"CartesianGrid(N={self.N}, L={self.L}, dx={self.dx:.4f})"
