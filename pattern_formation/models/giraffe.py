"""
Schnakenberg activator-substrate model for giraffe-like coat patterns.

Reference: Murray, J. D. (2003). Mathematical Biology II. Springer.
           Chapter 2–3: Reaction-diffusion models for animal coat patterns.

Model equations:
    f(u, v) = γ (a - u + u² v)
    g(u, v) = γ (b - u² v)

where:
    u = activator concentration
    v = substrate/inhibitor concentration
    a, b = source/sink parameters
    γ   = scale factor (controls pattern wavelength relative to domain)

Turing instability conditions:
    · δ = δ₂/δ₁ >> 1  (inhibitor diffuses much faster than activator)
    · Homogeneous steady state: u* = a+b,  v* = b/(a+b)²

Pattern type is governed by γ:
    · γ small  → few large patches   (giraffe-like polygons)
    · γ large  → many fine spots/stripes

Recommended parameters for giraffe polygonal patches:
    a=0.1, b=0.9, γ=800, δ₁=0.005, δ₂=1.0, dt=0.5
"""

import numpy as np
from pattern_formation.core.interface import BaseModel
from pattern_formation.core.grid import CartesianGrid


class GiraffeModel(BaseModel):
    """
    Schnakenberg model tuned for giraffe-like polygonal coat patterns.

    Parameters
    ----------
    a     : float  — activator source rate (default 0.1)
    b     : float  — substrate supply rate (default 0.9)
    gamma : float  — spatial scale factor  (default 800)
    """

    def __init__(self, a: float = 0.1, b: float = 0.9, gamma: float = 800.0):
        self.a     = a
        self.b     = b
        self.gamma = gamma

    # ── steady state ──────────────────────────────────────────────────────────
    @property
    def steady_state(self) -> tuple[float, float]:
        """Homogeneous steady state (u*, v*)."""
        u_ss = self.a + self.b
        v_ss = self.b / (self.a + self.b) ** 2
        return u_ss, v_ss

    # ── reaction terms ────────────────────────────────────────────────────────
    def f(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.gamma * (self.a - u + u**2 * v)

    def g(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.gamma * (self.b - u**2 * v)

    # ── initial conditions ────────────────────────────────────────────────────
    def initial_conditions(self, grid: CartesianGrid,
                           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Perturb the homogeneous steady state with small random noise.
        This seeds Turing instability across all wavenumbers.
        """
        rng   = np.random.default_rng(seed)
        u_ss, v_ss = self.steady_state
        noise = 0.05
        u0 = u_ss + noise * rng.standard_normal((grid.N, grid.N))
        v0 = v_ss + noise * rng.standard_normal((grid.N, grid.N))
        # Keep concentrations positive
        u0 = np.abs(u0)
        v0 = np.abs(v0)
        return u0, v0

    def __repr__(self) -> str:
        return f"GiraffeModel(a={self.a}, b={self.b}, gamma={self.gamma})"
