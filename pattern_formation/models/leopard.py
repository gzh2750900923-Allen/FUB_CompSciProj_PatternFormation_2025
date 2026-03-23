"""
Two-component Gierer-Meinhardt model for leopard/jaguar spot patterns.

Reference:
    Liu, R. T., Liaw, S. S., & Maini, P. K. (2006).
    "Two-stage Turing model for generating pigment patterns on the
    leopard and the jaguar." Physical Review E, 74, 011914.

    Murray, J. D. (2003). Mathematical Biology II. Springer.

Model (activator-inhibitor with saturation):
    f(u, v) = γ [ u²/(v (1 + κ u²))  -  μ u ]
    g(u, v) = γ [ u² - ν v ]

where:
    u = activator (short-range autocatalysis)
    v = inhibitor (long-range inhibition)
    κ = saturation parameter (prevents blow-up)
    μ = activator decay rate
    ν = inhibitor decay rate
    γ = scale factor

Homogeneous steady state:
    u* = (ν / μ·(1 + κ(ν/μ)²))^(1/2) ... (solved numerically)
    Approximation for small κ:  u* ≈ sqrt(ν/μ),  v* = u*²/ν

Turing instability requires δ = δ₂/δ₁ >> 1.

Recommended parameters for leopard spots:
    gamma=300, mu=0.5, nu=1.0, kappa=0.1
    δ₁=0.005, δ₂=1.0, dt=0.2
"""

import numpy as np
from pattern_formation.core.interface import BaseModel
from pattern_formation.core.grid import CartesianGrid


class LeopardModel(BaseModel):
    """
    Gierer-Meinhardt activator-inhibitor model for leopard spot patterns.

    Parameters
    ----------
    gamma : float  — spatial scale factor (default 300)
    mu    : float  — activator decay      (default 0.5)
    nu    : float  — inhibitor decay      (default 1.0)
    kappa : float  — saturation constant  (default 0.1)
    """

    def __init__(self, gamma: float = 300.0, mu: float = 0.5,
                 nu: float = 1.0, kappa: float = 0.1):
        self.gamma = gamma
        self.mu    = mu
        self.nu    = nu
        self.kappa = kappa

    # ── reaction terms ────────────────────────────────────────────────────────
    def f(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Activator: autocatalysis with saturation minus decay."""
        denom = np.maximum(v * (1.0 + self.kappa * u**2), 1e-12)   # avoid /0
        return self.gamma * (u**2 / denom - self.mu * u)

    def g(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inhibitor: produced by activator, decays linearly."""
        return self.gamma * (u**2 - self.nu * v)

    # ── steady state (approximate) ────────────────────────────────────────────
    @property
    def steady_state(self) -> tuple[float, float]:
        """Approximate homogeneous steady state for small κ."""
        u_ss = np.sqrt(self.nu / self.mu)
        v_ss = u_ss**2 / self.nu
        return float(u_ss), float(v_ss)

    # ── initial conditions ────────────────────────────────────────────────────
    def initial_conditions(self, grid: CartesianGrid,
                           seed: int = 0,
                           noise_scale: float = 0.05
                           ) -> tuple[np.ndarray, np.ndarray]:
        """
        Homogeneous steady state + spatially uniform random perturbations
        to seed the Turing instability.
        """
        rng   = np.random.default_rng(seed)
        u_ss, v_ss = self.steady_state
        u0 = u_ss + noise_scale * rng.standard_normal((grid.N, grid.N))
        v0 = v_ss + noise_scale * rng.standard_normal((grid.N, grid.N))
        u0 = np.abs(u0)
        v0 = np.abs(v0)
        return u0, v0

    def __repr__(self) -> str:
        return (f"LeopardModel(gamma={self.gamma}, mu={self.mu}, "
                f"nu={self.nu}, kappa={self.kappa})")
