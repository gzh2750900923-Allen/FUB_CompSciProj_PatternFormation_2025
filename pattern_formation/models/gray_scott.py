"""
Gray-Scott reaction model
-------------------------
f(u, v) = -u*v² + α*(1 - u)
g(u, v) =  u*v² - (α + β)*v

Standard parameter presets (Pearson 1993):
    spots   : alpha=0.035, beta=0.065
    stripes : alpha=0.060, beta=0.062
    maze    : alpha=0.029, beta=0.057
"""

import numpy as np
from pattern_formation.core.interface import BaseModel
from pattern_formation.core.grid import CartesianGrid


# Pearson (1993) named parameter sets
PRESETS = {
    "spots":   dict(alpha=0.035, beta=0.065),
    "stripes": dict(alpha=0.060, beta=0.062),
    "maze":    dict(alpha=0.029, beta=0.057),
    "worms":   dict(alpha=0.039, beta=0.058),
}


class GrayScottModel(BaseModel):
    """
    Parameters
    ----------
    alpha : float  — feed rate  (F in original notation)
    beta  : float  — kill rate  (k in original notation)
    """

    def __init__(self, alpha: float = 0.035, beta: float = 0.065):
        self.alpha = alpha
        self.beta  = beta

    @classmethod
    def from_preset(cls, name: str) -> "GrayScottModel":
        """Construct from a named Pearson preset, e.g. 'spots'."""
        if name not in PRESETS:
            raise KeyError(f"Unknown preset '{name}'. "
                           f"Available: {list(PRESETS)}")
        return cls(**PRESETS[name])

    # ── reaction terms ────────────────────────────────────────────────────────
    def f(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return -u * v**2 + self.alpha * (1.0 - u)

    def g(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u * v**2 - (self.alpha + self.beta) * v

    # ── initial condition helper ───────────────────────────────────────────────
    def initial_conditions(self, grid: CartesianGrid,
                           seed: int = 42
                           ) -> tuple[np.ndarray, np.ndarray]:
        """
        Classic Gray-Scott IC: u≈1, v≈0 everywhere,
        with a small central square perturbation seeding v.

        Returns (u0, v0) each of shape (N, N).
        """
        N   = grid.N
        rng = np.random.default_rng(seed)

        u0 = np.ones((N, N))
        v0 = np.zeros((N, N))

        # Central square perturbation
        r = N // 8
        cx = cy = N // 2
        u0[cx - r:cx + r, cy - r:cy + r] = 0.50
        v0[cx - r:cx + r, cy - r:cy + r] = 0.25

        # Tiny random noise to break symmetry
        u0 += 0.01 * rng.standard_normal((N, N))
        v0 += 0.01 * rng.standard_normal((N, N))

        # Clip to physical range
        u0 = np.clip(u0, 0.0, 1.0)
        v0 = np.clip(v0, 0.0, 1.0)
        return u0, v0

    def __repr__(self) -> str:
        return f"GrayScottModel(alpha={self.alpha}, beta={self.beta})"
