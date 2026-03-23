"""
Abstract base class (interface) for reaction-diffusion solvers.

All concrete solvers must subclass BaseSolver and implement `step` and `run`.
"""

from abc import ABC, abstractmethod
import numpy as np
from .grid import CartesianGrid


class BaseModel(ABC):
    """Interface for reaction kinetics f(u,v) and g(u,v)."""

    @abstractmethod
    def f(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Reaction term for species u."""

    @abstractmethod
    def g(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Reaction term for species v."""


class BaseSolver(ABC):
    """
    Interface for time-integrators of the reaction-diffusion PDE system.

    Parameters
    ----------
    model  : BaseModel   — reaction kinetics
    grid   : CartesianGrid
    delta1 : float       — diffusion coefficient for u
    delta2 : float       — diffusion coefficient for v
    dt     : float       — time step size
    """

    def __init__(self, model: BaseModel, grid: CartesianGrid,
                 delta1: float, delta2: float, dt: float):
        self.model = model
        self.grid = grid
        self.delta1 = delta1
        self.delta2 = delta2
        self.dt = dt

    @abstractmethod
    def step(self, u: np.ndarray, v: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the solution by one time step. Returns (u_new, v_new)."""

    def run(self, u0: np.ndarray, v0: np.ndarray,
            n_steps: int,
            callback=None,
            callback_every: int = 100
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate the system for *n_steps* time steps.

        Parameters
        ----------
        u0, v0         : initial conditions, shape (N, N)
        n_steps        : number of time steps
        callback       : optional callable(step, u, v) for logging/saving
        callback_every : call callback every this many steps

        Returns
        -------
        (u, v) at final time
        """
        u, v = u0.copy(), v0.copy()
        for k in range(n_steps):
            u, v = self.step(u, v)
            if callback is not None and k % callback_every == 0:
                callback(k, u, v)
        return u, v
