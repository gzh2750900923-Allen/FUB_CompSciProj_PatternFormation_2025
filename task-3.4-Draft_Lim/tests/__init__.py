"""
Unit test suite for the Turing Pattern Formation project.
This package contains verification scripts for:
- Spatial operators (Discrete Laplacian)
- Temporal solvers (Explicit Euler, Crank-Nicolson)
- Numerical convergence (Order validation)
- Parallel performance (Dask benchmarks)
"""

import os
import sys

# Ensure the parent directory is in the path so that 
# tests can import the core logic (pattern_formation package).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

__version__ = "0.1.0"
