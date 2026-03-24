"""Unit tests for CartesianGrid."""

import numpy as np
from pattern_formation.core.grid import CartesianGrid


def test_grid_spacing():
    grid = CartesianGrid(N=10, L=1.0)
    assert np.isclose(grid.dx, 0.1), f"Expected dx=0.1, got {grid.dx}"
    print("  [PASS] grid spacing correct")


def test_grid_shape():
    grid = CartesianGrid(N=32)
    assert grid.X.shape == (32, 32), f"X shape wrong: {grid.X.shape}"
    assert grid.Y.shape == (32, 32), f"Y shape wrong: {grid.Y.shape}"
    print("  [PASS] grid coordinate matrix shapes correct")


def test_zeros_shape():
    grid = CartesianGrid(N=16)
    z = grid.zeros()
    assert z.shape == (16, 16), f"zeros() shape wrong: {z.shape}"
    assert np.all(z == 0), "zeros() should return all zeros"
    print("  [PASS] zeros() shape and values correct")


def test_invalid_N():
    try:
        CartesianGrid(N=1)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [PASS] N=1 correctly raises ValueError")


if __name__ == "__main__":
    print("=== Grid Tests ===")
    test_grid_spacing()
    test_grid_shape()
    test_zeros_shape()
    test_invalid_N()
    print("All grid tests PASSED ✓")
