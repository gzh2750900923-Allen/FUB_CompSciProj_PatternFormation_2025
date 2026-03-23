"""Unit tests for the finite-difference Laplacian."""
import sys
import numpy as np
from pattern_formation.core.grid import CartesianGrid
from pattern_formation.core.laplacian import laplacian


def test_laplacian_of_constant():
    """Laplacian of a constant field must be zero."""
    grid = CartesianGrid(N=32)
    u = np.ones((32, 32)) * 5.0
    result = laplacian(u, grid)
    assert np.allclose(result, 0.0), \
        f"Laplacian of constant should be zero, max={np.max(np.abs(result)):.2e}"
    print("  [PASS] Laplacian(constant field) = 0")


def test_laplacian_of_linear():
    """Laplacian of a linear function f=x+y should be 0 (periodic approx)."""
    grid = CartesianGrid(N=64)
    u = grid.X + grid.Y
    result = laplacian(u, grid)
    interior = result[1:-1, 1:-1]
    assert np.allclose(interior, 0.0, atol=1e-10), \
        f"Laplacian of linear should be ~0, max={np.max(np.abs(interior)):.2e}"
    print("  [PASS] Laplacian(linear function) = 0 in interior")


def test_laplacian_second_order_convergence():
    """Check that error decreases as O(dx²): ratio ≈ 4 when N doubles."""
    errors = []
    Ns = [32, 64, 128]
    for N in Ns:
        grid = CartesianGrid(N=N)
        # sin(2πx)·sin(2πy) has exact Laplacian = -8π²·sin(2πx)·sin(2πy)
        u = np.sin(2 * np.pi * grid.X) * np.sin(2 * np.pi * grid.Y)
        lap_exact = -8 * np.pi**2 * u
        lap_num   = laplacian(u, grid)
        errors.append(np.max(np.abs(lap_num - lap_exact)))

    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]
    print(f"  errors: N=32 → {errors[0]:.3e},  "
          f"N=64 → {errors[1]:.3e},  N=128 → {errors[2]:.3e}")
    print(f"  convergence ratios: {ratio1:.2f}, {ratio2:.2f}  (expected ~4.0)")
    assert ratio1 > 3.5, f"Expected ratio ~4, got {ratio1:.2f}"
    assert ratio2 > 3.5, f"Expected ratio ~4, got {ratio2:.2f}"
    print("  [PASS] Second-order O(dx²) convergence confirmed")


if __name__ == "__main__":
    print("=== Laplacian Tests ===")
    test_laplacian_of_constant()
    test_laplacian_of_linear()
    test_laplacian_second_order_convergence()
    print("All Laplacian tests PASSED ✓")
