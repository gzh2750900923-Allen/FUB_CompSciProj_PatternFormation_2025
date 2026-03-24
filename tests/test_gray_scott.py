"""Unit tests for GrayScottModel reaction terms."""
import numpy as np
from pattern_formation.models.gray_scott import GrayScottModel


def test_gray_scott_shapes():
    """f(u,v) and g(u,v) must return arrays with the same shape as input."""
    model = GrayScottModel(alpha=0.035, beta=0.065)
    u = np.ones((16, 16)) * 0.5
    v = np.ones((16, 16)) * 0.25
    assert model.f(u, v).shape == (16, 16), \
        f"f output shape wrong: {model.f(u,v).shape}"
    assert model.g(u, v).shape == (16, 16), \
        f"g output shape wrong: {model.g(u,v).shape}"
    print("  [PASS] f(u,v) and g(u,v) output shapes correct")


def test_gray_scott_steady_state():
    """At homogeneous steady state (u=1, v=0): f=0, g=0."""
    model = GrayScottModel(alpha=0.035, beta=0.065)
    u = np.ones((8, 8))   # u* = 1
    v = np.zeros((8, 8))  # v* = 0
    f_val = model.f(u, v)
    g_val = model.g(u, v)
    assert np.allclose(f_val, 0.0), \
        f"f should be 0 at steady state, max={np.max(np.abs(f_val)):.2e}"
    assert np.allclose(g_val, 0.0), \
        f"g should be 0 at steady state, max={np.max(np.abs(g_val)):.2e}"
    print("  [PASS] Steady state (u=1, v=0): f=0, g=0")


if __name__ == "__main__":
    print("=== Gray-Scott Model Tests ===")
    test_gray_scott_shapes()
    test_gray_scott_steady_state()
    print("All Gray-Scott tests PASSED ✓")
