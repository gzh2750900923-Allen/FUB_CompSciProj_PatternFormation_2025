"""
Visualization utilities for reaction-diffusion concentration fields.

Public functions
----------------
plot_state          — side-by-side u / v colour maps (single solver)
plot_comparison     — 2×2 grid comparing two solvers (Explicit vs CN)
plot_convergence    — log-log time-step convergence plot
plot_benchmark      — bar chart of wall-clock times
plot_evolution      — row of snapshots showing pattern development over time
animate_history     — matplotlib animation from a list of (u, v) snapshots
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_or_show(fig, save_path: str | None) -> None:
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()


# ── 1. plot_state ─────────────────────────────────────────────────────────────

def plot_state(u: np.ndarray, v: np.ndarray,
               title: str = "Reaction-Diffusion State",
               cmap_u: str = "viridis",
               cmap_v: str = "plasma",
               save_path: str | None = None) -> None:
    """
    Side-by-side colour maps of activator u and inhibitor v.

    Parameters
    ----------
    u, v       : 2-D concentration arrays, shape (N, N)
    title      : figure title
    cmap_u/v   : matplotlib colormap names
    save_path  : file path to save (PNG/PDF); None → plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=13)

    kw = dict(origin="lower", aspect="equal", interpolation="nearest")
    im0 = axes[0].imshow(u.T, cmap=cmap_u, **kw)
    axes[0].set_title("u  (activator)"); axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(v.T, cmap=cmap_v, **kw)
    axes[1].set_title("v  (inhibitor)"); axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    _save_or_show(fig, save_path)


# ── 2. plot_comparison ────────────────────────────────────────────────────────

def plot_comparison(u1: np.ndarray, v1: np.ndarray,
                    u2: np.ndarray, v2: np.ndarray,
                    label1: str = "Solver A",
                    label2: str = "Solver B",
                    title: str = "Solver Comparison",
                    cmap_u: str = "RdBu_r",
                    cmap_v: str = "plasma",
                    save_path: str | None = None) -> None:
    """
    2×2 grid comparing two solvers side by side.

    Layout:
        [label1 — u]  [label1 — v]
        [label2 — u]  [label2 — v]

    Parameters
    ----------
    u1, v1     : results from solver 1 (e.g. Explicit)
    u2, v2     : results from solver 2 (e.g. Crank-Nicolson)
    label1/2   : row labels shown in subplot titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(title, fontsize=12)

    kw = dict(origin="lower", aspect="equal", interpolation="nearest")
    axes[0, 0].imshow(u1.T, cmap=cmap_u, **kw)
    axes[0, 0].set_title(f"{label1} — u"); axes[0, 0].axis("off")

    axes[0, 1].imshow(v1.T, cmap=cmap_v, **kw)
    axes[0, 1].set_title(f"{label1} — v"); axes[0, 1].axis("off")

    axes[1, 0].imshow(u2.T, cmap=cmap_u, **kw)
    axes[1, 0].set_title(f"{label2} — u"); axes[1, 0].axis("off")

    axes[1, 1].imshow(v2.T, cmap=cmap_v, **kw)
    axes[1, 1].set_title(f"{label2} — v"); axes[1, 1].axis("off")

    _save_or_show(fig, save_path)


# ── 3. plot_convergence ───────────────────────────────────────────────────────

def plot_convergence(dts: list,
                     errors_a: list,
                     errors_b: list,
                     label_a: str = "Explicit (Forward Euler)",
                     label_b: str = "Crank-Nicolson",
                     title: str = "Time-step convergence",
                     save_path: str | None = None) -> None:
    """
    Log-log convergence plot of two solvers with O(dt¹) and O(dt²) reference lines.

    Parameters
    ----------
    dts        : list of time step sizes (x-axis)
    errors_a/b : list of max-norm errors for each solver
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.loglog(dts, errors_a, "o-", label=label_a)
    ax.loglog(dts, errors_b, "s-", label=label_b)

    # Reference slopes anchored to first data point
    ax.loglog(dts, [errors_a[0] * (d / dts[0]) ** 1 for d in dts],
              "k--", alpha=0.45, label="O(dt¹)")
    ax.loglog(dts, [errors_b[0] * (d / dts[0]) ** 2 for d in dts],
              "k:",  alpha=0.45, label="O(dt²)")

    ax.set_xlabel("Δt")
    ax.set_ylabel("max |error|")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    _save_or_show(fig, save_path)


# ── 4. plot_benchmark ─────────────────────────────────────────────────────────

def plot_benchmark(labels: list,
                   times_a: list,
                   times_b: list,
                   label_a: str = "Explicit",
                   label_b: str = "Crank-Nicolson",
                   title: str = "Performance Benchmark",
                   ylabel: str = "Wall time (s)",
                   save_path: str | None = None) -> None:
    """
    Grouped bar chart comparing wall-clock times for two solvers.

    Parameters
    ----------
    labels     : x-axis group labels, e.g. ["N=64", "N=128"]
    times_a/b  : wall times for each solver at each label
    """
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, times_a, w, label=label_a)
    ax.bar(x + w / 2, times_b, w, label=label_b)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    _save_or_show(fig, save_path)


# ── 5. plot_evolution ─────────────────────────────────────────────────────────

def plot_evolution(snapshots: list,
                   times: list,
                   field: str = "u",
                   title: str = "Pattern Evolution",
                   cmap: str = "viridis",
                   save_path: str | None = None) -> None:
    """
    Row of snapshots showing how the pattern develops over time.

    Parameters
    ----------
    snapshots  : list of (u, v) tuples at successive time points
    times      : list of physical time values, one per snapshot
    field      : which field to display — "u" or "v"
    """
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    fig.suptitle(title, fontsize=12)

    idx = 0 if field == "u" else 1
    kw  = dict(origin="lower", aspect="equal", interpolation="lanczos")

    for ax, snap, t in zip(axes, snapshots, times):
        data = snap[idx] if isinstance(snap, (tuple, list)) else snap
        ax.imshow(data.T, cmap=cmap, **kw)
        ax.set_title(f"T = {t:.2f}", fontsize=9)
        ax.axis("off")

    _save_or_show(fig, save_path)


# ── 6. animate_history ────────────────────────────────────────────────────────

def animate_history(history: list,
                    interval_ms: int = 100,
                    field: str = "u",
                    cmap: str = "viridis",
                    save_path: str | None = None):
    """
    Matplotlib animation from a list of (u, v) snapshots.

    Parameters
    ----------
    history    : list of (u, v) tuples
    interval_ms: milliseconds between frames
    field      : "u" or "v"
    save_path  : save as GIF/MP4 if given; None → plt.show()

    Returns
    -------
    FuncAnimation object
    """
    import matplotlib.animation as animation

    idx  = 0 if field == "u" else 1
    data = history[0][idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    im = ax.imshow(data.T, origin="lower", cmap=cmap, animated=True)

    def update(frame):
        d = history[frame][idx]
        im.set_data(d.T)
        im.set_clim(d.min(), d.max())
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(history),
        interval=interval_ms, blit=True)

    if save_path:
        ani.save(save_path)
        plt.close(fig)
        print(f"  Animation saved → {save_path}")
    else:
        plt.show()

    return ani
