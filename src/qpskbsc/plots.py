"""Publication-style figures.

Saved as PDF (vector) so that they can be dropped straight into the manuscript.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "figure.figsize": (6.4, 4.8), "lines.linewidth": 1.3,
    "lines.markersize": 4, "savefig.bbox": "tight", "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3,
})

_LABEL = {"cyclic": "cyclic nulling (B1)", "map": "MAP nulling (B2)",
          "greedy": "greedy continuous (B3)", "distill": "distilled MPC"}
_MARK = {"cyclic": "s", "map": "o", "greedy": "^", "distill": "*"}


def _lab(c: str) -> str:
    return _LABEL.get(c, f"MPC $H_p={c[3:]}$ (B4)" if c.startswith("mpc") else c)


def _sem_errorbar(ax, x, df):
    ax.errorbar(x, df["Pe"], yerr=1.96 * df["Pe_se"], marker=_MARK.get(
        df["controller"].iloc[0], "d"), capsize=2, label=_lab(df["controller"].iloc[0]))


def curve(csv: Path, xfield: str, out: Path, xlabel: str, logx: bool = False,
          bounds_cols=("helstrom", "heterodyne")) -> Path:
    df = pd.read_csv(csv)
    fig, ax = plt.subplots()
    for c, g in df.groupby("controller", sort=False):
        g = g.sort_values(xfield)
        x = g[xfield].to_numpy()
        if np.isinf(x).any():
            x = np.where(np.isinf(x), np.nanmax(x[np.isfinite(x)]) * 2, x)
        _sem_errorbar(ax, x, g)
    g0 = df.groupby(xfield, sort=True).first().reset_index().sort_values(xfield)
    styles = {"helstrom": ("k", "-"), "heterodyne": ("0.45", "--"),
              "helstrom_eta": ("k", ":"), "homodyne": ("0.7", "-.")}
    for b in bounds_cols:
        if b in g0:
            col, ls = styles.get(b, ("0.6", ":"))
            ax.plot(g0[xfield], g0[b], color=col, ls=ls, lw=1.0,
                    label=b.replace("_", " "))
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"symbol-error probability $P_e$")
    ax.legend(frameon=False, ncol=1)
    fig.savefig(out)
    plt.close(fig)
    print(f"  figure -> {out}")
    return out


def gap_plot(csv: Path, out: Path) -> Path:
    """E1 companion: paired MPC-minus-greedy gap versus slew limit (H1)."""
    df = pd.read_csv(csv)
    if "paired_diff" not in df:
        return out
    fig, ax = plt.subplots()
    for c, g in df[df.controller.str.startswith("mpc")].groupby("controller"):
        g = g.sort_values("dumax_frac")
        x = g["dumax_frac"].to_numpy()
        x = np.where(np.isinf(x), np.nanmax(x[np.isfinite(x)]) * 2, x)
        ax.errorbar(x, -g["paired_diff"], yerr=1.96 * g["paired_diff_se"],
                    marker="^", capsize=2, label=_lab(c))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"slew limit $\Delta u_{\max}/u_{\max}$")
    ax.set_ylabel(r"$P_e^{\rm greedy}-P_e^{\rm MPC}$ (paired)")
    ax.legend(frameon=False)
    fig.savefig(out)
    plt.close(fig)
    print(f"  figure -> {out}")
    return out


def policy_map(csv: Path, out: Path, K_show: int | None = None) -> Path:
    """E5: scatter of optimised displacements in the complex plane."""
    df = pd.read_csv(csv)
    ks = sorted(df.k.unique()) if K_show is None else list(range(K_show))
    n = len(ks)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, ks):
        g = df[df.k == k]
        s = ax.scatter(g.uR, g.uI, c=g.bmax, s=2, cmap="viridis", vmin=0.25, vmax=1.0)
        ax.set_title(f"$k={k}$")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\mathrm{Re}\,u_k$")
    axes[0].set_ylabel(r"$\mathrm{Im}\,u_k$")
    fig.colorbar(s, ax=axes.tolist(), label=r"$\max_m b_k^m$", shrink=0.85)
    fig.savefig(out)
    plt.close(fig)
    print(f"  figure -> {out}")
    return out


def dnull_plot(csv: Path, out: Path) -> Path:
    df = pd.read_csv(csv)
    fig, ax = plt.subplots()
    ax.hist(df.dnull, bins=60, color="0.4")
    ax.set_xlabel(r"$d^{\rm null}_k$")
    ax.set_ylabel("count")
    fig.savefig(out)
    plt.close(fig)
    print(f"  figure -> {out}")
    return out
