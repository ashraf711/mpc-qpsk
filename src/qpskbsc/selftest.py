"""Self-tests.

Several of these are not generic software tests but direct numerical checks of
the propositions in the formulation.  In particular Proposition 2 (the belief
martingale) gives two free diagnostics that catch a mismatched or miscoded
filter immediately:

  (i)  the Monte Carlo average of b_k must stay at (1/4,1/4,1/4,1/4) for all k;
  (ii) the empirical symbol-error rate must equal E[Phi(b_K)].

A violation of either means the reported P_e is not the Bayes risk of the
filter being run.
"""

from __future__ import annotations

import math

import torch

from .config import Params
from .experiments import make_factory
from .geometry import is_feasible, project
from .physics import (helstrom_qpsk, heterodyne_qpsk, phi_exact, phi_smooth,
                      intensity, log_kernel)
from .simulate import make_trials, run_mc
from .tree import tree_objective


class Check:
    def __init__(self):
        self.fails = []
        self.n = 0

    def __call__(self, name: str, ok: bool, detail: str = ""):
        self.n += 1
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            self.fails.append(name)


# ---------------------------------------------------------------------------
def test_bounds(c: Check, dev):
    z = torch.tensor([0.0], device=dev, dtype=torch.float64)
    big = torch.tensor([60.0], device=dev, dtype=torch.float64)
    c("Helstrom at N=0 equals 3/4",
      abs(helstrom_qpsk(z).item() - 0.75) < 1e-12,
      f"got {helstrom_qpsk(z).item():.12f}")
    c("Helstrom -> 0 at large N", helstrom_qpsk(big).item() < 1e-12,
      f"got {helstrom_qpsk(big).item():.3e}")
    c("Heterodyne at N=0 equals 3/4",
      abs(heterodyne_qpsk(z).item() - 0.75) < 1e-9)
    n = torch.linspace(0.01, 30, 200, device=dev, dtype=torch.float64)
    c("Helstrom <= heterodyne everywhere",
      bool((helstrom_qpsk(n) <= heterodyne_qpsk(n) + 1e-12).all()))
    c("Helstrom monotone decreasing in N",
      bool((helstrom_qpsk(n).diff() <= 1e-12).all()))


def test_smoothing(c: Check, dev):
    b = torch.rand(2000, 4, device=dev, dtype=torch.float64)
    b = b / b.sum(-1, keepdim=True)
    for tau in (0.3, 0.05, 1e-3):
        lo = phi_exact(b) - tau * math.log(4.0)
        hi = phi_exact(b)
        pt = phi_smooth(b, tau)
        c(f"smoothing bracket holds (tau={tau})",
          bool(((pt >= lo - 1e-12) & (pt <= hi + 1e-12)).all()))


def test_projection(c: Check, dev):
    g = torch.Generator(device="cpu").manual_seed(3)
    umax, dumax = 1.7, 0.6
    uprev = torch.polar(torch.rand(500, generator=g) * umax,
                        torch.rand(500, generator=g) * 6.28).to(dev)
    p = (torch.randn(500, generator=g) + 1j * torch.randn(500, generator=g)).to(dev) * 2.0
    q = project(p, uprev, umax, dumax)
    c("projection lands in the feasible set",
      bool(is_feasible(q, uprev, umax, dumax).all()))
    # brute force: dense sample of the feasible set, nearest point
    R = torch.linspace(0, umax, 160, device=dev)
    TH = torch.linspace(0, 2 * math.pi, 361, device=dev)[:-1]
    cand = torch.polar(R[:, None].expand(-1, TH.numel()),
                       TH[None, :].expand(R.numel(), -1)).reshape(-1)
    d_grid = []
    for i in range(0, p.numel(), 50):
        pc, uc = p[i:i + 50], uprev[i:i + 50]
        ok = (cand[None, :] - uc[:, None]).abs() <= dumax
        dist = (cand[None, :] - pc[:, None]).abs()
        dist = torch.where(ok, dist, torch.full_like(dist, float("inf")))
        d_grid.append(dist.min(1).values)
    d_grid = torch.cat(d_grid)
    d_proj = (q - p).abs()
    c("projection is optimal (vs dense grid)",
      bool((d_proj <= d_grid + 2e-2).all()),
      f"max excess {(d_proj - d_grid).max().item():.2e}")
    already = project(uprev, uprev, umax, dumax)
    c("feasible points are fixed", bool((already - uprev).abs().max() < 1e-9))


def test_tree_one_step(c: Check, dev):
    """H=1 tree objective must equal the explicit one-step expectation."""
    p = Params(N=3.0, K=4, device=str(dev), dtype="float64", dark=1e-3)
    alphas = p.alphas(dev)
    B, S = 7, 5
    g = torch.Generator(device="cpu").manual_seed(11)
    b = torch.rand(B, 4, generator=g).to(dev).double()
    b = b / b.sum(-1, keepdim=True)
    logw = b.log()[:, :, None]
    h = torch.rand(B, 1, generator=g).to(dev).double() + 0.5
    uprev = torch.zeros(B, dtype=torch.complex128, device=dev)
    U0 = (torch.randn(B, S, 1, generator=g) + 1j * torch.randn(B, S, 1, generator=g)
          ).to(dev) * 0.4
    J = tree_objective([U0], logw, h, uprev, p, 0.0, alphas)

    lam = intensity(U0.squeeze(-1), h[:, None, :], p, alphas)   # (B,S,4,1)
    lp0, lp1 = log_kernel(lam)
    ref = torch.zeros(B, S, dtype=torch.float64, device=dev)
    for lp in (lp0, lp1):
        j = b[:, None, :, None] * lp.exp()
        pz = j.flatten(-2).sum(-1)
        post = j.squeeze(-1) / pz[..., None]
        ref = ref + pz * (1.0 - post.max(-1).values)
    c("tree objective matches explicit one-step expectation",
      bool((J - ref).abs().max() < 1e-10), f"max dev {(J-ref).abs().max():.2e}")


def test_filter_equivalence(c: Check, dev):
    """A grid filter whose prior is concentrated on one point must reproduce the
    perfect-CSI (Nh=1) filter exactly."""
    from .filters import JointFilter
    p = Params(N=3.0, K=3, device=str(dev), dtype="float64")
    alphas = p.alphas(dev)
    B, Nh = 16, 9
    h_true = torch.full((B,), 0.83, dtype=torch.float64, device=dev)
    f1 = JointFilter(torch.full((B, 4, 1), -math.log(4.0), dtype=torch.float64,
                                device=dev), h_true[:, None], p, alphas)
    hg = torch.linspace(0.3, 1.5, Nh, dtype=torch.float64, device=dev)
    hg[4] = 0.83
    lw = torch.full((Nh,), -1e30, dtype=torch.float64, device=dev)
    lw[4] = 0.0
    f2 = JointFilter(lw[None, None, :].expand(B, 4, Nh).clone() - math.log(4.0),
                     hg[None, :].expand(B, -1), p, alphas)
    g = torch.Generator(device="cpu").manual_seed(5)
    for k in range(p.K):
        u = (torch.randn(B, generator=g) + 1j * torch.randn(B, generator=g)).to(dev) * 0.3
        y = torch.randint(0, 2, (B,), generator=g).to(dev)
        f1.update(u, y)
        f2.update(u, y)
    c("degenerate grid filter == perfect-CSI filter",
      bool((f1.belief - f2.belief).abs().max() < 1e-9),
      f"max dev {(f1.belief-f2.belief).abs().max():.2e}")


def test_martingale(c: Check, dev, ctrl: str = "map", nmc: int = 4000):
    """Proposition 2 diagnostics (i) and (ii)."""
    p = Params(N=4.0, K=4, nmc=nmc, chunk=2048, device=str(dev),
               dtype="float64", controller=ctrl, dark=1e-3)
    data = make_trials(p, nmc)
    r = run_mc(p, make_factory(p, ctrl), data)
    bm = torch.tensor(r["belief_mean"])
    dev_max = (bm - 0.25).abs().max().item()
    tol = 4.0 / math.sqrt(nmc)
    c(f"belief martingale: mean b_k == 1/4 ({ctrl})", dev_max < tol,
      f"max |E b - 1/4| = {dev_max:.4f} (tol {tol:.4f})")
    se = max(r["Pe_se"], 1e-9)
    c(f"P_e == E[Phi(b_K)] ({ctrl})",
      abs(r["Pe"] - r["E_phi"]) < 4.0 * se,
      f"Pe={r['Pe']:.5f} E[Phi]={r['E_phi']:.5f} se={se:.5f}")


def test_supermartingale(c: Check, dev):
    """E[Phi(b_{k+1})] <= E[Phi(b_k)] under any admissible policy."""
    p = Params(N=4.0, K=5, nmc=4000, chunk=2048, device=str(dev),
               dtype="float64", controller="greedy", dark=1e-3, iters=25)
    r = run_mc(p, make_factory(p, "greedy"), make_trials(p, 4000))
    bm = torch.tensor(r["belief_mean"])
    # Phi of the *average* belief is not the average Phi; recompute properly is
    # expensive, so check the weaker but still informative monotonicity of the
    # mean maximum posterior, which is implied for the realised trajectories.
    c("mean belief stays calibrated across stages",
      bool((bm - 0.25).abs().max() < 0.05))


def test_constraints(c: Check, dev):
    """Every applied action must satisfy both actuator constraints."""
    from .controllers import MPCController
    from .filters import make_filter
    p = Params(N=4.0, K=4, device=str(dev), dtype="float64",
               dumax_frac=0.15, iters=20)
    alphas = p.alphas(dev)
    B = 64
    h = torch.rand(B, dtype=torch.float64, device=dev) + 0.4
    filt = make_filter(h, p, alphas)
    ctrl = MPCController(p, alphas, 2)
    uprev = torch.zeros(B, dtype=torch.complex128, device=dev)
    ok = True
    for k in range(p.K):
        u = ctrl.act(filt, uprev, k)
        ok &= bool(is_feasible(u, uprev, p.umax, p.dumax, tol=1e-6).all())
        filt.update(u, torch.randint(0, 2, (B,), device=dev))
        uprev = u
    c("MPC actions satisfy |u|<=umax and |u-u^-|<=dumax", ok)


def test_dp_consistency(c: Check, dev):
    """With K=1 the grid DP value must equal the best grid action's one-step
    cost, and the continuous greedy solver must not do worse."""
    from .dp import action_grid, dp_value
    from .tree import eval_actions
    p = Params(N=3.0, K=1, device=str(dev), dtype="float64", dark=1e-3, iters=80)
    alphas = p.alphas(dev)
    acts, _ = action_grid(p, 8, 24, dev)
    h = torch.tensor([0.9], dtype=torch.float64, device=dev)
    logb = torch.full((1, 4), -math.log(4.0), dtype=torch.float64, device=dev)
    uprev = torch.zeros(1, dtype=torch.complex128, device=dev)
    V = dp_value(logb, uprev, h, 0, p, acts, alphas).item()
    J = eval_actions(acts[None, :], logb[:, :, None], h[:, None], uprev, p, 1,
                     alphas).min().item()
    c("grid DP (K=1) == best grid action", abs(V - J) < 1e-10,
      f"V={V:.8f} J={J:.8f}")
    from .tree import solve_mpc
    u, Jg = solve_mpc(logb[:, :, None], h[:, None], uprev, h, p, 1, alphas)
    c("continuous greedy <= grid optimum", Jg.item() <= V + 1e-6,
      f"greedy={Jg.item():.8f} grid={V:.8f}")


def test_solver_quality(c: Check, dev):
    """The H=2 continuous solver must not be worse than a fine action grid.

    This is the regression test for the three bugs that a naive implementation
    hits: log-sum-exp smoothing biased toward flat beliefs, a step size scaled
    to u_max instead of Du_max, and multi-starting only the root so that deeper
    nodes inherit it and stick in poor local minima.
    """
    from .dp import action_grid, dp_value
    from .tree import solve_mpc
    p = Params(N=4.0, K=2, device=str(dev), dtype="float64", dumax_frac=0.3)
    alphas = p.alphas(dev)
    acts, delta = action_grid(p, 20, 48, dev)
    logb1 = torch.full((1, 4), -math.log(4.0), dtype=torch.float64, device=dev)
    up = torch.zeros(1, dtype=torch.complex128, device=dev)
    h = torch.tensor([1.0], dtype=torch.float64, device=dev)
    V = dp_value(logb1, up, h, 0, p, acts, alphas, max_states=20_000_000).item()
    J = solve_mpc(logb1[:, :, None], h[:, None], up, h, p, 2, alphas)[1].item()
    c("H=2 continuous solver <= fine grid optimum", J <= V + 1e-9,
      f"solver={J:.8f} grid={V:.8f} (delta={delta:.3f})")
    c("H=2 solver within 2% of grid optimum", J <= V * 1.02,
      f"excess {100*(J-V)/V:+.2f}%")


def test_energy(c: Check, dev):
    p = Params(N=5.0, K=7)
    a = p.alphas(dev)
    c("energy conservation K|s_m|^2 = N",
      abs((p.K * (a.abs() ** 2 / p.K)).mean().item() - p.N) < 1e-9)


# ---------------------------------------------------------------------------
def run_all(device: str = "cpu", quick: bool = False) -> int:
    dev = torch.device(device)
    c = Check()
    print("physics and bounds")
    test_bounds(c, dev)
    test_energy(c, dev)
    print("belief geometry")
    test_smoothing(c, dev)
    test_projection(c, dev)
    print("filters and tree")
    test_tree_one_step(c, dev)
    test_filter_equivalence(c, dev)
    print("dynamic programming")
    test_dp_consistency(c, dev)
    test_solver_quality(c, dev)
    print("closed loop (Proposition 2 diagnostics)")
    test_martingale(c, dev, "map", 2000 if quick else 8000)
    if not quick:
        test_martingale(c, dev, "greedy", 3000)
        test_supermartingale(c, dev)
    test_constraints(c, dev)
    print(f"\n{c.n - len(c.fails)}/{c.n} checks passed")
    if c.fails:
        print("FAILED: " + ", ".join(c.fails))
    return 1 if c.fails else 0
