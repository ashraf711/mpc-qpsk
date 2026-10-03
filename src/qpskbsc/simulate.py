"""Monte Carlo evaluation of a receiver policy.

All controllers are evaluated under *common random numbers*: the hypothesis,
the irradiance, the CSI estimation noise and the uniform variates driving the
detector are drawn once and reused, so policy differences are estimated on
paired samples.  At P_e ~ 1e-3 this is the difference between a resolvable
effect and noise.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from . import channel as ch
from .config import Params
from .controllers import Controller
from .filters import JointFilter, make_filter
from .geometry import is_feasible
from .physics import click_prob, intensity, phi_exact


# ---------------------------------------------------------------------------
@dataclass
class TrialData:
    """Common random numbers shared by every controller at one operating point."""
    H: torch.Tensor      # (n,)   long
    T: torch.Tensor      # (n,)   float64
    unif: torch.Tensor   # (n, K) float
    xhat: torch.Tensor   # (n,)   float, noisy ln T

    def __len__(self) -> int:
        return self.H.shape[0]

    def slice(self, a: int, b: int) -> "TrialData":
        return TrialData(self.H[a:b], self.T[a:b], self.unif[a:b], self.xhat[a:b])


def make_trials(p: Params, n: int, seed: int | None = None) -> TrialData:
    dev = torch.device(p.device)
    gen = torch.Generator(device="cpu").manual_seed(p.seed if seed is None else seed)
    H = torch.randint(0, 4, (n,), generator=gen).to(dev)
    T = ch.sample_T(n, p, gen, dev)
    unif = torch.rand(n, p.K, generator=gen, dtype=torch.float64).to(dev)
    v = torch.randn(n, generator=gen, dtype=torch.float64).to(dev)
    xhat = torch.log(T) + p.sigma_csi * v
    return TrialData(H, T, unif.to(p.tdtype), xhat)


# ---------------------------------------------------------------------------
class FrozenChannelFilter(JointFilter):
    """Ablation: photon counts update the symbol belief but never refine the
    channel posterior.  This is the 'frozen CSI' variant of Remark 5; it is a
    mismatched filter, not a posterior, and is labelled as such."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._lw_h = self.h_post_logw.clone()

    def update(self, u, y):
        super().update(u, y)
        lb = torch.logsumexp(self.logw, dim=-1)                 # (B,4)
        lb = lb - torch.logsumexp(lb, dim=-1, keepdim=True)
        self.logw = lb[:, :, None] + self._lw_h[:, None, :]


# ---------------------------------------------------------------------------
def _build_filter(p: Params, d: TrialData, h_true: torch.Tensor,
                  alphas: torch.Tensor, grid) -> JointFilter:
    mode = p.csi
    if mode == "perfect":
        return make_filter(h_true, p, alphas)
    if mode == "ce":
        hhat = (p.Lp * torch.exp(d.xhat)).sqrt().to(h_true.dtype)
        return make_filter(hhat, p, alphas)
    hg, lw_prior = grid
    if mode == "none":
        lw = lw_prior[None, :].expand(len(d), -1)
    elif mode in ("imperfect", "frozen"):
        lw = ch.csi_posterior_logw(d.xhat.to(hg.dtype), p, hg, lw_prior)
    else:
        raise ValueError(f"unknown csi mode '{mode}'")
    lw = lw.to(h_true.dtype)
    cls = FrozenChannelFilter if mode == "frozen" else JointFilter
    B, Nh = lw.shape
    logw0 = lw[:, None, :].expand(B, 4, Nh).clone() - math.log(4.0)
    return cls(logw0, hg[None, :].expand(B, -1).to(h_true.dtype), p, alphas)


# ---------------------------------------------------------------------------
def run_mc(p: Params, make_ctrl, data: TrialData | None = None,
           collect_states: bool = False) -> dict:
    """Run the Monte Carlo campaign for one controller at one operating point.

    ``make_ctrl`` is a zero-argument factory returning a fresh
    :class:`Controller` (so that per-chunk state is not shared).
    """
    dev = torch.device(p.device)
    alphas = p.alphas(dev)
    data = make_trials(p, p.nmc) if data is None else data
    n = len(data)

    grid = None
    if p.csi in ("none", "imperfect", "frozen"):
        grid = ch.quad_grid(p, p.nh_filter, dev, p.tdtype)

    tot = dict(err=0.0, phi=0.0, Eu=0.0, Su=0.0, dnull=0.0, act_amp=0.0,
               act_slew=0.0, nstage=0, t_ctrl=0.0)
    bsum = torch.zeros(p.K + 1, 4, dtype=torch.float64, device=dev)
    states = []

    for a in range(0, n, p.chunk):
        b = min(a + p.chunk, n)
        d = data.slice(a, b)
        B = b - a
        h_true = (p.Lp * d.T).sqrt().to(p.tdtype)
        filt = _build_filter(p, d, h_true, alphas, grid)
        ctrl: Controller = make_ctrl()
        uprev = torch.full((B,), p.uinit, dtype=p.cdtype, device=dev)

        bsum[0] += filt.belief.double().sum(0)
        for k in range(p.K):
            t0 = time.perf_counter()
            u = ctrl.act(filt, uprev, k)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            tot["t_ctrl"] += time.perf_counter() - t0

            if collect_states:
                states.append((filt.belief.detach().cpu(),
                               ctrl.h_point(filt).detach().cpu(),
                               uprev.detach().cpu(), k,
                               u.detach().cpu()))

            # --- metrics on the applied action --------------------------
            tot["Eu"] += u.abs().pow(2).double().sum().item()
            tot["Su"] += (u - uprev).abs().pow(2).double().sum().item()
            nulls = h_true[:, None].to(p.cdtype) * alphas[None, :] / math.sqrt(p.K)
            tot["dnull"] += (u[:, None] - nulls).abs().min(dim=1).values.double().sum().item()
            tot["act_amp"] += (u.abs() > p.umax - 1e-6 * max(p.umax, 1)).double().sum().item()
            tot["act_slew"] += ((u - uprev).abs() > p.dumax - 1e-6 * max(p.dumax, 1)
                                ).double().sum().item()
            tot["nstage"] += B

            # --- physical observation uses the TRUE channel and symbol ---
            lam_all = intensity(u, h_true[:, None], p, alphas)   # (B,4,1)
            lam_true = lam_all[torch.arange(B, device=dev), d.H, 0]
            pc = click_prob(lam_true)
            y = (d.unif[:, k] < pc).long()
            filt.update(u, y)
            uprev = u
            bsum[k + 1] += filt.belief.double().sum(0)

        bel = filt.belief
        Hhat = bel.argmax(dim=-1)
        tot["err"] += (Hhat != d.H).double().sum().item()
        tot["phi"] += phi_exact(bel).double().sum().item()

    pe = tot["err"] / n
    out = dict(
        Pe=pe,
        Pe_se=math.sqrt(max(pe * (1 - pe), 0.0) / n),
        E_phi=tot["phi"] / n,
        Eu=tot["Eu"] / n,
        Su=tot["Su"] / n,
        dnull=tot["dnull"] / max(tot["nstage"], 1),
        act_amp=tot["act_amp"] / max(tot["nstage"], 1),
        act_slew=tot["act_slew"] / max(tot["nstage"], 1),
        t_per_stage=tot["t_ctrl"] / max(tot["nstage"], 1),
        t_per_symbol=tot["t_ctrl"] / n,
        nmc=n,
        belief_mean=(bsum / n).cpu().tolist(),
    )
    if collect_states:
        out["states"] = states
    return out


# ---------------------------------------------------------------------------
def error_mask(p: Params, make_ctrl, data: TrialData) -> torch.Tensor:
    """Per-trial 0/1 error indicator, for paired (CRN) comparisons."""
    dev = torch.device(p.device)
    alphas = p.alphas(dev)
    grid = ch.quad_grid(p, p.nh_filter, dev, p.tdtype) \
        if p.csi in ("none", "imperfect", "frozen") else None
    out = []
    for a in range(0, len(data), p.chunk):
        b = min(a + p.chunk, len(data))
        d = data.slice(a, b)
        B = b - a
        h_true = (p.Lp * d.T).sqrt().to(p.tdtype)
        filt = _build_filter(p, d, h_true, alphas, grid)
        ctrl = make_ctrl()
        uprev = torch.full((B,), p.uinit, dtype=p.cdtype, device=dev)
        for k in range(p.K):
            u = ctrl.act(filt, uprev, k)
            lam_all = intensity(u, h_true[:, None], p, alphas)
            lam_true = lam_all[torch.arange(B, device=dev), d.H, 0]
            y = (d.unif[:, k] < click_prob(lam_true)).long()
            filt.update(u, y)
            uprev = u
        out.append((filt.belief.argmax(-1) != d.H).double())
    return torch.cat(out)


def paired_diff(e1: torch.Tensor, e2: torch.Tensor) -> dict:
    """Paired difference P_e(1) - P_e(2) with its standard error."""
    d = (e1 - e2)
    n = d.numel()
    mean = d.mean().item()
    se = (d.std(unbiased=True) / math.sqrt(n)).item()
    return dict(diff=mean, diff_se=se, lo=mean - 1.96 * se, hi=mean + 1.96 * se,
                significant=bool(abs(mean) > 1.96 * se))


# ---------------------------------------------------------------------------
def optimize_hnom(p: Params, data: TrialData, n_grid: int = 21,
                  span: tuple[float, float] | None = None) -> float:
    """Offline one-dimensional search for the best constant nominal channel used
    by the cyclic baseline (Eq. 32 of the formulation).  Giving B1 its best
    member prevents the comparison from being against a straw man."""
    from .controllers import CyclicNulling
    lo, hi = span if span else (0.2 * ch.mean_field(p), 2.0 * ch.mean_field(p))
    cands = torch.linspace(lo, hi, n_grid).tolist()
    pp = p.replace(nmc=min(p.nmc, 20000), controller="cyclic")
    dd = data.slice(0, pp.nmc)
    best, best_pe = cands[0], 1.0
    for hnom in cands:
        alphas = pp.alphas(torch.device(pp.device))
        r = run_mc(pp, lambda: CyclicNulling(pp, alphas, hnom), dd)
        if r["Pe"] < best_pe:
            best, best_pe = hnom, r["Pe"]
    return float(best)
