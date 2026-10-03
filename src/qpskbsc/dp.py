"""Exact dynamic programming on a finite action grid, and the certified bracket.

Proposition 3 of the formulation: if the continuous feasible set is replaced by
a finite grid with covering radius delta, the resulting value satisfies

    V_grid - K L_Q delta  <=  V*  <=  min(V_grid, J(any admissible policy)),

so a candidate policy's suboptimality is certified by J - (V_grid - K L_Q delta).
Note that V_grid is NOT an upper bound on J: restricting the action set to a grid
can only raise the optimal cost, so a continuous policy may achieve a closed-loop
cost below V_grid.  Only the delta-corrected lower bound is valid, and the
certificate is informative only once delta is small.  Only tractable for small K (the reachable tree grows as
(2G)^K), which is exactly the regime where the certificate is wanted.
"""

from __future__ import annotations

import math

import torch

from .config import Params
from .physics import intensity, log_kernel, view_h

_INF = float("inf")


# ---------------------------------------------------------------------------
def action_grid(p: Params, n_a: int, n_p: int, device, include_zero: bool = True
                ) -> tuple[torch.Tensor, float]:
    """Polar grid over the amplitude disk, with its covering radius."""
    amps = torch.linspace(p.umax / n_a, p.umax, n_a, device=device, dtype=p.tdtype)
    ph = torch.arange(n_p, device=device, dtype=p.tdtype) * (2 * math.pi / n_p)
    g = torch.polar(amps[:, None].expand(-1, n_p), ph[None, :].expand(n_a, -1)
                    ).reshape(-1).to(p.cdtype)
    if include_zero:
        g = torch.cat([torch.zeros(1, dtype=p.cdtype, device=device), g])
    # radial half-spacing and worst-case arc half-spacing at the outer radius
    delta = max(p.umax / (2.0 * n_a), p.umax * math.pi / n_p)
    return g, float(delta)


# ---------------------------------------------------------------------------
def dp_value(logb: torch.Tensor, uprev: torch.Tensor, h: torch.Tensor, k: int,
             p: Params, actions: torch.Tensor, alphas: torch.Tensor,
             max_states: int = 4_000_000) -> torch.Tensor:
    """Exact backward value V_k(b, h, u^-) restricted to ``actions``.

    All arguments are batched over states: ``logb`` (Ns,4), ``uprev`` (Ns,),
    ``h`` (Ns,).
    """
    if k >= p.K:
        return 1.0 - logb.exp().max(dim=-1).values

    Ns, G = logb.shape[0], actions.shape[0]
    if Ns * G > max_states:
        raise MemoryError(
            f"DP state explosion: {Ns}x{G} exceeds max_states={max_states}. "
            "Reduce --dp-na/--dp-np or K.")

    u = actions[None, :].expand(Ns, G)                            # (Ns,G)
    feas = (u.abs() <= p.umax + 1e-9) & ((u - uprev[:, None]).abs() <= p.dumax + 1e-9)

    hv = view_h(h[:, None], u.dim())                              # (Ns,1,1)
    lam = intensity(u, hv, p, alphas)                             # (Ns,G,4,1)
    lp0, lp1 = log_kernel(lam)
    lb = logb[:, None, :, None]                                   # (Ns,1,4,1)

    stage = p.rho_e * u.abs().pow(2) + p.rho_d * (u - uprev[:, None]).abs().pow(2)
    Q = stage.clone()

    hrep = h.repeat_interleave(G)
    urep = u.reshape(-1)
    for lp in (lp0, lp1):
        joint = lb + lp
        lz = torch.logsumexp(joint.flatten(-2), dim=-1)           # (Ns,G)
        child = (joint - lz[..., None, None]).squeeze(-1)         # (Ns,G,4)
        V = dp_value(child.reshape(-1, 4), urep, hrep, k + 1, p, actions,
                     alphas, max_states)
        Q = Q + lz.exp() * V.reshape(Ns, G)

    Q = torch.where(feas, Q, torch.full_like(Q, _INF))
    return Q.min(dim=1).values


# ---------------------------------------------------------------------------
def lipschitz_estimate(p: Params, actions: torch.Tensor, alphas: torch.Tensor,
                       h: torch.Tensor, n_probe: int = 256,
                       eps: float | None = None) -> float:
    """Numerical estimate of L_Q via finite differences of the one-step Q at
    randomly sampled beliefs and actuator states.  Reported alongside the
    analytic bound of Proposition 3."""
    dev = actions.device
    eps = eps if eps is not None else 1e-3 * max(p.umax, 1e-9)
    g = torch.Generator(device="cpu").manual_seed(1234)
    b = torch.rand(n_probe, 4, generator=g).to(dev).to(p.tdtype)
    logb = torch.log(b / b.sum(1, keepdim=True))
    uprev = torch.zeros(n_probe, dtype=p.cdtype, device=dev)
    hh = h[torch.randint(0, h.numel(), (n_probe,), generator=g)].to(dev)

    ang = torch.rand(n_probe, generator=g).to(dev).to(p.tdtype) * 2 * math.pi
    rad = torch.rand(n_probe, generator=g).to(dev).to(p.tdtype) * p.umax
    u0 = torch.polar(rad, ang).to(p.cdtype)
    u1 = u0 + eps

    def q_one(u):
        lam = intensity(u[:, None], view_h(hh[:, None], 2), p, alphas)
        lp0, lp1 = log_kernel(lam)
        lb = logb[:, None, :, None]
        tot = p.rho_e * u[:, None].abs().pow(2)
        for lp in (lp0, lp1):
            joint = lb + lp
            lz = torch.logsumexp(joint.flatten(-2), dim=-1)
            child = (joint - lz[..., None, None]).squeeze(-1)
            tot = tot + lz.exp() * (1.0 - child.exp().max(-1).values)
        return tot.squeeze(1)

    return float(((q_one(u1) - q_one(u0)).abs() / eps).max().item())


# ---------------------------------------------------------------------------
def dp_bracket(p: Params, n_a: int = 8, n_p: int = 16, n_h: int = 16,
               max_states: int = 4_000_000) -> dict:
    """Fading-averaged grid-DP value and the certified bracket width.

    The channel is integrated on a quadrature grid, so the returned value is
    E_h[V_0(b_0, h, u_init)] -- directly comparable to the Monte Carlo cost of
    any perfect-CSI policy.
    """
    from . import channel as ch
    dev = torch.device(p.device)
    alphas = p.alphas(dev)
    acts, delta = action_grid(p, n_a, n_p, dev)
    hg, lwh = ch.quad_grid(p, n_h, dev, p.tdtype)
    w = lwh.exp()

    logb0 = torch.full((hg.numel(), 4), -math.log(4.0), dtype=p.tdtype, device=dev)
    uprev = torch.full((hg.numel(),), p.uinit, dtype=p.cdtype, device=dev)
    V = dp_value(logb0, uprev, hg, 0, p, acts, alphas, max_states)
    Vbar = float((w * V).sum().item())

    L = lipschitz_estimate(p, acts, alphas, hg)
    gap = p.K * L * delta
    return dict(V_grid=Vbar, delta=delta, L_Q=L, bracket=gap,
                lower=Vbar - gap, upper=Vbar, n_actions=int(acts.numel()),
                n_a=n_a, n_p=n_p, n_h=n_h)
