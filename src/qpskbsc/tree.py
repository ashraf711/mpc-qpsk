"""Scenario-tree belief-space MPC.

The decision variable is a *tree* of controls: at prediction depth l there are
2^l nodes (one per on/off observation history), and node j at depth l+1 is the
child of node j//2 at depth l.  The expectation over future observations is
evaluated by exact enumeration of the 2^H leaves, so the objective is
deterministic and differentiable.

The information state carried through the tree is the joint log-weight array
``logw`` of shape (..., 4, Nh) over (hypothesis, channel grid point).  Perfect
CSI is the special case Nh = 1 with a per-trial channel value, so a single code
path covers all CSI regimes.
"""

from __future__ import annotations

import math

import torch

from .config import Params
from .geometry import project
from .physics import intensity, log_kernel, phi_exact, phi_smooth, view_h


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------
def tree_objective(U: list[torch.Tensor], logw0: torch.Tensor, h: torch.Tensor,
                   uprev: torch.Tensor, p: Params, tau: float,
                   alphas: torch.Tensor) -> torch.Tensor:
    """Expected truncated cost of a control tree.

    Parameters
    ----------
    U      : list of length H; ``U[l]`` is complex with shape (B, S, 2**l)
    logw0  : (B, 4, Nh) normalised joint log weights at the root
    h      : (B, Nh) channel grid (Nh = 1 under perfect CSI)
    uprev  : (B,) complex, the actuator state entering the horizon
    tau    : smoothing parameter for the terminal cost (0 -> exact)

    Returns
    -------
    (B, S) tensor of expected costs
    """
    B, S = U[0].shape[0], U[0].shape[1]
    H = len(U)
    Nh = logw0.shape[-1]

    logw = logw0[:, None, None, :, :].expand(B, S, 1, 4, Nh)
    logP = torch.zeros(B, S, 1, dtype=logw0.dtype, device=logw0.device)
    u_par = uprev[:, None, None].expand(B, S, 1)
    cost = torch.zeros(B, S, dtype=logw0.dtype, device=logw0.device)

    for l in range(H):
        u = U[l]                                    # (B, S, 2**l)
        w = logP.exp()
        if p.rho_e > 0 or p.rho_d > 0:
            stage = p.rho_e * u.abs().pow(2) + p.rho_d * (u - u_par).abs().pow(2)
            cost = cost + (w * stage).sum(-1)

        hv = view_h(h, u.dim())                     # (B, 1, 1, Nh)
        lam = intensity(u, hv, p, alphas)           # (B, S, 2**l, 4, Nh)
        lp0, lp1 = log_kernel(lam)

        children_w, children_P = [], []
        for lp in (lp0, lp1):
            joint = logw + lp
            lz = torch.logsumexp(joint.flatten(-2), dim=-1)          # (B,S,2**l)
            children_w.append(joint - lz[..., None, None])
            children_P.append(logP + lz)
        logw = torch.stack(children_w, dim=3).reshape(B, S, 2 ** (l + 1), 4, Nh)
        logP = torch.stack(children_P, dim=3).reshape(B, S, 2 ** (l + 1))
        u_par = u.repeat_interleave(2, dim=-1)

    b = torch.logsumexp(logw, dim=-1).exp()          # (B, S, 2**H, 4)
    phi = phi_exact(b) if tau <= 0 else phi_smooth(b, tau)
    return cost + (logP.exp() * phi).sum(-1)


# ---------------------------------------------------------------------------
# multi-start initialisation
# ---------------------------------------------------------------------------
def build_starts(uprev: torch.Tensor, h_point: torch.Tensor, p: Params,
                 alphas: torch.Tensor) -> torch.Tensor:
    """Feasible multi-start displacements, shape (B, S).

    Starts are a polar grid over the amplitude disk, the four nulling actions
    evaluated at the controller's point estimate of h, the previous action, and
    the origin.  Section 6.4 of the formulation (C_4 degeneracy) requires this.
    """
    B = uprev.shape[0]
    dev, cd, rd = uprev.device, uprev.dtype, h_point.dtype
    amps = torch.linspace(1.0 / p.n_starts_amp, 1.0, p.n_starts_amp,
                          device=dev, dtype=rd) * p.umax
    phases = torch.arange(p.n_starts_phase, device=dev, dtype=rd) \
        * (2 * math.pi / p.n_starts_phase)
    grid = torch.polar(amps[:, None].expand(-1, p.n_starts_phase),
                       phases[None, :].expand(p.n_starts_amp, -1)).reshape(-1).to(cd)
    grid = grid[None, :].expand(B, -1)

    nulls = (h_point[:, None].to(cd) * alphas[None, :] / math.sqrt(p.K))  # (B,4)
    extra = torch.stack([uprev, torch.zeros_like(uprev)], dim=1)          # (B,2)
    starts = torch.cat([grid, nulls, extra], dim=1)                       # (B,S)
    return project(starts, uprev[:, None], p.umax, p.dumax)


def _propagate(logw: torch.Tensor, u: torch.Tensor, h: torch.Tensor,
               p: Params, alphas: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """One prediction step: (B,S,n,4,Nh) -> (B,S,2n,4,Nh) plus log branch probs."""
    hv = view_h(h, u.dim())
    lam = intensity(u, hv, p, alphas)
    lp0, lp1 = log_kernel(lam)
    ws, zs = [], []
    for lp in (lp0, lp1):
        joint = logw + lp
        lz = torch.logsumexp(joint.flatten(-2), dim=-1)
        ws.append(joint - lz[..., None, None])
        zs.append(lz)
    B, S = u.shape[0], u.shape[1]
    n2 = 2 * u.shape[2]
    return (torch.stack(ws, dim=3).reshape(B, S, n2, 4, logw.shape[-1]),
            torch.stack(zs, dim=3).reshape(B, S, n2))


def init_tree(starts: torch.Tensor, logw0: torch.Tensor, h: torch.Tensor,
              uprev: torch.Tensor, h_point: torch.Tensor, p: Params,
              horizon: int, alphas: torch.Tensor, jitter: float = 0.0,
              gen: torch.Generator | None = None) -> list[torch.Tensor]:
    """Belief-aware initialisation of every node of the control tree.

    Multi-starting only the root leaves the deeper nodes inheriting the root
    action, and the deeper subproblems then converge to poor local minima --
    measured at roughly 25% excess cost on a two-stage reference problem.  Each
    deeper node is therefore initialised at the nulling action for *its own*
    predicted MAP hypothesis (the B2 policy, a good warm start), optionally
    jittered so that several tree initialisations can be run in parallel.
    """
    B, S = starts.shape
    Nh = logw0.shape[-1]
    U = [starts[:, :, None]]
    logw = logw0[:, None, None, :, :].expand(B, S, 1, 4, Nh)
    u_par = uprev[:, None, None].expand(B, S, 1)
    hp = h_point[:, None, None].to(starts.dtype)

    for l in range(horizon):
        if l > 0:
            b = torch.logsumexp(logw, dim=-1).exp()          # (B,S,n,4)
            m = b.argmax(dim=-1)
            u = hp * alphas[m] / math.sqrt(p.K)
            if jitter > 0:
                sh = u.shape
                noise = torch.randn(*sh, generator=gen, device="cpu").to(u.device) \
                    + 1j * torch.randn(*sh, generator=gen, device="cpu").to(u.device)
                u = u + (jitter * p.dumax) * noise.to(u.dtype)
            u = project(u, u_par, p.umax, p.dumax)
            U.append(u)
        if l + 1 < horizon:
            logw, _ = _propagate(logw, U[l], h, p, alphas)
            u_par = U[l].repeat_interleave(2, dim=-1)
    return U


def _project_tree_(U: list[torch.Tensor], uprev: torch.Tensor, p: Params) -> None:
    """In-place projection of every node onto its parent's feasible set."""
    par = uprev[:, None, None].expand_as(U[0])
    for l in range(len(U)):
        U[l].copy_(project(U[l], par, p.umax, p.dumax))
        if l + 1 < len(U):
            par = U[l].repeat_interleave(2, dim=-1)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def solve_mpc(logw0: torch.Tensor, h: torch.Tensor, uprev: torch.Tensor,
              h_point: torch.Tensor, p: Params, horizon: int,
              alphas: torch.Tensor, iters: int | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the scenario-tree MPC problem for a batch of information states.

    The objective is nonsmooth (Phi is piecewise linear) and its minimiser
    frequently lies exactly on the slew boundary, where a projected
    momentum method does not converge: the iterate passes through the optimum
    and then drifts along the active constraint.  Two devices handle this.

    1. *Best-iterate tracking.*  Every iterate is scored with the exact Phi and
       the running argmin is retained per multi-start, so the returned control
       is the best point ever visited rather than the last one.  This makes the
       method monotone in the returned value by construction.
    2. *Alternating refinement.*  Beliefs at prediction depth l depend only on
       controls at depths < l, so with the root frozen the deeper levels form a
       separable, well conditioned subproblem; with the deeper levels frozen the
       root step is justified by Danskin's theorem.

    Log-sum-exp smoothing of Phi is available (``tau0 > 0``) but is off by
    default: the smoothing gap tau*log(sum exp(b/tau)) - max_m b^m is largest
    for *flat* beliefs, so the surrogate systematically rewards uninformative
    measurements.  Exact subgradients are better behaved here.

    Returns
    -------
    (u_star, J_star) : applied displacement (B,) complex and its exact
    (unsmoothed) predicted cost (B,).
    """
    iters = p.iters if iters is None else iters
    B = uprev.shape[0]
    starts = build_starts(uprev, h_point, p, alphas)     # (B, S)
    S = starts.shape[1]
    idx = torch.arange(B, device=uprev.device)

    if horizon == 1:
        U = [starts[:, :, None].clone()]
    else:
        gen = torch.Generator().manual_seed(p.seed + 9973 * horizon)
        trees = [init_tree(starts, logw0, h, uprev, h_point, p, horizon, alphas,
                           jitter=j, gen=gen)
                 for j in ([0.0] + [p.init_jitter] * (p.n_tree_inits - 1))]
        U = [torch.cat([t[l] for t in trees], dim=1).contiguous()
             for l in range(horizon)]
        S = U[0].shape[1]
        starts = U[0][:, :, 0]
        idx = torch.arange(B, device=uprev.device)
    xs = [torch.stack([u.real, u.imag], dim=-1).contiguous().requires_grad_(True)
          for u in U]
    # Step size is scaled to the *slew* radius, not the amplitude radius: the
    # local feasible region has radius Du_max, and a step comparable to u_max
    # overshoots it so badly that no descent step ever improves on the
    # initialisation.
    lr = p.lr * max(min(p.dumax, p.umax), 1e-9)

    Jbest = torch.full((B, S), float("inf"), dtype=logw0.dtype, device=logw0.device)
    Ubest = [u.clone() for u in U]

    def _step(opt, tau):
        nonlocal Jbest
        Uc = [torch.complex(x[..., 0], x[..., 1]) for x in xs]
        J = tree_objective(Uc, logw0, h, uprev, p, tau, alphas)
        with torch.no_grad():
            Jex = J if tau <= 0 else tree_objective(Uc, logw0, h, uprev, p, 0.0,
                                                    alphas)
            imp = Jex < Jbest
            if imp.any():
                Jbest = torch.where(imp, Jex, Jbest)
                for ub, uc in zip(Ubest, Uc):
                    ub.copy_(torch.where(imp.unsqueeze(-1), uc, ub))
        opt.zero_grad(set_to_none=True)
        J.sum().backward()
        opt.step()
        with torch.no_grad():
            Uc = [torch.complex(x[..., 0], x[..., 1]) for x in xs]
            _project_tree_(Uc, uprev, p)
            for x, uc in zip(xs, Uc):
                x[..., 0].copy_(uc.real)
                x[..., 1].copy_(uc.imag)

    # ---- phase A: joint projected-gradient descent -------------------------
    optA = torch.optim.Adam(xs, lr=lr)
    for i in range(iters):
        frac = i / max(iters - 1, 1)
        tau = 0.0 if p.tau0 <= 0 else p.tau0 * (max(p.tau1, 1e-12) / p.tau0) ** frac
        _step(optA, tau)

    # ---- phase B: alternating inner / outer refinement ---------------------
    if horizon > 1 and p.polish_rounds > 0:
        inner = torch.optim.Adam(xs[1:], lr=lr * 0.5)
        outer = torch.optim.Adam(xs[:1], lr=lr * 0.5)
        for _ in range(p.polish_rounds):
            for _ in range(p.polish_iters):
                _step(inner, 0.0)
            for _ in range(p.polish_iters):
                _step(outer, 0.0)
        for _ in range(p.polish_iters):
            _step(inner, 0.0)

    with torch.no_grad():
        Jex = tree_objective([u for u in Ubest], logw0, h, uprev, p, 0.0, alphas)
        Jex = torch.minimum(Jex, Jbest)
        best = Jex.argmin(dim=1)
        u_star = Ubest[0][idx, best, 0]
        J_star = Jex.gather(1, best[:, None]).squeeze(1)
    return u_star.detach(), J_star.detach()


def eval_actions(actions: torch.Tensor, logw0: torch.Tensor, h: torch.Tensor,
                 uprev: torch.Tensor, p: Params, horizon: int,
                 alphas: torch.Tensor) -> torch.Tensor:
    """Exact tree cost of a batch of *first* actions, with all deeper nodes
    held at their parent value.  Used by the greedy sanity check and by the
    dynamic-programming bracket."""
    B, S = actions.shape
    U = [actions[:, :, None].expand(B, S, 2 ** l).clone() for l in range(horizon)]
    with torch.no_grad():
        return tree_objective(U, logw0, h, uprev, p, 0.0, alphas)
