"""Receiver control policies (baselines B1-B4 of the formulation).

B1 cyclic nulling          -- no feedback, fixed nominal channel
B2 Bayesian MAP nulling    -- posterior feedback, discrete action set
B3 greedy continuous       -- MPC with H_p = 1 (same solver, same constraints)
B4 scenario-tree MPC       -- H_p > 1
D  distilled policy        -- network regression onto B4
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from .config import Params
from .filters import JointFilter
from .geometry import project
from .tree import solve_mpc


class Controller(ABC):
    name = "base"

    def __init__(self, p: Params, alphas: torch.Tensor):
        self.p = p
        self.alphas = alphas

    @abstractmethod
    def act(self, filt: JointFilter, uprev: torch.Tensor, k: int) -> torch.Tensor:
        """Return the displacement u_k, shape (B,) complex."""

    # -- shared helpers ----------------------------------------------------
    def h_point(self, filt: JointFilter) -> torch.Tensor:
        """Controller's point estimate of the channel."""
        if filt.h.shape[1] == 1:
            return filt.h[:, 0]
        return filt.h_mean

    def ctrl_state(self, filt: JointFilter) -> tuple[torch.Tensor, torch.Tensor]:
        """(logw0, h_grid) used inside the optimiser, possibly coarsened."""
        if filt.h.shape[1] <= self.p.nh_ctrl:
            return filt.logw, filt.h
        # stratified subsample of the channel grid by batch-mean posterior mass
        lw_h = filt.h_post_logw                       # (B, Nh)
        w = lw_h.exp().mean(0)
        cdf = torch.cumsum(w / w.sum(), 0)
        q = (torch.arange(self.p.nh_ctrl, device=w.device, dtype=w.dtype) + 0.5) \
            / self.p.nh_ctrl
        idx = torch.unique(torch.searchsorted(cdf, q).clamp(max=w.shape[0] - 1))
        lw = filt.logw[:, :, idx]
        lw = lw - torch.logsumexp(lw.flatten(-2), dim=-1)[:, None, None]
        return lw, filt.h[:, idx]


# ---------------------------------------------------------------------------
class CyclicNulling(Controller):
    """B1: tested hypothesis cycles, displacement uses a fixed nominal channel."""

    name = "cyclic"

    def __init__(self, p: Params, alphas: torch.Tensor, hnom: float):
        super().__init__(p, alphas)
        self.hnom = float(hnom)

    def act(self, filt, uprev, k):
        m = k % 4
        u = torch.full_like(uprev, complex(self.hnom, 0.0)) \
            * self.alphas[m] / math.sqrt(self.p.K)
        return project(u, uprev, self.p.umax, self.p.dumax)


# ---------------------------------------------------------------------------
class MapNulling(Controller):
    """B2: null the currently most probable hypothesis."""

    name = "map"

    def act(self, filt, uprev, k):
        m = filt.belief.argmax(dim=-1)
        h = self.h_point(filt).to(uprev.dtype)
        u = h * self.alphas[m] / math.sqrt(self.p.K)
        return project(u, uprev, self.p.umax, self.p.dumax)


# ---------------------------------------------------------------------------
class MPCController(Controller):
    """B3 (H_p = 1) and B4 (H_p > 1): identical code path, differing horizon."""

    def __init__(self, p: Params, alphas: torch.Tensor, Hp: int):
        super().__init__(p, alphas)
        self.Hp = int(Hp)
        self.name = "greedy" if Hp == 1 else f"mpc{Hp}"
        self.last_J = None

    def act(self, filt, uprev, k):
        Hk = min(self.Hp, self.p.K - k)
        logw0, hg = self.ctrl_state(filt)
        u, J = solve_mpc(logw0, hg, uprev, self.h_point(filt), self.p, Hk,
                         self.alphas)
        self.last_J = J
        return u


# ---------------------------------------------------------------------------
class DistilledController(Controller):
    """Amortised policy: a network trained by regression onto MPC actions."""

    name = "distill"

    def __init__(self, p: Params, alphas: torch.Tensor, net):
        super().__init__(p, alphas)
        self.net = net

    def act(self, filt, uprev, k):
        feats = policy_features(filt, uprev, k, self.p, self.h_point(filt))
        with torch.no_grad():
            u = self.net(feats, uprev, self.p)
        return project(u, uprev, self.p.umax, self.p.dumax)


def policy_features(filt: JointFilter, uprev: torch.Tensor, k: int, p: Params,
                    hpt: torch.Tensor) -> torch.Tensor:
    """Feature vector for the distilled policy: (b, log h, u^-, k/K)."""
    b = filt.belief
    scale = max(p.umax, 1e-9)
    return torch.cat([
        b,
        torch.log(hpt.clamp_min(1e-6))[:, None],
        (uprev.real / scale)[:, None],
        (uprev.imag / scale)[:, None],
        torch.full_like(b[:, :1], k / max(p.K, 1)),
    ], dim=1)


# ---------------------------------------------------------------------------
def build_controller(p: Params, alphas: torch.Tensor, hnom: float | None = None,
                     net=None) -> Controller:
    c = p.controller
    if c == "cyclic":
        if hnom is None:
            raise ValueError("cyclic baseline requires hnom")
        return CyclicNulling(p, alphas, hnom)
    if c == "map":
        return MapNulling(p, alphas)
    if c == "greedy":
        return MPCController(p, alphas, 1)
    if c == "mpc":
        return MPCController(p, alphas, p.Hp)
    if c == "distill":
        if net is None:
            raise ValueError("distilled controller requires a trained network")
        return DistilledController(p, alphas, net)
    raise ValueError(f"unknown controller '{c}'")
