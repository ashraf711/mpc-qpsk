"""Amortised (distilled) policy.

A per-stage iterative MPC solve is orders of magnitude slower than an optical
symbol interval.  This module trains a small network to reproduce the MPC map
(b, h, u^-, k) -> u on the reachable belief set.  The output layer is
parameterised so that the slew and amplitude constraints hold *by construction*,
so the distilled policy is admissible without post-hoc clipping.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import Params
from .controllers import MPCController, policy_features
from .geometry import project
from .simulate import TrialData, make_trials
from . import channel as ch
from .filters import make_filter
from .physics import click_prob, intensity


class PolicyNet(nn.Module):
    """Feasible-by-construction policy network.

    The network emits v in R^2; the action is
        u = u^- + Du_max * tanh(|v|) * v/|v|
    followed by a radial clip to |u| <= u_max, which is a projection onto the
    amplitude disk and therefore keeps the result inside U(u^-).
    """

    def __init__(self, n_in: int = 8, width: int = 128, depth: int = 3):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers += [nn.Linear(d, 2)]
        self.f = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor, uprev: torch.Tensor, p: Params
                ) -> torch.Tensor:
        v = self.f(feats.to(next(self.parameters()).dtype))
        nrm = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        step = p.dumax * torch.tanh(nrm) * v / nrm
        u = uprev + torch.complex(step[:, 0], step[:, 1]).to(uprev.dtype)
        scale = torch.where(u.abs() > p.umax, p.umax / u.abs().clamp_min(1e-12),
                            torch.ones_like(u.abs()))
        return u * scale.to(u.dtype)


# ---------------------------------------------------------------------------
def collect_dataset(p: Params, n_symbols: int, teacher_Hp: int) -> dict:
    """Roll out the MPC teacher and record (features, action) pairs from the
    states it actually visits."""
    dev = torch.device(p.device)
    alphas = p.alphas(dev)
    data = make_trials(p, n_symbols, seed=p.seed + 777)
    grid = ch.quad_grid(p, p.nh_filter, dev, p.tdtype) \
        if p.csi in ("none", "imperfect", "frozen") else None
    from .simulate import _build_filter

    F, U, PREV = [], [], []
    for a in range(0, n_symbols, p.chunk):
        b = min(a + p.chunk, n_symbols)
        d = data.slice(a, b)
        B = b - a
        h_true = (p.Lp * d.T).sqrt().to(p.tdtype)
        filt = _build_filter(p, d, h_true, alphas, grid)
        ctrl = MPCController(p, alphas, teacher_Hp)
        uprev = torch.full((B,), p.uinit, dtype=p.cdtype, device=dev)
        for k in range(p.K):
            u = ctrl.act(filt, uprev, k)
            F.append(policy_features(filt, uprev, k, p, ctrl.h_point(filt)).cpu())
            U.append(u.cpu())
            PREV.append(uprev.cpu())
            lam = intensity(u, h_true[:, None], p, alphas)
            lam_t = lam[torch.arange(B, device=dev), d.H, 0]
            y = (d.unif[:, k] < click_prob(lam_t)).long()
            filt.update(u, y)
            uprev = u
    return dict(feats=torch.cat(F), act=torch.cat(U), prev=torch.cat(PREV))


def train_policy(p: Params, ds: dict, epochs: int = 40, lr: float = 2e-3,
                 batch: int = 8192, width: int = 128, depth: int = 3,
                 verbose: bool = True) -> PolicyNet:
    dev = torch.device(p.device)
    feats = ds["feats"].to(dev)
    act = ds["act"].to(dev)
    prev = ds["prev"].to(dev)
    net = PolicyNet(n_in=feats.shape[1], width=width, depth=depth).to(dev).to(p.tdtype)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = feats.shape[0]
    scale = max(p.umax, 1e-9)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            u = net(feats[idx], prev[idx], p)
            loss = ((u - act[idx]).abs() / scale).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  distill epoch {ep:3d}  rel. MSE {tot / n:.5f}")
    return net.eval()
