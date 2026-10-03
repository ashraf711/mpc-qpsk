"""Exact joint symbol-channel Bayesian filter.

The information state is the joint posterior over (H, h) carried as a
log-weight array of shape (B, 4, Nh).  Perfect CSI is the degenerate case
Nh = 1 with the true channel value on the grid; the no-CSI and imperfect-CSI
receivers use a genuine quadrature grid.  Because the channel is *fixed* across
the K slices (block fading), marginalising the likelihood independently at each
stage would be a different -- and incorrect -- probabilistic model; this filter
avoids that by keeping h in the state.
"""

from __future__ import annotations

import torch

from .config import Params
from .physics import intensity, log_kernel, view_h


class JointFilter:
    """Batched joint posterior over (hypothesis, channel)."""

    def __init__(self, logw0: torch.Tensor, h_grid: torch.Tensor, p: Params,
                 alphas: torch.Tensor):
        """
        Parameters
        ----------
        logw0  : (B, 4, Nh) normalised initial joint log weights
        h_grid : (B, Nh) channel support (may differ per trial under perfect CSI)
        """
        self.logw = logw0
        self.h = h_grid
        self.p = p
        self.alphas = alphas

    # -- accessors ---------------------------------------------------------
    @property
    def belief(self) -> torch.Tensor:
        """Symbol belief b_k, shape (B, 4)."""
        return torch.logsumexp(self.logw, dim=-1).exp()

    @property
    def h_mean(self) -> torch.Tensor:
        """Posterior mean channel E[h | I_k], shape (B,)."""
        w = self.logw.exp().sum(dim=1)            # (B, Nh)
        return (w * self.h).sum(-1) / w.sum(-1).clamp_min(1e-30)

    @property
    def h_post_logw(self) -> torch.Tensor:
        """Marginal channel posterior log weights, shape (B, Nh)."""
        lw = torch.logsumexp(self.logw, dim=1)
        return lw - torch.logsumexp(lw, dim=-1, keepdim=True)

    # -- update ------------------------------------------------------------
    def update(self, u: torch.Tensor, y: torch.Tensor) -> None:
        """Incorporate observation ``y`` (0/1 int tensor) taken under ``u``."""
        lam = intensity(u, self.h, self.p, self.alphas)      # (B, 4, Nh)
        lp0, lp1 = log_kernel(lam)
        lp = torch.where(y[:, None, None].bool(), lp1, lp0)
        lw = self.logw + lp
        z = torch.logsumexp(lw.flatten(-2), dim=-1)
        self.logw = lw - z[:, None, None]


# ---------------------------------------------------------------------------
def make_filter(h_true: torch.Tensor, p: Params, alphas: torch.Tensor,
                h_grid: torch.Tensor | None = None,
                logw_h: torch.Tensor | None = None) -> JointFilter:
    """Build the filter appropriate to ``p.csi``.

    perfect            : Dirac at the true channel (Nh = 1)
    none / imperfect   : quadrature grid with prior / CSI-posterior weights
    ce                 : Dirac at the *estimated* channel (certainty equivalent,
                         a deliberately mismatched filter kept as an ablation)
    """
    B = h_true.shape[0]
    dtype, dev = h_true.dtype, h_true.device
    if h_grid is None:
        hg = h_true[:, None]                                  # (B, 1)
        lw = torch.zeros(B, 1, dtype=dtype, device=dev)
    else:
        hg = h_grid[None, :].expand(B, -1) if h_grid.dim() == 1 else h_grid
        lw = logw_h
    logw0 = lw[:, None, :].expand(B, 4, hg.shape[1]).clone() - torch.log(
        torch.tensor(4.0, dtype=dtype, device=dev))
    return JointFilter(logw0, hg, p, alphas)
