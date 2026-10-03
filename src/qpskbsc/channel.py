"""Gamma-Gamma atmospheric channel: sampling, quadrature grid, CSI posteriors."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import kv, gammaln

from .config import Params


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_T(n: int, p: Params, gen: torch.Generator, device) -> torch.Tensor:
    """Sample the normalised irradiance T = U V with E[T] = 1."""
    # torch has no Gamma sampler on a Generator, so use numpy with a derived seed
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen, device=gen.device).item())
    rng = np.random.default_rng(seed)
    u = rng.gamma(shape=p.k1, scale=1.0 / p.k1, size=n)
    v = rng.gamma(shape=p.k2, scale=1.0 / p.k2, size=n)
    return torch.as_tensor(u * v, dtype=torch.float64, device=device)


def field(T: torch.Tensor, p: Params) -> torch.Tensor:
    """h = sqrt(Lp * T)."""
    return (p.Lp * T).sqrt()


def mean_field(p: Params) -> float:
    """E[h] = sqrt(Lp) E[sqrt(T)], closed form (Eq. 6 of the formulation)."""
    def f(k):
        return math.exp(gammaln(k + 0.5) - gammaln(k)) / math.sqrt(k)
    return math.sqrt(p.Lp) * f(p.k1) * f(p.k2)


# ---------------------------------------------------------------------------
# quadrature grid over T
# ---------------------------------------------------------------------------
def gg_logpdf(t: np.ndarray, p: Params) -> np.ndarray:
    """log f_T(t) for the Gamma-Gamma density (Eq. 4)."""
    k1, k2 = p.k1, p.k2
    a = 0.5 * (k1 + k2)
    z = 2.0 * np.sqrt(k1 * k2 * t)
    # kve = exp(z) * kv  -> use scaled Bessel for numerical range
    from scipy.special import kve
    logk = np.log(np.maximum(kve(k1 - k2, z), 1e-300)) - z
    return (math.log(2.0) + a * math.log(k1 * k2) - gammaln(k1) - gammaln(k2)
            + (a - 1.0) * np.log(t) + logk)


def quad_grid(p: Params, nh: int, device, dtype=torch.float64,
              lo_q: float = 1e-5, hi_q: float = 1 - 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic log-spaced quadrature grid over T.

    Returns ``(h_grid, log_w)`` with ``h_grid`` of shape (nh,) and normalised
    log weights ``log_w`` of shape (nh,).  Integration is performed in
    ``x = ln T`` where the Gamma-Gamma density is well conditioned.
    """
    if nh <= 1:
        h = torch.ones(1, dtype=dtype, device=device)
        return h, torch.zeros(1, dtype=dtype, device=device)
    # bracket the log-irradiance by moment matching, then widen generously
    var_lnT = _var_lnT(p)
    s = math.sqrt(var_lnT)
    xlo, xhi = -6.0 * s - 2.0, 6.0 * s + 2.0
    x = np.linspace(xlo, xhi, nh)
    t = np.exp(x)
    # density in x:  f_X(x) = f_T(e^x) e^x
    logf = gg_logpdf(t, p) + x
    dx = x[1] - x[0]
    logw = logf + math.log(dx)
    logw = logw - _logsumexp(logw)
    h = np.sqrt(p.Lp * t)
    return (torch.as_tensor(h, dtype=dtype, device=device),
            torch.as_tensor(logw, dtype=dtype, device=device))


def _var_lnT(p: Params) -> float:
    from scipy.special import polygamma
    return float(polygamma(1, p.k1) + polygamma(1, p.k2))


def _logsumexp(a: np.ndarray) -> float:
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum()))


# ---------------------------------------------------------------------------
# CSI posterior
# ---------------------------------------------------------------------------
def csi_posterior_logw(xhat: torch.Tensor, p: Params, h_grid: torch.Tensor,
                       logw_prior: torch.Tensor) -> torch.Tensor:
    """Log weights of p(h | hat h) on the quadrature grid.

    The estimator supplies ``xhat = ln T + v``, ``v ~ N(0, sigma_csi^2)``.  The
    posterior is the prior reweighted by the Gaussian likelihood in ln T.

    Parameters
    ----------
    xhat : (B,) tensor of log-irradiance estimates
    h_grid, logw_prior : (Nh,) grid and prior log weights

    Returns
    -------
    (B, Nh) normalised log weights
    """
    lnT = 2.0 * torch.log(h_grid / math.sqrt(p.Lp))  # (Nh,)
    if p.sigma_csi <= 0:
        # degenerate: put all mass on the nearest grid point
        idx = (lnT.unsqueeze(0) - xhat.unsqueeze(1)).abs().argmin(dim=1)
        out = torch.full((xhat.shape[0], h_grid.shape[0]), -1e30,
                         dtype=h_grid.dtype, device=h_grid.device)
        out.scatter_(1, idx.unsqueeze(1), 0.0)
        return out
    ll = -0.5 * (lnT.unsqueeze(0) - xhat.unsqueeze(1)) ** 2 / (p.sigma_csi ** 2)
    lw = logw_prior.unsqueeze(0) + ll
    return lw - torch.logsumexp(lw, dim=1, keepdim=True)


def coarsen(h_grid: torch.Tensor, logw: torch.Tensor, nh_out: int
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample a (B, Nh) log-weight array down to ``nh_out`` points.

    Used to keep the controller's internal channel grid small while the filter
    runs at full resolution.  Points are chosen by equal-probability
    stratification of the marginal weight so that mass, not spacing, is
    preserved.
    """
    Nh = h_grid.shape[0]
    if nh_out >= Nh:
        return h_grid, logw
    w = logw.exp().mean(dim=0) if logw.dim() == 2 else logw.exp()
    cdf = torch.cumsum(w / w.sum(), dim=0)
    targets = (torch.arange(nh_out, device=w.device, dtype=w.dtype) + 0.5) / nh_out
    idx = torch.searchsorted(cdf, targets).clamp(max=Nh - 1)
    idx = torch.unique(idx)
    hg = h_grid[idx]
    lw = logw[..., idx]
    lw = lw - torch.logsumexp(lw, dim=-1, keepdim=True)
    return hg, lw
