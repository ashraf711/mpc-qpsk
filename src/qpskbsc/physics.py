"""Photon-counting physics, Bayes risk, and fundamental limits.

Implements, in order:
  * the displaced Poisson intensity  lambda_m(h,u)          [Eq. (9) of the formulation]
  * the on-off log-likelihood kernel                        [Eq. (11)]
  * the terminal Bayes risk Phi and its log-sum-exp smoothing [Eqs. (13),(39)]
  * the Helstrom / square-root-measurement limit            [Eq. (35)]
  * the ideal heterodyne (standard quantum) limit           [Eq. (36)]
"""

from __future__ import annotations

import math

import torch

from .config import LAM_MIN, LOG_MIN, Params

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# intensity and kernel
# ---------------------------------------------------------------------------
def view_h(h: torch.Tensor, u_ndim: int) -> torch.Tensor:
    """Reshape a channel grid ``h`` of shape (B, Nh) for broadcasting against a
    control tensor with ``u_ndim`` dimensions whose leading dimension is B.

    Returns shape (B, 1, ..., 1, Nh) with ``u_ndim - 1`` singleton axes.
    """
    B, Nh = h.shape
    return h.reshape(B, *([1] * (u_ndim - 1)), Nh)


def intensity(u: torch.Tensor, h: torch.Tensor, p: Params,
              alphas: torch.Tensor | None = None) -> torch.Tensor:
    """Poisson intensity lambda_m(h,u).

    Parameters
    ----------
    u : complex tensor, shape (*B,)
    h : real tensor,  shape (*B_h, Nh) broadcastable with (*B, 1)
    alphas : complex tensor (4,)

    Returns
    -------
    real tensor of shape (*B, 4, Nh)
    """
    if alphas is None:
        alphas = p.alphas(u.device)
    # r_m = h * alpha_m / sqrt(K)   ->  (..., 4, Nh)
    r = h.unsqueeze(-2).to(alphas.dtype) * (alphas.unsqueeze(-1) / math.sqrt(p.K))
    uu = u.unsqueeze(-1).unsqueeze(-1)
    nbar = r.real.pow(2) + r.imag.pow(2) + uu.real.pow(2) + uu.imag.pow(2) \
        - 2.0 * p.xi * (r * uu.conj()).real
    lam = p.eta * nbar + p.dark
    return lam.clamp_min(LAM_MIN)


def log_kernel(lam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (log p(Y=0|.), log p(Y=1|.)) for an on-off detector.

    ``log(1 - exp(-lam))`` is evaluated as ``log(-expm1(-lam))`` which is
    accurate for small lam (the dark-count dominated regime).
    """
    lp0 = -lam
    lp1 = torch.log((-torch.expm1(-lam)).clamp_min(1e-300)).clamp_min(LOG_MIN)
    return lp0, lp1


def click_prob(lam: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(-lam)


# ---------------------------------------------------------------------------
# Bayes risk
# ---------------------------------------------------------------------------
def phi_exact(b: torch.Tensor) -> torch.Tensor:
    """Terminal Bayes risk Phi(b) = 1 - max_m b^m.  ``b`` has shape (*B, 4)."""
    return 1.0 - b.max(dim=-1).values


def phi_smooth(b: torch.Tensor, tau: float) -> torch.Tensor:
    """Log-sum-exp smoothing Phi_tau(b) = 1 - tau*log sum_m exp(b^m/tau).

    Satisfies  Phi(b) - tau*log 4  <=  Phi_tau(b)  <=  Phi(b).
    """
    if tau <= 0:
        return phi_exact(b)
    return 1.0 - tau * torch.logsumexp(b / tau, dim=-1)


def marginal_belief(logw: torch.Tensor) -> torch.Tensor:
    """Symbol belief from a joint log-weight array of shape (*B, 4, Nh)."""
    return torch.logsumexp(logw, dim=-1).exp()


def normalize_logw(logw: torch.Tensor) -> torch.Tensor:
    """Normalise a joint log-weight array over its last two axes (4, Nh)."""
    z = torch.logsumexp(logw.flatten(-2), dim=-1)
    return logw - z.unsqueeze(-1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# fundamental limits
# ---------------------------------------------------------------------------
def helstrom_qpsk(neff: torch.Tensor) -> torch.Tensor:
    """Minimum error probability for QPSK coherent states (square-root
    measurement, optimal for geometrically uniform sets).

    ``neff`` is the received mean photon number |h|^2 N.  Returns P_e.
    """
    neff = neff.to(torch.float64)
    # Gram first row  c_d = exp(-neff (1 - e^{j d pi/2}))
    d = torch.arange(4, device=neff.device, dtype=torch.float64)
    ang = d * (math.pi / 2.0)
    expo_re = -neff.unsqueeze(-1) * (1.0 - torch.cos(ang))
    expo_im = neff.unsqueeze(-1) * torch.sin(ang)
    c = torch.polar(torch.exp(expo_re), expo_im)  # (*, 4) complex128
    # eigenvalues of the circulant Gram matrix
    q = torch.arange(4, device=neff.device, dtype=torch.float64)
    W = torch.polar(torch.ones(4, 4, dtype=torch.float64, device=neff.device),
                    -2.0 * math.pi * q.unsqueeze(1) * d.unsqueeze(0) / 4.0)
    lam = (c.unsqueeze(-2) * W).sum(-1).real.clamp_min(0.0)  # (*, 4)
    thr = 1e-14 * lam.max(dim=-1, keepdim=True).values
    lam = torch.where(lam < thr, torch.zeros_like(lam), lam)
    pc = lam.sqrt().sum(-1).pow(2) / 16.0
    return (1.0 - pc).clamp(0.0, 1.0)


def _qfunc(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(x / SQRT2)


def heterodyne_qpsk(neff: torch.Tensor) -> torch.Tensor:
    """Ideal heterodyne (standard quantum limit) symbol-error probability."""
    neff = neff.to(torch.float64).clamp_min(0.0)
    q = _qfunc(neff.sqrt())
    return (1.0 - (1.0 - q).pow(2)).clamp(0.0, 1.0)


def homodyne_qpsk(neff: torch.Tensor) -> torch.Tensor:
    """Single-quadrature homodyne on QPSK.

    Only one quadrature is measured, so the two symbols sharing its sign remain
    confusable and the receiver guesses between them: with per-quadrature mean
    +/- sqrt(neff/2) and variance 1/2, the sign is correct with probability
    1 - Q(sqrt(neff)), and the residual binary choice is a coin flip, giving
    P_e = 1 - (1/2)(1 - Q(sqrt(neff))).  Reported only as a weak reference.
    """
    neff = neff.to(torch.float64).clamp_min(0.0)
    q = _qfunc(neff.sqrt())
    return (1.0 - 0.5 * (1.0 - q)).clamp(0.0, 1.0)
