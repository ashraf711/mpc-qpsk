"""Projection onto the feasible displacement set.

The feasible set is
    U(u^-) = { u in C : |u| <= u_max , |u - u^-| <= Du_max },
the intersection of two closed disks, hence convex and compact.  Because
``u^-`` itself is always feasible the intersection is never empty.

Projection is computed in closed form.  For convex sets A, B, if the projection
of p onto A already lies in B then it is the projection onto A ∩ B; the
symmetric statement holds for B.  If neither single-set projection is feasible
the projection lies on both boundary circles, i.e. at one of the two
circle-circle intersection points, and we take the nearer one.
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def _proj_disk(p: torch.Tensor, c: torch.Tensor | complex, r) -> torch.Tensor:
    d = (p - c).abs()
    scale = torch.where(d > r, r / d.clamp_min(_EPS), torch.ones_like(d))
    return c + (p - c) * scale.to(p.dtype)


def _circle_intersection(p: torch.Tensor, c: torch.Tensor, R, r) -> torch.Tensor:
    """Nearest point to p on {|z| = R} ∩ {|z - c| = r}."""
    d = c.abs()
    dsafe = d.clamp_min(_EPS)
    a = (dsafe ** 2 + R ** 2 - r ** 2) / (2.0 * dsafe)
    hh = (R ** 2 - a ** 2).clamp_min(0.0).sqrt()
    e = c / dsafe.to(c.dtype)
    base = a.to(c.dtype) * e
    off = hh.to(c.dtype) * e * torch.full_like(e, 1j)
    z1, z2 = base + off, base - off
    pick = (p - z1).abs() <= (p - z2).abs()
    z = torch.where(pick, z1, z2)
    # concentric degenerate case: feasible set is the smaller disk about 0
    rmin = torch.minimum(torch.as_tensor(R, dtype=d.dtype, device=d.device),
                         torch.as_tensor(r, dtype=d.dtype, device=d.device))
    z_conc = _proj_disk(p, torch.zeros_like(c), rmin)
    return torch.where(d > 1e-9, z, z_conc)


def project(p: torch.Tensor, uprev: torch.Tensor, umax: float, dumax: float,
            tol: float = 1e-9) -> torch.Tensor:
    """Euclidean projection of ``p`` onto ``U(uprev)``.

    ``p`` and ``uprev`` are complex tensors broadcastable to a common shape.
    """
    p, uprev = torch.broadcast_tensors(p, uprev)
    zero = torch.zeros_like(uprev)

    feasible = (p.abs() <= umax + tol) & ((p - uprev).abs() <= dumax + tol)

    p1 = _proj_disk(p, zero, umax)          # onto |u| <= umax
    ok1 = (p1 - uprev).abs() <= dumax + tol

    p2 = _proj_disk(p, uprev, dumax)        # onto |u - uprev| <= dumax
    ok2 = p2.abs() <= umax + tol

    z = _circle_intersection(p, uprev, umax, dumax)

    out = torch.where(ok2, p2, z)
    out = torch.where(ok1, p1, out)
    out = torch.where(feasible, p, out)
    return out


def is_feasible(u: torch.Tensor, uprev: torch.Tensor, umax: float, dumax: float,
                tol: float = 1e-6) -> torch.Tensor:
    return (u.abs() <= umax + tol) & ((u - uprev).abs() <= dumax + tol)
