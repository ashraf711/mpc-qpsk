"""Configuration objects.

All physical and control parameters live in :class:`Params`.  Every field is
exposed on the command line so that sweeps can be driven entirely from shell
scripts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any

import torch

# ---------------------------------------------------------------------------
# numerical floors
# ---------------------------------------------------------------------------
LAM_MIN = 1e-12  # floor on Poisson intensity, keeps log(1-exp(-lam)) finite
LOG_MIN = -60.0  # floor on log-probabilities


@dataclass
class Params:
    """Physics, actuator, cost and solver parameters.

    Notation follows the formulation document:
      N       mean transmitted photon number per QPSK symbol
      K       number of equal-energy temporal slices per symbol
      eta     detector quantum efficiency
      xi      signal-LO interference visibility
      dark    mean dark counts per slice (lambda_d)
      Lp      deterministic power gain (path loss)
      k1,k2   Gamma-Gamma shape parameters (kappa_1, kappa_2)
    """

    # ---- signal -----------------------------------------------------------
    N: float = 4.0
    K: int = 4
    phi0: float = math.pi / 4.0

    # ---- detector ---------------------------------------------------------
    eta: float = 0.9
    xi: float = 1.0
    dark: float = 1e-4

    # ---- channel ----------------------------------------------------------
    Lp: float = 1.0
    k1: float = 4.0
    k2: float = 2.0

    # ---- actuator ---------------------------------------------------------
    # u_max = umax_scale * sqrt(N/K) unless umax_abs is given (>0).
    umax_scale: float = 2.0
    umax_abs: float = -1.0
    # Delta u_max = dumax_frac * u_max ; use inf for an unconstrained actuator.
    dumax_frac: float = float("inf")
    uinit: complex = 0.0 + 0.0j

    # ---- stage cost -------------------------------------------------------
    rho_e: float = 0.0
    rho_d: float = 0.0

    # ---- control ----------------------------------------------------------
    Hp: int = 2
    controller: str = "mpc"  # cyclic | map | greedy | mpc | distill

    # ---- CSI --------------------------------------------------------------
    csi: str = "perfect"  # perfect | none | imperfect | ce | frozen
    sigma_csi: float = 0.0
    nh_filter: int = 128  # channel quadrature points used by the filter
    nh_ctrl: int = 16  # channel quadrature points used inside the controller

    # ---- solver -----------------------------------------------------------
    n_starts_amp: int = 4
    n_starts_phase: int = 16
    iters: int = 100
    lr: float = 0.006
    tau0: float = 0.0
    tau1: float = 5e-3
    n_tree_inits: int = 4
    init_jitter: float = 0.5
    polish_rounds: int = 3
    polish_iters: int = 20

    # ---- Monte Carlo ------------------------------------------------------
    nmc: int = 20000
    chunk: int = 4096
    seed: int = 0

    # ---- runtime ----------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"

    # ---- baseline tuning --------------------------------------------------
    hnom: float = -1.0  # <0 -> optimise offline

    # ------------------------------------------------------------------
    @property
    def umax(self) -> float:
        if self.umax_abs > 0:
            return float(self.umax_abs)
        return float(self.umax_scale * math.sqrt(self.N / self.K))

    @property
    def dumax(self) -> float:
        """Slew bound, clipped to 2*u_max (the largest attainable step)."""
        if math.isinf(self.dumax_frac):
            return 2.0 * self.umax
        return float(min(self.dumax_frac * self.umax, 2.0 * self.umax))

    @property
    def tdtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    @property
    def cdtype(self) -> torch.dtype:
        return torch.complex128 if self.dtype == "float64" else torch.complex64

    @property
    def n_starts(self) -> int:
        # polar grid + 4 nulling actions + previous action + origin
        return self.n_starts_amp * self.n_starts_phase + 6

    def alphas(self, device=None) -> torch.Tensor:
        """Transmitted coherent amplitudes alpha_m, shape (4,)."""
        m = torch.arange(4, device=device or self.device, dtype=self.tdtype)
        phi = self.phi0 + m * (math.pi / 2.0)
        amp = math.sqrt(self.N)
        return torch.polar(torch.full_like(phi, amp), phi).to(self.cdtype)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["uinit"] = [float(self.uinit.real), float(self.uinit.imag)]
        d["umax"] = self.umax
        d["dumax"] = self.dumax
        return d

    def json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def replace(self, **kw) -> "Params":
        return dataclasses.replace(self, **kw)


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------
_SKIP = {"uinit"}


def add_param_args(p: argparse.ArgumentParser) -> None:
    """Add one CLI flag per Params field (``--field-name``)."""
    defaults = Params()
    for f in dataclasses.fields(Params):
        if f.name in _SKIP:
            continue
        flag = "--" + f.name.replace("_", "-")
        cur = getattr(defaults, f.name)
        if f.type is bool or isinstance(cur, bool):
            p.add_argument(flag, dest=f.name, type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=None)
        elif isinstance(cur, int) and not isinstance(cur, bool):
            p.add_argument(flag, dest=f.name, type=int, default=None)
        elif isinstance(cur, float):
            p.add_argument(flag, dest=f.name, type=float, default=None)
        else:
            p.add_argument(flag, dest=f.name, type=str, default=None)


def params_from_args(args: argparse.Namespace) -> Params:
    kw = {}
    for f in dataclasses.fields(Params):
        if f.name in _SKIP:
            continue
        v = getattr(args, f.name, None)
        if v is not None:
            kw[f.name] = v
    return Params(**kw)
