"""Pytest entry points.

The substantive checks live in ``qpskbsc.selftest`` because they are numerical
validations of the propositions, not unit tests in the usual sense, and they
are also useful from the command line (``qpskbsc selftest``).
"""

import math

import pytest
import torch

from qpskbsc.config import Params
from qpskbsc.geometry import is_feasible, project
from qpskbsc.physics import helstrom_qpsk, heterodyne_qpsk


def test_selftest_suite_passes():
    from qpskbsc.selftest import run_all
    assert run_all("cpu", quick=True) == 0


def test_helstrom_endpoints():
    z = torch.tensor([0.0], dtype=torch.float64)
    assert abs(helstrom_qpsk(z).item() - 0.75) < 1e-12
    assert abs(heterodyne_qpsk(z).item() - 0.75) < 1e-12


@pytest.mark.parametrize("dumax", [0.05, 0.5, 5.0])
def test_projection_feasible(dumax):
    g = torch.Generator().manual_seed(0)
    uprev = torch.polar(torch.rand(200, generator=g), torch.rand(200, generator=g) * 6.3)
    p = (torch.randn(200, generator=g) + 1j * torch.randn(200, generator=g)) * 3
    q = project(p, uprev, 1.0, dumax)
    assert is_feasible(q, uprev, 1.0, dumax).all()


def test_energy_conservation():
    p = Params(N=7.0, K=5)
    a = p.alphas(torch.device("cpu"))
    assert abs(p.K * (a.abs()[0] ** 2 / p.K).item() - p.N) < 1e-9
