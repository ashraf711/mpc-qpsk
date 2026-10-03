"""Experiment drivers.

Each function returns a pandas DataFrame and writes CSV + JSON into the output
directory.  E1 is the primary result (the actuator-constrained advantage);
everything else supports it.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch

from . import channel as ch
from .config import Params
from .controllers import CyclicNulling, MPCController, MapNulling
from .dp import dp_bracket
from .physics import helstrom_qpsk, heterodyne_qpsk, homodyne_qpsk
from .simulate import (TrialData, error_mask, make_trials, optimize_hnom,
                       paired_diff, run_mc)


# ---------------------------------------------------------------------------
def parse_controllers(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]


def make_factory(p: Params, name: str, hnom: float | None = None, net=None):
    dev = torch.device(p.device)
    alphas = p.alphas(dev)
    if name == "cyclic":
        return lambda: CyclicNulling(p, alphas, hnom)
    if name == "map":
        return lambda: MapNulling(p, alphas)
    if name == "greedy":
        return lambda: MPCController(p, alphas, 1)
    if name.startswith("mpc"):
        H = int(name[3:]) if len(name) > 3 else p.Hp
        return lambda: MPCController(p, alphas, H)
    if name == "distill":
        from .controllers import DistilledController
        return lambda: DistilledController(p, alphas, net)
    raise ValueError(f"unknown controller '{name}'")


def bounds(p: Params, data: TrialData) -> dict:
    """Fading-averaged fundamental limits on the same irradiance sample."""
    neff = (p.Lp * data.T * p.N)
    return dict(
        helstrom=float(helstrom_qpsk(neff).mean().item()),
        helstrom_eta=float(helstrom_qpsk(neff * p.eta).mean().item()),
        heterodyne=float(heterodyne_qpsk(neff).mean().item()),
        heterodyne_eta=float(heterodyne_qpsk(neff * p.eta).mean().item()),
        homodyne=float(homodyne_qpsk(neff).mean().item()),
    )


def _save(df: pd.DataFrame, meta: dict, out: Path, tag: str) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{tag}.csv", index=False)
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[{tag}] wrote {out/f'{tag}.csv'}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
def sweep(p: Params, field: str, values: list, ctrls: list[str], out: Path,
          tag: str, paired_vs: str | None = None) -> pd.DataFrame:
    """Generic sweep of one Params field over ``values`` for several controllers."""
    rows, meta = [], dict(base=p.to_dict(), field=field, values=values,
                          controllers=ctrls)
    for v in values:
        pv = p.replace(**{field: v})
        data = make_trials(pv, pv.nmc)
        bd = bounds(pv, data)
        hnom = pv.hnom if pv.hnom > 0 else None
        if "cyclic" in ctrls and hnom is None:
            hnom = optimize_hnom(pv, data)
        masks = {}
        for c in ctrls:
            t0 = time.perf_counter()
            r = run_mc(pv, make_factory(pv, c, hnom), data)
            if paired_vs:
                masks[c] = error_mask(pv, make_factory(pv, c, hnom), data)
            row = dict(controller=c, **{field: v}, **r, **bd,
                       hnom=hnom if c == "cyclic" else float("nan"),
                       umax=pv.umax, dumax=pv.dumax,
                       wall_s=time.perf_counter() - t0)
            row.pop("belief_mean", None)
            rows.append(row)
            print(f"  {field}={v}  {c:8s}  Pe={r['Pe']:.5g} "
                  f"+-{r['Pe_se']:.2g}  dnull={r['dnull']:.4g}")
        if paired_vs and paired_vs in masks:
            for c, m in masks.items():
                if c == paired_vs:
                    continue
                d = paired_diff(m, masks[paired_vs])
                for row in rows:
                    if row["controller"] == c and row[field] == v:
                        row.update({f"paired_{k}": vv for k, vv in d.items()})
    return _save(pd.DataFrame(rows), meta, out, tag)


# ---------------------------------------------------------------------------
def e1_slew(p: Params, fracs: list[float], ctrls: list[str], out: Path):
    """E1 (primary): P_e versus the actuator slew limit.  Hypothesis H1 predicts
    a gap that is largest for tight slew and vanishes as the limit relaxes."""
    return sweep(p, "dumax_frac", fracs, ctrls, out, "e1_slew", paired_vs="greedy")


def e2_photon(p: Params, Ns: list[float], ctrls: list[str], out: Path):
    """E2: P_e versus mean photon number, against Helstrom and heterodyne."""
    return sweep(p, "N", Ns, ctrls, out, "e2_photon", paired_vs="greedy")


def e3_slices(p: Params, Ks: list[int], ctrls: list[str], out: Path):
    """E3: P_e versus K.  H3 predicts an interior optimum: more slices buy
    adaptivity but accumulate K*lambda_d dark counts, and (with a bandwidth-
    limited modulator) tighten the slew bound."""
    return sweep(p, "K", Ks, ctrls, out, "e3_slices", paired_vs="greedy")


def e4_horizon(p: Params, Hps: list[int], out: Path):
    """E4: P_e versus prediction horizon depth."""
    ctrls = [f"mpc{h}" for h in Hps]
    return sweep(p, "Hp", [p.Hp], ctrls, out, "e4_horizon", paired_vs=ctrls[0])


def e5_policy(p: Params, out: Path, n: int = 4000) -> pd.DataFrame:
    """E5: policy map in the complex plane and distance from nulling actions."""
    dev = torch.device(p.device)
    alphas = p.alphas(dev)
    pv = p.replace(nmc=n)
    data = make_trials(pv, n)
    r = run_mc(pv, make_factory(pv, f"mpc{pv.Hp}"), data, collect_states=True)
    rows = []
    for b, hpt, uprev, k, u in r["states"]:
        nulls = hpt[:, None].to(u.dtype) * alphas.cpu()[None, :] / math.sqrt(pv.K)
        d = (u[:, None] - nulls).abs().min(dim=1)
        rows.append(pd.DataFrame(dict(
            k=k, uR=u.real.numpy(), uI=u.imag.numpy(),
            uprevR=uprev.real.numpy(), uprevI=uprev.imag.numpy(),
            h=hpt.numpy(), dnull=d.values.numpy(),
            null_idx=d.indices.numpy(), bmax=b.max(-1).values.numpy(),
            entropy=(-(b.clamp_min(1e-12) * b.clamp_min(1e-12).log()).sum(-1)).numpy(),
        )))
    df = pd.concat(rows, ignore_index=True)
    return _save(df, dict(base=pv.to_dict()), out, "e5_policy")


def e6_csi(p: Params, sigmas: list[float], modes: list[str], out: Path
           ) -> pd.DataFrame:
    """E6: robustness to imperfect CSI, exact joint filter vs mismatched ones."""
    rows = []
    for s in sigmas:
        for mode in modes:
            pv = p.replace(sigma_csi=s, csi=mode)
            data = make_trials(pv, pv.nmc)
            r = run_mc(pv, make_factory(pv, pv.controller), data)
            r.pop("belief_mean", None)
            rows.append(dict(sigma_csi=s, csi=mode, controller=pv.controller,
                             **r, **bounds(pv, data)))
            print(f"  sigma={s} csi={mode:9s} Pe={r['Pe']:.5g}")
    return _save(pd.DataFrame(rows), dict(base=p.to_dict()), out, "e6_csi")


def e7_bracket(p: Params, out: Path, n_a: int = 6, n_p: int = 12,
               n_h: int = 12, ctrls: list[str] | None = None) -> pd.DataFrame:
    """E7: certified optimality bracket (Proposition 3)."""
    ctrls = ctrls or ["map", "greedy", f"mpc{p.Hp}"]
    br = dp_bracket(p, n_a=n_a, n_p=n_p, n_h=n_h)
    data = make_trials(p, p.nmc)
    rows = []
    for c in ctrls:
        r = run_mc(p, make_factory(p, c), data)
        cost = r["E_phi"] + p.rho_e * r["Eu"] + p.rho_d * r["Su"]
        rows.append(dict(controller=c, closed_loop_cost=cost, Pe=r["Pe"],
                         **br, gap_to_grid=cost - br["V_grid"],
                         certified_gap=cost - br["lower"]))
        print(f"  {c:8s} cost={cost:.5g}  V_grid={br['V_grid']:.5g} "
              f"certified gap<= {cost - br['lower']:.4g}")
    return _save(pd.DataFrame(rows), dict(base=p.to_dict(), bracket=br),
                 out, "e7_bracket")


def e8_distill(p: Params, out: Path, n_train: int = 20000, epochs: int = 40
               ) -> pd.DataFrame:
    """E8: distilled policy -- error probability and latency versus exact MPC."""
    from .distill import collect_dataset, train_policy
    ds = collect_dataset(p, n_train, p.Hp)
    net = train_policy(p, ds, epochs=epochs)
    data = make_trials(p, p.nmc)
    rows = []
    for c, fac in ((f"mpc{p.Hp}", make_factory(p, f"mpc{p.Hp}")),
                   ("distill", make_factory(p, "distill", net=net))):
        r = run_mc(p, fac, data)
        r.pop("belief_mean", None)
        rows.append(dict(controller=c, **r))
        print(f"  {c:8s} Pe={r['Pe']:.5g}  t/stage={r['t_per_stage']*1e6:.1f} us")
    torch.save(net.state_dict(), out / "distilled_policy.pt")
    return _save(pd.DataFrame(rows), dict(base=p.to_dict()), out, "e8_distill")


def e9_effort(p: Params, ctrls: list[str], out: Path) -> pd.DataFrame:
    """E9: performance / effort / complexity table."""
    return sweep(p, "N", [p.N], ctrls, out, "e9_effort", paired_vs="greedy")
