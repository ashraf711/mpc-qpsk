"""Command-line interface.

Examples
--------
  qpskbsc selftest
  qpskbsc run --controller mpc --Hp 2 --N 4 --K 4 --nmc 200000
  qpskbsc e1 --fracs 0.05,0.1,0.25,0.5,1,inf --controllers map,greedy,mpc2,mpc3
  qpskbsc all --out results --devices cuda:0,cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from .config import Params, add_param_args, params_from_args


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# multi-GPU sharding: sweep points are distributed across devices
# ---------------------------------------------------------------------------
def _worker(rank: int, devices: list[str], pdict: dict, field: str,
            values: list, ctrls: list[str], out: str, tag: str, paired: str | None):
    import pandas as pd
    from .experiments import sweep
    p = Params(**{k: v for k, v in pdict.items()
                  if k in Params.__dataclass_fields__ and k != "uinit"})
    p = p.replace(device=devices[rank])
    mine = values[rank::len(devices)]
    if not mine:
        return
    df = sweep(p, field, mine, ctrls, Path(out), f"{tag}_part{rank}", paired)
    del df


def run_sharded(p: Params, field: str, values: list, ctrls: list[str],
                out: Path, tag: str, devices: list[str], paired: str | None):
    import pandas as pd
    import torch.multiprocessing as mp
    if len(devices) <= 1:
        from .experiments import sweep
        return sweep(p.replace(device=devices[0]), field, values, ctrls, out,
                     tag, paired)
    mp.spawn(_worker, args=(devices, p.to_dict(), field, values, ctrls,
                            str(out), tag, paired),
             nprocs=len(devices), join=True)
    parts = sorted(out.glob(f"{tag}_part*.csv"))
    df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    df = df.sort_values([field, "controller"])
    df.to_csv(out / f"{tag}.csv", index=False)
    for f in parts:
        f.unlink()
    print(f"[{tag}] merged {len(parts)} shards -> {out/f'{tag}.csv'}")
    return df


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qpskbsc",
                                 description="Belief-space predictive control of an "
                                             "adaptive displacement receiver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def base(name, help_):
        q = sub.add_parser(name, help=help_)
        add_param_args(q)
        q.add_argument("--out", type=str, default="results")
        q.add_argument("--devices", type=str, default=None,
                       help="comma-separated torch devices for sweep sharding")
        return q

    q = sub.add_parser("selftest", help="run numerical self-tests")
    q.add_argument("--device", type=str, default="cpu")
    q.add_argument("--quick", action="store_true")

    q = base("run", "single operating point, one controller")
    q.add_argument("--controllers", type=str, default=None,
                   help="override with a comma-separated list")

    q = base("e1", "P_e vs slew limit (primary result)")
    q.add_argument("--fracs", type=str, default="0.05,0.1,0.25,0.5,1,inf")
    q.add_argument("--controllers", type=str, default="map,greedy,mpc2,mpc3")

    q = base("e2", "P_e vs mean photon number")
    q.add_argument("--Ns", type=str, default="0.5,1,2,4,8,12,16,20")
    q.add_argument("--controllers", type=str, default="cyclic,map,greedy,mpc2,mpc3")

    q = base("e3", "P_e vs number of slices K")
    q.add_argument("--Ks", type=str, default="2,4,6,8,12,16")
    q.add_argument("--controllers", type=str, default="map,greedy,mpc2")

    q = base("e4", "P_e vs prediction horizon")
    q.add_argument("--Hps", type=str, default="1,2,3")

    q = base("e5", "policy map and d_null")
    q.add_argument("--n", type=int, default=4000)

    q = base("e6", "imperfect-CSI robustness")
    q.add_argument("--sigmas", type=str, default="0,0.05,0.1,0.2,0.4")
    q.add_argument("--modes", type=str, default="perfect,imperfect,ce,frozen,none")

    q = base("e7", "certified optimality bracket")
    q.add_argument("--dp-na", type=int, default=6)
    q.add_argument("--dp-np", type=int, default=12)
    q.add_argument("--dp-nh", type=int, default=12)
    q.add_argument("--controllers", type=str, default="map,greedy,mpc2")

    q = base("e8", "policy distillation")
    q.add_argument("--n-train", type=int, default=20000)
    q.add_argument("--epochs", type=int, default=40)

    q = base("e9", "effort and complexity table")
    q.add_argument("--controllers", type=str, default="cyclic,map,greedy,mpc2,mpc3")

    q = base("all", "run the full campaign")
    q.add_argument("--controllers", type=str, default="cyclic,map,greedy,mpc2,mpc3")
    q.add_argument("--fracs", type=str, default="0.05,0.1,0.25,0.5,1,inf")
    q.add_argument("--Ns", type=str, default="0.5,1,2,4,8,12,16,20")
    q.add_argument("--Ks", type=str, default="2,4,6,8,12,16")

    q = sub.add_parser("plot", help="render figures from result CSVs")
    q.add_argument("--out", type=str, default="results")

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        from .selftest import run_all
        return run_all(args.device, args.quick)

    if args.cmd == "plot":
        return _do_plots(Path(args.out))

    p = params_from_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    devices = _strs(args.devices) if args.devices else [p.device]
    (out / "params.json").write_text(p.json())
    print(p.json())

    from . import experiments as ex

    if args.cmd == "run":
        ctrls = _strs(args.controllers) if args.controllers else [p.controller]
        ex.sweep(p, "N", [p.N], ctrls, out, "run", paired_vs="greedy"
                 if "greedy" in ctrls else None)
        return 0

    if args.cmd == "e1":
        run_sharded(p, "dumax_frac", _floats(args.fracs), _strs(args.controllers),
                    out, "e1_slew", devices, "greedy")
        return 0
    if args.cmd == "e2":
        run_sharded(p, "N", _floats(args.Ns), _strs(args.controllers),
                    out, "e2_photon", devices, "greedy")
        return 0
    if args.cmd == "e3":
        run_sharded(p, "K", _ints(args.Ks), _strs(args.controllers),
                    out, "e3_slices", devices, "greedy")
        return 0
    if args.cmd == "e4":
        ex.e4_horizon(p, _ints(args.Hps), out)
        return 0
    if args.cmd == "e5":
        ex.e5_policy(p, out, args.n)
        return 0
    if args.cmd == "e6":
        ex.e6_csi(p, _floats(args.sigmas), _strs(args.modes), out)
        return 0
    if args.cmd == "e7":
        ex.e7_bracket(p, out, args.dp_na, args.dp_np, args.dp_nh,
                      _strs(args.controllers))
        return 0
    if args.cmd == "e8":
        ex.e8_distill(p, out, args.n_train, args.epochs)
        return 0
    if args.cmd == "e9":
        ex.e9_effort(p, _strs(args.controllers), out)
        return 0

    if args.cmd == "all":
        ctrls = _strs(args.controllers)
        print("\n=== E1: P_e vs slew limit (primary) ===")
        run_sharded(p, "dumax_frac", _floats(args.fracs),
                    [c for c in ctrls if c != "cyclic"], out, "e1_slew",
                    devices, "greedy")
        print("\n=== E2: P_e vs N ===")
        run_sharded(p, "N", _floats(args.Ns), ctrls, out, "e2_photon",
                    devices, "greedy")
        print("\n=== E3: P_e vs K ===")
        run_sharded(p, "K", _ints(args.Ks), [c for c in ctrls if c != "cyclic"],
                    out, "e3_slices", devices, "greedy")
        print("\n=== E5: policy map ===")
        ex.e5_policy(p, out)
        print("\n=== E6: CSI robustness ===")
        ex.e6_csi(p, [0.0, 0.05, 0.1, 0.2, 0.4],
                  ["perfect", "imperfect", "ce", "frozen", "none"], out)
        print("\n=== E7: certified bracket ===")
        ex.e7_bracket(p.replace(K=min(p.K, 3)), out)
        print("\n=== E8: distillation ===")
        ex.e8_distill(p, out)
        print("\n=== E9: effort table ===")
        ex.e9_effort(p, ctrls, out)
        _do_plots(out)
        return 0

    ap.error(f"unhandled command {args.cmd}")
    return 2


def _do_plots(out: Path) -> int:
    from . import plots
    jobs = [
        ("e1_slew.csv", "dumax_frac", "e1_slew.pdf",
         r"slew limit $\Delta u_{\max}/u_{\max}$", True),
        ("e2_photon.csv", "N", "e2_photon.pdf",
         r"mean photon number $N$", False),
        ("e3_slices.csv", "K", "e3_slices.pdf",
         r"measurement slices $K$", False),
    ]
    for csv, xf, fig, xl, logx in jobs:
        f = out / csv
        if f.exists():
            plots.curve(f, xf, out / fig, xl, logx=logx)
    if (out / "e1_slew.csv").exists():
        plots.gap_plot(out / "e1_slew.csv", out / "e1_gap.pdf")
    if (out / "e5_policy.csv").exists():
        plots.policy_map(out / "e5_policy.csv", out / "e5_policy.pdf")
        plots.dnull_plot(out / "e5_policy.csv", out / "e5_dnull.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
