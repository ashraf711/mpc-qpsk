# qpskbsc

Belief-space predictive control of an actuator-constrained adaptive displacement
receiver for QPSK coherent-state discrimination.

Companion simulation code for the ACC formulation. GPU-accelerated (PyTorch),
CLI-driven, `uv`-packaged.

## Install

```bash
cd qpskbsc
uv venv && source .venv/bin/activate
uv pip install -e .
# optional, for the pytest wrapper
uv pip install pytest
```

If you want the CUDA build specifically:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Verify before trusting any number

```bash
qpskbsc selftest                 # 22 checks, ~2 min on CPU
```

These are not cosmetic. Two of them are direct numerical tests of Proposition 2
(the belief martingale): the Monte Carlo mean of `b_k` must stay at `1/4`, and
the empirical `P_e` must equal `E[Phi(b_K)]`. A violation of either means the
reported error probability is not the Bayes risk of the filter actually being
run, which is the failure mode a mismatched channel filter produces. A third
checks the H=2 continuous solver against a fine action grid.

## Quick start

```bash
# single operating point, several controllers
qpskbsc run --N 4 --K 4 --nmc 200000 --controllers map,greedy,mpc2,mpc3

# E1: the primary result -- P_e vs the actuator slew limit
qpskbsc e1 --N 4 --K 4 --nmc 500000 \
    --fracs 0.05,0.1,0.25,0.5,1,inf \
    --controllers map,greedy,mpc2,mpc3 --out results

# full campaign across both GPUs
qpskbsc all --out results --devices cuda:0,cuda:1 --nmc 500000

# figures from existing CSVs
qpskbsc plot --out results
```

Every field of `Params` is a CLI flag (`--rho-e`, `--sigma-csi`, `--nh-filter`,
`--polish-rounds`, ...), so sweeps are shell one-liners.

## Experiments

| cmd | what | hypothesis |
|---|---|---|
| `e1` | `P_e` vs slew limit `Δu_max/u_max` | **H1**, headline |
| `e2` | `P_e` vs photon number `N`, vs Helstrom / heterodyne | positioning |
| `e3` | `P_e` vs slices `K` | **H3**, interior optimum |
| `e4` | `P_e` vs horizon `H_p` | look-ahead depth |
| `e5` | policy map in `C`, `d_null` histogram | **H2**, non-nulling |
| `e6` | `P_e` vs `σ²_CSI`, exact vs mismatched filters | **H4** |
| `e7` | certified optimality bracket (Prop. 3) | certificate |
| `e8` | distilled policy: `P_e` and latency vs MPC | real-time |
| `e9` | effort `E_u`, variation `S_u`, solve time | cost table |

Outputs: `<tag>.csv`, `<tag>_meta.json`, and PDF figures.

## What the code does

- **Exact joint symbol–channel filter** (`filters.py`). The information state is
  a log-weight array over `(hypothesis, channel grid point)`. Perfect CSI is the
  degenerate `Nh=1` case, so one code path covers all CSI regimes. Because the
  channel is fixed across the `K` slices, marginalizing the likelihood
  independently at each stage is a *different* probabilistic model; the
  mismatched variants (`--csi ce`, `--csi frozen`) are retained only as labelled
  ablations.
- **Scenario-tree MPC** (`tree.py`). Exact enumeration of the `2^H` leaves, so
  the objective is deterministic and differentiable. Solved by batched
  multi-start projected Adam.
- **Closed-form projection** onto the intersection of two disks (`geometry.py`).
- **Grid DP + certified bracket** (`dp.py`), Prop. 3.
- **Fundamental limits** (`physics.py`): Helstrom via the circulant Gram matrix,
  and ideal heterodyne.
- **Common random numbers** throughout, so controller differences are paired
  (`paired_diff`, `paired_significant` columns).

## Three bugs worth knowing about

These cost real debugging time and are now covered by regression tests. If you
modify the solver, re-run `selftest`.

1. **Log-sum-exp smoothing of `Phi` is biased toward flat beliefs.** The gap
   `τ·log Σ exp(b_m/τ) − max_m b_m` is largest for uniform `b`, so minimizing
   the smoothed surrogate rewards *uninformative* measurements. More iterations
   made the objective monotonically worse. Smoothing is off by default
   (`--tau0 0`); exact subgradients are better behaved.
2. **Step size must scale with `Δu_max`, not `u_max`.** The local feasible
   region has radius `Δu_max`; a step scaled to `u_max` overshoots so badly that
   no descent step ever improves on the initialization, and the returned value
   is frozen regardless of iteration count.
3. **Multi-starting only the root is not enough.** Deeper nodes inheriting the
   root action stick in local minima worth ~25% excess cost. Each node is now
   initialized at the nulling action for *its own* predicted MAP hypothesis,
   with `--n-tree-inits` jittered replicas.

## Cost

Complexity is `2^H` leaves per solve. `H_p ∈ {1,2,3}` is the practical range;
`H_p=4` is 15 decision nodes and still runs but is slow. The DP bracket (`e7`)
grows as `(2G)^K` and is meant for `K ≤ 3`.

`--chunk` trades memory for speed; reduce it under `--csi none` where the
controller carries an `Nh`-point channel grid (`--nh-ctrl`).

## Reproducing the headline claim

```bash
qpskbsc e1 --N 4 --K 4 --nmc 500000 --fracs 0.05,0.1,0.25,0.5,1,inf \
    --controllers map,greedy,mpc2,mpc3 --devices cuda:0,cuda:1 --out results
qpskbsc plot --out results
```

Read `results/e1_slew.csv`: `paired_diff` is `P_e(controller) − P_e(greedy)` on
paired samples, `paired_significant` is the 95% test, and `act_slew` is the
fraction of stages at which the slew constraint is active. H1 predicts the gap
is significant exactly where `act_slew` is large and vanishes as
`Δu_max/u_max → ∞`.

**Run this first, before writing any manuscript text.** In the unconstrained
regime MPC may be statistically indistinguishable from greedy; that is a fine
outcome for a paper framed around the constrained regime and a fatal one for a
paper framed around "prediction helps" generically.
