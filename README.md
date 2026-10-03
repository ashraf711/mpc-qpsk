# qpskbsc

**Belief-Space Predictive Control for Actuator-Constrained Adaptive Quantum Receivers**

Simulation code for adaptive coherent-state QPSK discrimination using sequential Bayesian inference, controlled optical displacement, photon counting, channel-state information (CSI), and finite-horizon predictive control.

The repository accompanies ongoing research on **belief-space model predictive control (MPC) for adaptive quantum optical receivers**, with particular emphasis on practical actuator limitations and free-space optical fading.

The central question is:

> How should a quantum receiver choose its next optical displacement when the transmitted QPSK symbol is unknown, the receiver learns sequentially from photon-counting measurements, and the displacement actuator cannot move arbitrarily fast?

The implementation is written in Python/PyTorch, supports GPU acceleration, provides command-line experiment drivers, and includes numerical validation tests for the Bayesian filter and optimization routines.

---

## Research problem

The transmitted alphabet consists of four coherent states,

\[
\mathcal{H}
=
\left\{
|\alpha_0\rangle,
|\alpha_1\rangle,
|\alpha_2\rangle,
|\alpha_3\rangle
\right\},
\]

corresponding to QPSK symbols

\[
\alpha_m
=
\sqrt{N}
e^{j(\phi_0+m\pi/2)},
\qquad
m\in\{0,1,2,3\},
\]

where \(N\) is the mean photon number per symbol.

Each received symbol is divided into \(K\) sequential measurement slices. At stage \(k\), the receiver applies a complex displacement

\[
v_k \in \mathbb{C},
\]

performs photon-counting detection, updates its posterior belief over the four hypotheses, and chooses the next displacement.

The information state is the Bayesian belief vector

\[
\mathbf b_k
=
\begin{bmatrix}
b_k^0 &
b_k^1 &
b_k^2 &
b_k^3
\end{bmatrix}^{T},
\]

with

\[
b_k^m
=
P(H=m\mid\mathcal I_k).
\]

The receiver therefore operates as a closed-loop system:

\[
\text{belief}
\rightarrow
\text{displacement}
\rightarrow
\text{photon count}
\rightarrow
\text{Bayesian update}
\rightarrow
\text{next displacement}.
\]

---

## Main idea

Conventional adaptive displacement receivers commonly choose the next action by nulling the symbol with the largest current posterior probability.

This repository investigates a more general approach.

Instead of selecting the displacement greedily, the receiver solves a finite-horizon stochastic control problem and chooses the complex displacement that minimizes predicted terminal Bayes risk while respecting actuator constraints.

Conceptually,

\[
v_k^\star
=
\arg\min_{v_k}
\mathbb E
\left[
1-\max_m b_{K}^{m}
\mid
\mathbf b_k
\right],
\]

subject to constraints such as

\[
|v_k|\leq v_{\max},
\]

and

\[
|v_k-v_{k-1}|
\leq
\Delta v_{\max}.
\]

The second constraint models finite actuator bandwidth or slew rate.

The predictive controller can therefore deliberately choose a displacement that is not optimal for the immediate measurement if doing so improves future information acquisition or leaves the actuator in a more favorable position for subsequent stages.

---

## Receiver model

For hypothesis \(H=m\), channel coefficient \(h\), and displacement \(v_k\), the photon-counting observation model is based on the post-displacement mean photon number

\[
\lambda_m(h,v_k)
=
\eta
\left(
\frac{|h\alpha_m|^2}{K}
+
|v_k|^2
-
2\xi
\operatorname{Re}
\left\{
\frac{h\alpha_m}{\sqrt K}v_k^\ast
\right\}
\right)
+
\lambda_d,
\]

where:

- \(N\): mean photons per transmitted symbol,
- \(K\): number of adaptive measurement slices,
- \(\eta\): detector efficiency,
- \(\xi\): interference visibility,
- \(\lambda_d\): dark-count contribution,
- \(h\): channel coefficient,
- \(v_k\): controlled complex displacement.

For an on/off single-photon detector (SPD),

\[
P(Y_k=0\mid H=m,h,v_k)
=
e^{-\lambda_m(h,v_k)},
\]

and

\[
P(Y_k=1\mid H=m,h,v_k)
=
1-e^{-\lambda_m(h,v_k)}.
\]

The resulting observations are incorporated sequentially through Bayesian inference.

---

## Bayesian inference

For perfect CSI, the posterior update is

\[
b_{k+1}^{m}
=
\frac{
b_k^m
P(Y_k\mid H=m,h,v_k)
}{
\sum_j
b_k^j
P(Y_k\mid H=j,h,v_k)
}.
\]

For uncertain CSI, the implementation can maintain a **joint posterior over symbol hypothesis and channel state**.

The information state is represented numerically by log weights over

\[
(H,h),
\]

rather than independently marginalizing the channel likelihood at every measurement stage.

This distinction is important because the atmospheric channel is assumed fixed during the \(K\) measurements of one symbol.

---

## Channel-state information

The receiver supports several CSI regimes.

### Perfect CSI

The receiver knows the instantaneous channel coefficient and conditions its Bayesian update and displacement decisions on that value.

### Imperfect CSI

The receiver observes a noisy channel estimate. A log-domain model can be used,

\[
x=\ln T,
\qquad
\hat x=x+w,
\qquad
w\sim\mathcal N(0,\sigma_v^2),
\]

where \(T\) is the channel intensity gain.

The receiver then maintains uncertainty over the channel together with uncertainty over the transmitted symbol.

### No CSI

The receiver operates using only the statistical channel model and sequential photon-count observations.

### Mismatched CSI baselines

Certainty-equivalent and frozen-posterior approximations are retained as comparison cases.

These are useful for quantifying the benefit of correctly propagating channel uncertainty through the sequential Bayesian filter.

---

## Atmospheric fading

The free-space channel can be modeled using Gamma-Gamma atmospheric turbulence,

\[
T=UV,
\]

with

\[
U\sim
\Gamma(\kappa_1,1/\kappa_1),
\qquad
V\sim
\Gamma(\kappa_2,1/\kappa_2).
\]

The complex received amplitude is scaled according to the channel realization before adaptive displacement and detection.

This allows comparison among:

- perfect CSI,
- imperfect CSI,
- no CSI,
- exact joint filtering,
- mismatched CSI processing.

---

## Controllers

Several receiver policies are implemented for comparison.

### Cyclic nulling

The receiver follows a predetermined sequence of QPSK nulling actions without adapting the displacement to the posterior belief.

### MAP nulling

At stage \(k\),

\[
\hat m_k
=
\arg\max_m b_k^m,
\]

and the displacement attempts to null the currently most probable hypothesis.

This is a natural Bayesian adaptive baseline.

### Greedy continuous control

The displacement is optimized continuously but only for the next measurement step.

This corresponds to predictive horizon

\[
K_p=1.
\]

It distinguishes the effect of **continuous displacement optimization** from the effect of **multi-step prediction**.

### Predictive MPC

For

\[
K_p>1,
\]

the receiver evaluates future observation branches using a scenario tree and optimizes the current displacement with respect to predicted terminal Bayes risk.

The primary comparisons use

\[
K_p\in\{1,2,3\}.
\]

### Distilled policy

A lower-complexity policy can be trained or constructed from MPC-generated behavior for applications where full online optimization is too expensive.

---

## Scenario-tree optimization

For binary on/off observations and predictive horizon \(H\), the controller considers

\[
2^H
\]

possible observation sequences.

The scenario tree propagates:

1. the current Bayesian belief,
2. each candidate displacement,
3. both possible photon-count outcomes,
4. the corresponding posterior beliefs,
5. future displacement decisions.

The resulting optimization problem is deterministic once the complete observation tree is enumerated.

Continuous displacements are optimized numerically using batched multi-start projected optimization.

---

## Actuator constraints

The receiver explicitly models practical displacement hardware.

Two constraints are particularly important.

### Amplitude constraint

\[
|v_k|
\leq
v_{\max}.
\]

### Slew constraint

\[
|v_k-v_{k-1}|
\leq
\Delta v_{\max}.
\]

The normalized slew ratio

\[
\frac{\Delta v_{\max}}{v_{\max}}
\]

is used to characterize actuator bandwidth.

A small ratio corresponds to a strongly bandwidth-limited receiver, whereas

\[
\Delta v_{\max}\rightarrow\infty
\]

approaches an unconstrained displacement actuator.

One of the main objectives of the project is to determine when predictive control provides an advantage specifically because consecutive control actions are coupled through this constraint.

---

## Terminal decision and Bayes risk

After the final measurement stage, the receiver performs a MAP decision,

\[
\hat m
=
\arg\max_m b_K^m.
\]

Under equal misclassification costs, the conditional Bayes error is

\[
\Phi(\mathbf b_K)
=
1-\max_m b_K^m.
\]

The average symbol-error probability is therefore

\[
P_e
=
\mathbb E
\left[
\Phi(\mathbf b_K)
\right].
\]

This quantity is the primary performance metric throughout the repository.

---

## Performance metrics

The experiments evaluate several complementary metrics:

- symbol-error probability \(P_e\),
- relative reduction in \(P_e\),
- paired controller differences,
- actuator slew activity,
- displacement energy,
- displacement variation,
- optimization latency,
- performance versus photon number \(N\),
- performance versus number of measurement slices \(K\),
- predictive-horizon sensitivity,
- CSI robustness,
- distance from conventional nulling actions,
- optimality-gap certificates.

Fundamental or conventional reference curves include:

- Helstrom discrimination limits where applicable,
- ideal heterodyne detection,
- MAP nulling,
- greedy continuous displacement control.

---

## Repository structure

```text
qpskbsc/
├── src/
│   └── qpskbsc/
│       ├── filters.py
│       ├── tree.py
│       ├── geometry.py
│       ├── dp.py
│       ├── physics.py
│       └── ...
├── tests/
├── README.md
└── pyproject.toml
```

Important modules include:

- **`filters.py`** — Bayesian symbol/channel filtering,
- **`tree.py`** — finite-horizon scenario-tree MPC,
- **`geometry.py`** — projection onto actuator-feasible sets,
- **`dp.py`** — grid dynamic programming and optimality certificates,
- **`physics.py`** — coherent-state receiver models and fundamental benchmarks.

---

## Installation

The project uses `uv` for environment and package management.

```bash
cd qpskbsc

uv venv
source .venv/bin/activate

uv pip install -e .
```

For the optional pytest wrapper:

```bash
uv pip install pytest
```

For a CUDA-enabled PyTorch installation:

```bash
uv pip install torch \
    --index-url https://download.pytorch.org/whl/cu124
```

---

## Verification

Before using simulation results, run:

```bash
qpskbsc selftest
```

The validation suite checks numerical properties of both the probabilistic model and the optimization routines.

Important tests include the Bayesian belief martingale property,

\[
\mathbb E[b_k^m]
=
P(H=m),
\]

which equals \(1/4\) under a uniform QPSK prior, and the identity

\[
P_e
=
\mathbb E[\Phi(\mathbf b_K)].
\]

These provide strong checks that the posterior being propagated by the implemented filter is consistent with the Monte Carlo error probability.

The validation suite also compares the short-horizon continuous optimizer against a fine action grid.

---

## Quick start

Run several controllers at one operating point:

```bash
qpskbsc run \
    --N 4 \
    --K 4 \
    --nmc 200000 \
    --controllers map,greedy,mpc2,mpc3
```

Run the primary actuator-slew experiment:

```bash
qpskbsc e1 \
    --N 4 \
    --K 4 \
    --nmc 500000 \
    --fracs 0.05,0.1,0.25,0.5,1,inf \
    --controllers map,greedy,mpc2,mpc3 \
    --out results
```

Run the campaign across two GPUs:

```bash
qpskbsc all \
    --out results \
    --devices cuda:0,cuda:1 \
    --nmc 500000
```

Generate figures from existing results:

```bash
qpskbsc plot --out results
```

Most fields of the simulation parameter structure are exposed as command-line options, allowing additional parameter sweeps to be scripted directly.

Examples include:

```text
--rho-e
--sigma-csi
--nh-filter
--polish-rounds
--n-tree-inits
--chunk
```

---

## Experiments

| Command | Experiment | Purpose |
|---|---|---|
| `e1` | \(P_e\) versus normalized actuator slew limit | quantify the value of prediction under actuator coupling |
| `e2` | \(P_e\) versus mean photon number \(N\) | compare receiver performance with Helstrom and heterodyne references |
| `e3` | \(P_e\) versus measurement slices \(K\) | investigate the tradeoff between additional measurements and actuator limitations |
| `e4` | \(P_e\) versus predictive horizon \(K_p\) | determine how much look-ahead is useful |
| `e5` | displacement-policy maps and nulling distance | determine whether optimized actions depart from conventional nulling |
| `e6` | \(P_e\) versus CSI uncertainty | compare exact joint filtering with mismatched CSI processing |
| `e7` | finite-grid dynamic-programming comparison | construct an optimality bracket |
| `e8` | distilled policy versus online MPC | evaluate accuracy/latency tradeoffs |
| `e9` | actuator effort and computational cost | quantify practical controller requirements |

Typical experiment outputs include

```text
<tag>.csv
<tag>_meta.json
```

together with generated publication figures where requested.

---

## Primary slew experiment

The primary experiment studies

\[
P_e
\quad\text{versus}\quad
\frac{\Delta v_{\max}}{v_{\max}}.
\]

Run:

```bash
qpskbsc e1 \
    --N 4 \
    --K 4 \
    --nmc 500000 \
    --fracs 0.05,0.1,0.25,0.5,1,inf \
    --controllers map,greedy,mpc2,mpc3 \
    --devices cuda:0,cuda:1 \
    --out results
```

The central comparison separates two effects:

1. **continuous displacement optimization**, measured by comparing greedy continuous control with MAP nulling;
2. **predictive control**, measured by comparing \(K_p>1\) MPC with the greedy \(K_p=1\) controller.

The expected benefit of prediction is strongest when the slew constraint substantially couples consecutive displacement decisions.

As the actuator approaches the unconstrained regime, predictive and greedy policies may become statistically indistinguishable. This is an important result rather than a failure: it identifies the physical regime in which look-ahead provides value.

---

## Paired Monte Carlo comparisons

The simulations use common random numbers when comparing controllers.

For two controllers evaluated on the same Monte Carlo samples, the paired quantity

```text
paired_diff
```

represents

\[
P_e^{(\mathrm{controller})}
-
P_e^{(\mathrm{reference})}.
\]

Paired comparisons reduce Monte Carlo uncertainty in controller differences and make it easier to determine whether relatively small performance changes are statistically meaningful.

The result files additionally record significance information where appropriate.

---

## CSI robustness experiment

The CSI study compares:

- perfect CSI,
- exact Bayesian filtering with imperfect CSI,
- certainty-equivalent CSI,
- frozen CSI posterior approximations,
- no CSI.

For uncertain channels, maintaining the joint symbol-channel posterior is important because the same channel realization persists across the sequential measurements of one transmitted symbol.

Consequently,

\[
P(H,h\mid Y_{0:k})
\]

cannot in general be replaced by independently averaging the channel likelihood at each stage without changing the underlying probabilistic model.

---

## Policy interpretation

The optimized receiver is not restricted to conventional nulling actions.

Policy-map experiments examine where the optimized displacement lies in the complex plane and compute its distance from the nearest conventional nulling action,

\[
d_k^{\mathrm{null}}.
\]

This helps determine whether performance improvements arise from simply choosing a different symbol to null or from genuinely non-nulling continuous displacement decisions.

---

## Dynamic-programming validation

For short horizons, a discretized action-space dynamic program is used as an additional reference.

The comparison provides a numerical bracket between the continuous closed-loop controller and a sufficiently fine discrete approximation.

Because the discrete search grows rapidly with horizon and action-grid resolution, this validation is intended primarily for small \(K\).

---

## Computational complexity

For binary on/off detection, a predictive horizon \(H\) contains

\[
2^H
\]

observation leaves.

The number of decision nodes grows as

\[
2^H-1.
\]

Consequently,

\[
K_p\in\{1,2,3\}
\]

is the main practical operating range investigated in the simulations.

Larger horizons are possible but become increasingly expensive.

The channel-uncertainty case is also more demanding because every belief carries a numerical distribution over the channel grid.

The `--chunk` option can be reduced when GPU memory becomes limiting.

---

## Numerical implementation notes

Several implementation details were found to be important for reliable optimization.

### Exact terminal risk

The terminal objective uses

\[
\Phi(\mathbf b)
=
1-\max_m b^m
\]

directly.

A log-sum-exp approximation to the maximum can distort the optimization because the smoothing error is largest for nearly uniform beliefs, inadvertently favoring less informative posterior distributions.

### Slew-aware optimization scale

The local feasible region is determined by

\[
\Delta v_{\max},
\]

not only by the global displacement bound \(v_{\max}\).

Optimization step sizes therefore need to respect the local actuator-feasible scale, particularly in strongly slew-limited regimes.

### Multi-start scenario-tree initialization

Initializing only the root action is insufficient for deeper predictive horizons.

Tree nodes are initialized using the predicted MAP hypothesis at the corresponding belief state, with multiple perturbed initializations available through the multi-start solver.

These cases are covered by regression tests.

---

## Reproducibility recommendations

Before using a result in a manuscript:

1. Run `qpskbsc selftest`.
2. Record the full simulation parameter configuration.
3. Use sufficiently large Monte Carlo sample counts.
4. Report confidence intervals or paired uncertainty for controller comparisons.
5. Compare predictive MPC against the greedy continuous controller, not only against conventional MAP nulling.
6. Check the unconstrained-actuator regime separately from the slew-limited regime.
7. Verify CSI experiments using the probabilistically correct joint filter before interpreting mismatched-filter baselines.

---

## Research scope

The repository is intended to study the interaction among:

- quantum coherent-state discrimination,
- sequential Bayesian inference,
- adaptive optical displacement,
- photon-counting measurements,
- atmospheric channel uncertainty,
- receiver CSI,
- actuator amplitude and slew constraints,
- finite-horizon stochastic control,
- belief-space MPC,
- dynamic programming,
- computational complexity.

The broader objective is to develop quantum optical receivers whose adaptive measurement policies account explicitly for both **information acquisition** and **realistic hardware constraints**.

---

## Status

This repository is research code associated with ongoing work on actuator-constrained Bayesian adaptive quantum receivers.

Results, APIs, experiment definitions, and numerical settings may evolve as the formulation and manuscript are refined.
