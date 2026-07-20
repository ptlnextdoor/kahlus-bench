# Kahlus-Bench pre-registration

> Version 0.1.0 — locked at first commit. The point of a benchmark is that its
> rules do not move to flatter a method, including ours.

This document freezes: the estimand, the scenarios, the honesty metrics, the
scoring protocol, the certification threshold, and the floor metadata. Any
future change is a new version of the benchmark with a new version number, and
the prior version's scorecard remains reproducible from the git history.

## 1. Estimand

The benchmark's estimand is the **conditional directed information** per
directed edge, in bits:

$$ \mathrm{DI}(i \to j \,\|\, \text{rest}) \;=\; I\big(X_j(t) \,;\, X_i(\text{past}) \,\big|\, X_j(\text{past}),\, X_{k\ne i,j}(\text{past})\big) $$

This is the Quinn-Kiyavash-Coleman (2015, IEEE TIT) estimand that neuroforecast
already implements, and the named next step the Rao et al. stomach-brain paper
explicitly asks for in its own limitations. Every method scored by this
benchmark is scored against this estimand, regardless of what the method itself
was originally designed to estimate.

A method that estimates *pairwise* correlation, pairwise transfer entropy, or
phase-amplitude coupling is, by the terms of this benchmark, *also* being scored
on its ability to recover $\mathrm{DI}(i\to j\,\|\,\text{rest})$ — and the
pre-registered result is that pairwise methods fail under confounding and
mediation. That is the headline finding.

## 2. Scenarios

Three scenarios, all linear-Gaussian VAR(1) systems whose conditional
directed information per edge is **analytically known** from the transition
matrix via the closed form in `kahlus_bench.synthetic.ground_truth._analytic_cdi_var1`.

  1. **`LinearGaussianVAR`** — a three-stream chain with two true direct edges,
     graded coupling strengths $a \in \{0.05, 0.15, 0.30\}$. The analytic truth
     is the support of the transition matrix.
  2. **`ConfounderVAR`** — a hidden common driver $C$ feeds $X$ and $Y$. The
     true direct graph is $\{C\to X, C\to Y\}$ only. The pairwise correlation
     $X\!\leftrightarrow\!Y$ is large and positive (true to the synthetic
     design), and that is the trap.
  3. **`MediatedVAR`** — the chain $X \to Z \to Y$. The pairwise link $X\to Y$
     is nonzero, but the conditional $\mathrm{DI}(X\to Y\,\|\,\text{rest})$ is
     exactly zero because conditioning on $Z$ mediates it away.

A scenario is a *controlled source*. It emits leakage-sealed draws carrying
the observable series plus a held-private `TrueGraph` with the direct,
mediated, and null edge sets and the per-edge true directed information.
Methods see only the observable. The `SealedScorer` is the only object that
reads `TrueGraph`.

## 3. Honesty metrics (pre-registered definitions)

A method **certifies** a directed edge $i\to j$ iff its 95% lower confidence
bound exceeds the certification threshold **and** its point estimate is
strictly positive:

$$ \text{certified}(i, j) \iff \mathrm{LCB}_{95}(i\to j) > \tau \;\;\text{and}\;\; \widehat{\mathrm{DI}}(i\to j) > 0, $$

where $\tau = 10^{-3}$ bits is the pre-registered **noise floor**. The noise floor
is not a free parameter; it is the smallest magnitude at which a bootstrap
lower bound on a true-null edge is empirically distinguishable from finite-
sample noise (rates of $10^{-6}$–$10^{-4}$ bits), calibrated against the
bootstrap distribution of `ConditionalVARMethod` on `ConfounderVAR` null edges.

Metric definitions are then:

  * **`precision`** — $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$, where TP is a
    certified edge in `direct_edges` and FP is a certified edge in `null_edges`
    or a certified direct edge in `mediated_edges`.
  * **`recall`** — $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$, where FN is a true
    direct edge the method failed to certify.
  * **`null_certified_rate`** (the r=0.97 metric) — fraction of true-zero edges
    falsely certified. The pre-registered expectation: pairwise methods score
    high, the conditional estimator scores zero (within finite-sample noise).
  * **`mediated_false_edge_rate`** — fraction of mediated edges the method
    certified **as direct** (where `direct_fraction > 0.5` in its `EdgeReport`,
    or where it has no opinion). The pre-registered expectation: pairwise
    methods score 1.0, the conditional estimator scores 0.
  * **`floor_bias_bits`** — the method's mean claimed detection floor minus the
    trusted analytic detection floor. Positive means the method over-claims
    its own power.

## 4. Trusted detection floor (metadata, not gate)

The **trusted detection floor** is the smallest *true* directed information
whose method LCB cleared the certification threshold. It is reported
alongside the scorecard as calibration metadata. It is **not** used as a
per-method certification gate; raising the certification threshold to the
trusted floor would erase the difference between methods. The benchmark exists
to expose which methods wrongly certify at the honest threshold, not to
benchmark how high a method can lift its own threshold.

## 5. Sealing

The benchmark's leakage control is structural, not a convention:

  * A scenario emits `SealedDraw` objects; the `SealedScorer` is the only
    object that imports and reads `SealedDraw.truth`.
  * Methods only ever hold `ObservableDraw`. The `Method` protocol accepts
    `ObservableDraw` and returns `MethodReport`; there is no parameter for
    truth.
  * A method that imports `kahlus_bench.scoring.sealed`, the `SealedDraw`
    dataclass, or its `truth` field is a benchmark violation caught at code
    review. There is no runtime guard — the sealing is a discipline the
    community enforces when submitting to a leaderboard, in the same spirit as
    CASP.

## 6. Frozen at this version

Anything not above (additional scenarios, additional methods, additional
metrics) is a future version of the benchmark, not part of v0.1.0. The
scorecard produced from the git tree at the first commit of this file is the
v0.1.0 scorecard and remains reproducible for as long as the git history lives.

## 7. What this benchmark does *not* claim

It does not claim:

  1. that the conditional VAR baseline is the *best* estimator — only that it
     passes the assays the pairwise baselines provably fail;
  2. that the three VAR(1) scenarios span the space of neuro-physiology — only
     that they expose the three generic failure modes the literature has been
     silently suffering (under-detection, false-certification under
     confounding, and false-direct certification under mediation);
  3. that synthetic benchmarks settle method choice — only that they prevent a
     method from claiming what it cannot certify.

The benchmark's first honest result is that it cannot — by construction —
certify a method "best." It can certify a method honest. That is the
contribution.
