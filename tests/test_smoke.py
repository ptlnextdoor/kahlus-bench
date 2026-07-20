"""Kahlus-Bench smoke tests.

These tests are the *gate the benchmark itself must pass* before it can be
trusted to score anything else. They are deliberately written so the expected
honesty outcomes are encoded as assertions: a pairwise method MUST false-certify
under ConfounderVAR (the r=0.97 trap), and a conditional estimator MUST NOT.
If any of these flip, something in the sealed scorer or scenario is broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from kahlus_bench.synthetic.ground_truth import (
    ConfounderVAR,
    LinearGaussianVAR,
    MediatedVAR,
)
from kahlus_bench.scoring.sealed import SealedScorer, ScenarioAdapter
from kahlus_bench.methods.baselines import (
    ConditionalVARMethod,
    CorrelationMethod,
    PairwiseTransferEntropyMethod,
)


# ---- Sanity: scenarios emit stable series and analytic truth --------------


def test_linear_gaussian_var_produces_known_truth():
    sc = LinearGaussianVAR(n_samples=2048, seed=1, a=0.20)
    draw = sc.sample()
    assert draw.observable.series.shape == (2048, 3)
    assert np.isfinite(draw.observable.series).all()
    # Two true direct edges, with strictly positive true DI.
    assert (1, 0) in draw.truth.direct_edges
    assert (2, 1) in draw.truth.direct_edges
    assert draw.truth.true_bits[(1, 0)] > 0
    assert draw.truth.true_bits[(2, 1)] > 0
    # The non-edge (2, 0) must have zero or near-zero analytic DI.
    assert draw.truth.true_bits.get((2, 0), 0.0) < 1e-6


def test_confounder_var_truth_has_zero_xy_direct_edge():
    sc = ConfounderVAR(n_samples=2048, seed=2)
    draw = sc.sample()
    # X<->Y must be a null direct edge after conditioning on C.
    assert (1, 2) in draw.truth.null_edges
    assert (2, 1) in draw.truth.null_edges
    assert draw.truth.true_bits.get((1, 2), 0.0) < 1e-6
    # Pairwise correlation X<->Y must be large -- that's the trap.
    rho = np.corrcoef(draw.observable.series[:, 1], draw.observable.series[:, 2])[0, 1]
    assert abs(rho) > 0.3, "confounder scenario should produce sizable X<->Y correlation"


def test_mediated_var_truth_has_xy_mediated():
    sc = MediatedVAR(n_samples=2048, seed=3)
    draw = sc.sample()
    assert (0, 2) in draw.truth.mediated_edges
    assert draw.truth.true_bits.get((0, 2), 0.0) < 1e-6
    # pairwise corr X->Y (lag-1) must be nonzero -- that's the trap.
    rho = np.corrcoef(draw.observable.series[:-1, 0], draw.observable.series[1:, 2])[0, 1]
    assert abs(rho) > 0.1, "mediated scenario should produce nonzero pairwise X->Y"


# ---- The honesty axis: pairwise methods fail confounder, conditional passes


@pytest.mark.parametrize("MethodClass", [
    ConditionalVARMethod,
])
def test_conditional_var_passes_confounder(MethodClass):
    sc = ConfounderVAR(n_samples=2048, seed=10)
    scorer = SealedScorer(ScenarioAdapter("ConfounderVAR", sc.sample), n_draws=4)
    method = MethodClass()
    rep = scorer.score(method)
    assert rep.null_certified_rate < 0.25, (
        f"{method.name} falsely certified confounded edges: "
        f"null-cert rate = {rep.null_certified_rate:.3f}"
    )


@pytest.mark.parametrize("MethodClass", [
    CorrelationMethod,
    PairwiseTransferEntropyMethod,
])
def test_pairwise_methods_fail_confounder(MethodClass):
    """The r=0.97 trap. A pure pairwise estimator MUST false-certify at least
    one X<->Y edge under ConfounderVAR. If this test fails, the benchmark no
    longer reproduces the documented failure mode -- something is wrong."""
    sc = ConfounderVAR(n_samples=2048, seed=11)
    scorer = SealedScorer(ScenarioAdapter("ConfounderVAR", sc.sample), n_draws=4)
    method = MethodClass()
    rep = scorer.score(method)
    assert rep.null_certified_rate > 0.0, (
        f"{method.name} did NOT reproduce the r=0.97 trap under ConfounderVAR; "
        f"expected a false positive on the X<->Y null edge."
    )


@pytest.mark.parametrize("MethodClass", [
    CorrelationMethod,
    PairwiseTransferEntropyMethod,
])
def test_pairwise_methods_fail_mediation(MethodClass):
    """A pure pairwise estimator MUST false-certify the mediated X->Y edge
    as direct under MediatedVAR. Confirms the mediated-false-edge metric is
    meaningfully nonzero for pairwise methods."""
    sc = MediatedVAR(n_samples=2048, seed=12)
    scorer = SealedScorer(ScenarioAdapter("MediatedVAR", sc.sample), n_draws=4)
    method = MethodClass()
    rep = scorer.score(method)
    assert rep.mediated_false_edge_rate > 0.0, (
        f"{method.name} did NOT reproduce the mediated X->Y false-cert; "
        f"expected mediated-false-edge > 0."
    )


def test_conditional_var_passes_mediation():
    """The conditional VAR estimator conditions on the rest and so must NOT
    certify the mediated X->Y edge as direct."""
    sc = MediatedVAR(n_samples=2048, seed=13)
    scorer = SealedScorer(ScenarioAdapter("MediatedVAR", sc.sample), n_draws=4)
    method = ConditionalVARMethod()
    rep = scorer.score(method)
    assert rep.mediated_false_edge_rate < 0.5, (
        f"ConditionalVAR certified mediated edges as direct: "
        f"mediated-false = {rep.mediated_false_edge_rate:.3f}"
    )


# ---- Detection: conditional VAR recovers the true edges in the clean scenario


def test_conditional_var_detects_linear_gaussian_edges():
    sc = LinearGaussianVAR(n_samples=4096, seed=20, a=0.25)
    scorer = SealedScorer(ScenarioAdapter("LinearGaussianVAR", sc.sample), n_draws=4)
    method = ConditionalVARMethod()
    rep = scorer.score(method)
    assert rep.recall >= 0.5, f"ConditionalVAR recall too low: {rep.recall:.2f}"
    assert rep.precision >= 0.5, f"ConditionalVAR precision too low: {rep.precision:.2f}"
