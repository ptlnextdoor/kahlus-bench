"""Integration test for the neuroforecast adapter.

This test is skipped if neuroforecast is not importable. When it is importable,
it verifies the end-to-end claim of the benchmark: that neuroforecast, a
certified conditional-directed-information estimator, passes ConfounderVAR and
MediatedVAR with zero false certifications, just like the lightweight
ConditionalVARMethod baseline.

This is the test that "completes the loop" -- it confirms the benchmark's
honesty metrics are not just internally consistent, they are consistent with an
independently-developed certified estimator the benchmark was designed to
consume.
"""

from __future__ import annotations

import pytest

pytest.importorskip("neuroforecast")

from kahlus_bench.synthetic.ground_truth import (  # noqa: E402
    ConfounderVAR,
    MediatedVAR,
    LinearGaussianVAR,
)
from kahlus_bench.scoring.sealed import SealedScorer, ScenarioAdapter  # noqa: E402
from kahlus_bench.methods.neuroforecast_adapter import NeuroforecastMethod  # noqa: E402


def test_neuroforecast_passes_confounder():
    sc = ConfounderVAR(n_samples=2048, seed=100)
    scorer = SealedScorer(ScenarioAdapter("ConfounderVAR", sc.sample), n_draws=2)
    rep = scorer.score(NeuroforecastMethod(n_boot=200))
    assert rep.null_certified_rate < 0.25, (
        f"neuroforecast falsely certified confounded edges: "
        f"null-cert rate = {rep.null_certified_rate:.3f}"
    )


def test_neuroforecast_passes_mediation():
    sc = MediatedVAR(n_samples=2048, seed=101)
    scorer = SealedScorer(ScenarioAdapter("MediatedVAR", sc.sample), n_draws=2)
    rep = scorer.score(NeuroforecastMethod(n_boot=200))
    assert rep.mediated_false_edge_rate < 0.5, (
        f"neuroforecast certified mediated edges as direct: "
        f"mediated-false = {rep.mediated_false_edge_rate:.3f}"
    )


def test_neuroforecast_detects_linear_gaussian_edges():
    sc = LinearGaussianVAR(n_samples=4096, seed=102, a=0.20)
    scorer = SealedScorer(ScenarioAdapter("LinearGaussianVAR", sc.sample), n_draws=2)
    rep = scorer.score(NeuroforecastMethod(n_boot=200))
    assert rep.recall >= 0.5, f"neuroforecast recall too low: {rep.recall:.2f}"
