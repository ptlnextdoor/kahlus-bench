"""The sealed scorer.

The ``SealedScorer`` is the only object in the benchmark that reads ground
truth. Every method runs against the scorer, the scorer compares the method's
certified edges to the scenario's true graph, and the scorer emits a
``ScorerReport`` with precision/recall on the certified set, the null-certified
rate (the r=0.97 metric), the mediated-false-edge rate (the direct-vs-mediated
metric), and the floor-bias in bits.

A trusted-floor option uses the scenario's analytic ground truth to override a
method's claimed detection floor where the scenario knows better. This is a
benchmark convenience, not a method convenience: the floor the scorer reports
is the one a method would have to beat on real data, which is what the field
needs but never reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from kahlus_bench.contract import (
    EdgeReport,
    Method,
    MethodReport,
    ObservableDraw,
    ScorerReport,
    SealedDraw,
)


# A scenario is anything with .sample() -> SealedDraw and a .name attribute.
@dataclass
class ScenarioAdapter:
    name: str
    sample: Callable[[], SealedDraw]


class SealedScorer:
    """Run a method over many draws from one scenario and score its certified
    edge set against truth.

    Args:
        scenario: an object exposing ``sample() -> SealedDraw`` and a ``name``.
        n_draws: how many independent draws to score over.
        floor_mode: ``"trusted"`` uses the scenario's known per-edge truth to
            set the detection floor at the smallest true DI the method's LCB
            actually cleared across draws; ``"claimed"`` uses the method's own
            claimed floor. Default ``"trusted"`` because the field needs a
            calibrated floor not a marketing floor.
    """

    def __init__(
        self,
        scenario: ScenarioAdapter,
        n_draws: int = 16,
        floor_mode: str = "trusted",
        floor_floor: float = 1e-3,
    ) -> None:
        self.scenario = scenario
        self.n_draws = n_draws
        self.floor_mode = floor_mode
        # The certification EDGE threshold defaults to a small positive noise
        # floor rather than exactly zero. The honest motivation: any finite-
        # sample bootstrap estimator returns *tiny* positive lower bounds on
        # edges whose true directed information is exactly zero (typically on
        # the order of 1e-6 to 1e-4 bits), simply because the lower quantile
        # of a finite-sample bootstrap distribution is rarely exactly zero.
        # An entirely-zero threshold would then treat bootstrap noise as a
        # certification. The 1e-3 bit floor is the smallest magnitude a method
        # must clear to count as "claiming" an edge; methods whose LCB exceeds
        # this floor are responsible for the claim. The detection floor
        # (separate) is the smallest *true DI* whose LCB clears this threshold,
        # reported as calibration metadata.
        self.certify_threshold_bits = float(floor_floor)

    # -- detection floor (metadata): smallest nonzero true DI cleared ----

    def _detection_floor(self, draw_reports: list[tuple[SealedDraw, MethodReport]]) -> float:
        """The smallest true DI the method successfully certified. metadata."""
        cleared: list[float] = []
        for draw, report in draw_reports:
            for (i, j), edge in report.edges.items():
                if (i, j) in draw.truth.direct_edges:
                    true = draw.truth.true_bits.get((i, j), 0.0)
                    if true > 0 and edge.lcb95_bits > self.certify_threshold_bits:
                        cleared.append(true)
        return min(cleared) if cleared else float("inf")

    # -- one method over all draws ------------------------------------------

    def score(self, method: Method) -> ScorerReport:
        draws_reports: list[tuple[SealedDraw, MethodReport]] = []
        for _ in range(self.n_draws):
            draw = self.scenario.sample()
            report = method(draw.observable)
            draws_reports.append((draw, report))

        # The detection-floor metadata is reported alongside the threshold.
        # We do NOT use it as the certification gate; certification runs at
        # the fixed threshold (default 0).
        if self.floor_mode == "trusted":
            floor = self._detection_floor(draws_reports)
        else:
            floor = float(min((r.claimed_floor_bits for _, r in draws_reports), default=0.0))

        threshold = self.certify_threshold_bits

        # Counters across all draws and all candidate edges.
        tp = fp = fn = 0
        null_certified = 0
        mediated_certified = 0
        null_total = 0
        mediated_total = 0

        for draw, report in draws_reports:
            truth = draw.truth
            certified = set()
            for (i, j), edge in report.edges.items():
                if edge.lcb95_bits > threshold and edge.cdi_bits > 0:
                    certified.add((i, j))

            for edge in truth.direct_edges:
                if edge in certified:
                    tp += 1
                else:
                    fn += 1
            for edge in truth.null_edges:
                null_total += 1
                if edge in certified:
                    null_certified += 1
                    fp += 1
            for edge in truth.mediated_edges:
                mediated_total += 1
                cert_edge = report.edges.get(edge)
                if edge in certified and _claims_direct(cert_edge):
                    mediated_certified += 1
                    fp += 1
                elif edge in certified and not _claims_direct(cert_edge):
                    pass  # method labeled mediated; do NOT count as fp-direct.

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        null_certified_rate = (
            null_certified / null_total if null_total else 0.0
        )
        mediated_false_edge_rate = (
            mediated_certified / mediated_total if mediated_total else 0.0
        )

        # Floor bias: does the method's claimed floor agree with the trusted one?
        claimed_floor = float(np.mean([
            r.claimed_floor_bits for _, r in draws_reports
        ]))
        floor_bias = claimed_floor - (floor if floor != float("inf") else 0.0)

        return ScorerReport(
            method_name=getattr(method, "name", method.__class__.__name__),
            scenario_name=self.scenario.name,
            n_draws=self.n_draws,
            precision=float(precision),
            recall=float(recall),
            null_certified_rate=float(null_certified_rate),
            mediated_false_edge_rate=float(mediated_false_edge_rate),
            floor_bias_bits=float(floor_bias),
            tp=tp, fp=fp, fn=fn,
            mediated_certified=mediated_certified,
            null_certified=null_certified,
        )


def _claims_direct(edge: EdgeReport | None) -> bool:
    """A method claims the edge is direct if its direct_fraction > 0.5, or if
    it has no opinion (None / no report) -- conservative: assume it claims
    direct, accept the penalty."""
    if edge is None:
        return True
    return edge.direct_fraction > 0.5
