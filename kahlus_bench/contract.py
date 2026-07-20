"""The sealed contract: Method, Scenario, EdgeReport, ScorerReport.

This module is the keystone of the benchmark. Everything else is plumbing.
The contract is deliberately small and deliberately rigid:

  * A ``Method`` is a *black box* that sees only observable time series and a
    ``ScenarioSpec`` describing sample sizes and stream counts. It NEVER sees
    ground truth, never sees the scenario's internal state, and never sees the
    scorer. It returns an ``EdgeReport`` of point estimates + lower bounds in
    bits per directed edge.

  * A ``Scenario`` is a *controlled source* that knows the true graph and can
    emit as many leakage-sealed draws as requested. It returns an
    ``ObservableDraw`` carrying (series, clusters, true_graph, true_bits). The
    method receives ``series`` and ``clusters`` only; the truth is held by the
    scorer.

  * Leakage control is structural, not a convention: ``Scenario.sample`` returns
    its draws, and the ``SealedScorer`` is the only object that reads
    ``true_graph`` from a draw. A method that imports the scenario, the scorer,
    or any ground-truth field is a benchmark violation caught at review time
    (and discouraged by the module layout: methods have no import path to
    ``scoring.sealed`` internals).

The honesty metrics are encoded in ``ScorerReport``:

  * ``precision`` / ``recall`` on the *certified* edge set (edges whose lower
    bound clears the detection floor), not the point estimates.

  * ``null_certified_rate`` -- the keystone. Of edges the scenario guarantees
    carry ZERO directed information, what fraction did the method falsely
    certify? A method with high recall and a nontrivial null-certified rate is
    *the r=0.97 failure mode*, named and quantified.

  * ``mediated_false_edge_rate`` -- the second keystone. Of edges that exist in
    the pairwise sense but are *mediated* (zero after conditioning on the rest),
    what fraction did the method still certify as direct? This is the metric a
    pure correlation / PAC / pairwise-transfer-entropy estimator is expected to
    fail, which is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# What a scenario hands the scorer (truth side) and what it hands a method
# (observable side). These are deliberately separate types so a method that
# accidentally asks for truth gets a static-error nudge if type-checked.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    """Describes an observable draw to a method. Contains no ground truth."""

    n_samples: int            # length of each series (per draw)
    n_streams: int            # number of simultaneously recorded streams
    lag: int                  # past-window length the method may use
    cluster_blocks: tuple[int, ...]  # subject/segment boundaries for bootstrap
    sample_seed: int          # so a method may fold it into any internals


@dataclass
class ObservableDraw:
    """What a method is allowed to touch."""

    series: np.ndarray        # (n_samples, n_streams) float64
    clusters: np.ndarray     # (n_samples,) int -- block id per timestep
    spec: ScenarioSpec


@dataclass
class TrueGraph:
    """Ground truth held only by the scenario and the scorer."""

    direct_edges: frozenset[tuple[int, int]]   # i -> j direct DI > 0
    mediated_edges: frozenset[tuple[int, int]] # pairwise nonzero, true DI = 0
    null_edges: frozenset[tuple[int, int]]     # pairwise zero AND true DI = 0
    true_bits: dict[tuple[int, int], float]    # true directed info per edge

    @property
    def all_edges(self) -> list[tuple[int, int]]:
        n = 0
        for e in self.direct_edges | self.mediated_edges | self.null_edges:
            n = max(n, max(e))
        return [(i, j) for i in range(n + 1) for j in range(n + 1) if i != j]


@dataclass
class SealedDraw:
    """Internal: scorer-facing wrapper a scenario returns. Methods never see
    the ``truth`` field."""

    observable: ObservableDraw
    truth: TrueGraph


# ---------------------------------------------------------------------------
# What a method returns, and the method protocol.
# ---------------------------------------------------------------------------


@dataclass
class EdgeReport:
    """A method's per-edge certified claim.

    All quantities are in bits. Edges not reported are treated as null with
    lcb 0 by the scorer.

    Attributes:
        cdi_bits:  point estimate of directed info i -> j, conditioned on rest.
        lcb95_bits: 95% lower confidence bound (bootstrap or otherwise).
        direct_fraction: in [0,1] -- method's belief this edge is direct vs
            mediated. 1.0 = claims fully direct, 0.0 = claims fully mediated.
            Methods that cannot separate direct from mediated should report 1.0
            and accept the mediated_false_edge_rate penalty.
    """

    cdi_bits: float
    lcb95_bits: float
    direct_fraction: float = 1.0


@dataclass
class MethodReport:
    """A method's full output over one draw: per-edge reports + a detection
    floor the method claims, in bits. The scorer may override the floor with
    the scenario's calibrated floor if it trusts the scenario more (it does)."""

    edges: dict[tuple[int, int], EdgeReport] = field(default_factory=dict)
    claimed_floor_bits: float = 0.0


class Method(Protocol):
    """The sealed method interface. Implement as a callable or a class with
    ``__call__``. Pure function: same (draw, spec) -> same MethodReport."""

    name: str

    def __call__(self, draw: ObservableDraw) -> MethodReport:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Scorer output.
# ---------------------------------------------------------------------------


@dataclass
class ScorerReport:
    method_name: str
    scenario_name: str
    n_draws: int

    # Certified-edge precision and recall. "Certified" = lcb95 clears the
    # detection floor.recall = certified true-direct / all true-direct.
    precision: float
    recall: float

    # Keystone honesty metrics, all in [0,1].
    null_certified_rate: float       # false-certify rate on true-zero edges
    mediated_false_edge_rate: float  # false-direct rate on mediated edges

    # Calibration: how close is the method's claimed floor to the scenario's
    # known calibrated floor? Positive = method over-claims power (bad).
    floor_bias_bits: float

    # The raw confusion counts over all draws, for audit.
    tp: int
    fp: int
    fn: int
    mediated_certified: int
    null_certified: int

    def summary(self) -> str:
        return (
            f"{self.method_name} on {self.scenario_name} "
            f"(n={self.n_draws}): "
            f"P={self.precision:.2f} R={self.recall:.2f} | "
            f"null-cert={self.null_certified_rate:.3f} "
            f"mediated-false={self.mediated_false_edge_rate:.3f} | "
            f"floor-bias={self.floor_bias_bits:+.4f} bits"
        )


# Type aliases for the scenario and scorer callables ---------------------------------
ScenarioSampler = Callable[[int], SealedDraw]
