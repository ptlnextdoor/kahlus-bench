"""KAHLUS-BENCH: a leakage-sealed benchmark for neural-coupling and
forecastability claims.

The field has a measurement crisis, not a modeling crisis: every paper reports
a different coupling on a different dataset with a different estimator and a
different leakage protocol, so nobody can say which findings are real, how many
bits, in which direction, or whether a null is a true negative or merely
underpowered. Kahlus-Bench is the CASP analog: synthetic systems with KNOWN
ground-truth directed information at graded strengths, together with a sealed
scoring protocol that reports certified bits + detection floor + direct-vs-
mediated separation. Every method (correlation, PAC, transfer entropy, DINE,
neuroforecast, future foundation models) is scored the same certified way.

The contract this package enforces:

  * A ``Method`` is a callable (series, spec) -> ``EdgeReport``. It never sees
    ground truth. It returns point estimates + lower bounds in bits.

  * A ``Scenario`` is a sampler that knows the true graph and can produce as
    many leakage-sealed draws as the scorer asks for. Crucially, the scenario
    hands the *method* only the observable series, never the simulator's internal
    state -- the only path to ground truth is through the scorer, which the
    method never touches.

  * The ``Scorer`` runs the method on each draw, compares its certified edges
    to the true graph, and reports precision/recall on the certified edge set,
    the detection floor in bits, and -- the keystone metric -- the *null
    certified rate*: how often the method certifies an edge that the scenario
    guarantees carries zero directed information. A method that lights up under
    a pure confounder fails the bench, no matter its recall.

This is the artifact that needs no permission, no lab, no vendor data. It is also
the strongest thing to show Coleman in a 10-minute meeting: a sealed, pre-
registered standard that turns "we found a coupling" into "we certify X bits,
direction i->j, above the calibrated floor."

See ``README.md`` and ``PRE_REGISTRATION.md`` for the full specification.
"""

from kahlus_bench.contract import (
    EdgeReport,
    Method,
    ScorerReport,
    ScenarioSpec,
    SealedDraw,
    TrueGraph,
    ObservableDraw,
    MethodReport,
)
from kahlus_bench.synthetic.ground_truth import (
    LinearGaussianVAR,
    ConfounderVAR,
    MediatedVAR,
)
from kahlus_bench.scoring.sealed import SealedScorer, ScenarioAdapter

__all__ = [
    "EdgeReport",
    "Method",
    "ScorerReport",
    "ScenarioSpec",
    "SealedDraw",
    "TrueGraph",
    "ObservableDraw",
    "MethodReport",
    "LinearGaussianVAR",
    "ConfounderVAR",
    "MediatedVAR",
    "SealedScorer",
    "ScenarioAdapter",
]

__version__ = "0.1.0"
