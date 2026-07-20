"""Subpackage: baseline methods for Kahlus-Bench.

Each method implements the ``Method`` protocol: a pure callable that takes an
``ObservableDraw`` and returns a ``MethodReport`` of per-edge certified
estimates in bits. Methods NEVER receive ground truth.

The baselines are intentionally the field's usual suspects, so the benchmark
demonstrates the precise failure modes the literature reproduces:

  * ``CorrelationMethod`` -- pairwise Pearson correlation with a bootstrap LCB
    in bits. Cannot separate direct from mediated; cannot condition on the
    rest. Expected to fail ConfounderVAR (false certify) and MediatedVAR
    (mediated-false-edge).
  * ``PairwiseTransferEntropyMethod`` -- pairwise lagged transfer entropy with
    a bootstrap LCB. Same expected failures, sharper on VAR data.
  * ``PhaseAmplitudeCouplingMethod`` -- PAC between phase of a slow band and
    power of a fast band, with a permutation LCB. A stand-in for the
    correlational analyses the Rao et al. stomach-brain paper explicitly flags
    as "precluding inference about directionality."
  * ``ConditionalVARMethod`` -- a conditional linear-Gaussian VAR fit with a
    cluster-bootstrap LCB, the direct-vs-mediated-aware baseline. This is the
    honest, lightweight proxy for neuroforecast's estimator.

Eventually neuroforecast itself drops in here as a method and is scored the
same way, which is the whole thesis: the benchmark treats everyone identically
and the certified estimator should be the one that passes all three scenarios.
"""

from kahlus_bench.methods.baselines import (
    CorrelationMethod,
    PairwiseTransferEntropyMethod,
    PhaseAmplitudeCouplingMethod,
    ConditionalVARMethod,
)

__all__ = [
    "CorrelationMethod",
    "PairwiseTransferEntropyMethod",
    "PhaseAmplitudeCouplingMethod",
    "ConditionalVARMethod",
]
