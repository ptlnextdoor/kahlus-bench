"""Subpackage: synthetic ground-truth generators for Kahlus-Bench."""

from kahlus_bench.synthetic.ground_truth import (
    LinearGaussianVAR,
    ConfounderVAR,
    MediatedVAR,
    default_scenarios,
)

__all__ = [
    "LinearGaussianVAR",
    "ConfounderVAR",
    "MediatedVAR",
    "default_scenarios",
]
