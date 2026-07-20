"""Kahlus-Bench CLI: run a one-line scorecard per method per scenario.

Usage:
    python -m kahlus_bench.run [--n-draws 4] [--n-samples 4096]

Produces a stdout scorecard where each row is one (method, scenario) pair and
each cell is the honesty metric the literature never reports: precision and
recall on the *certified* edge set, the null-certified rate (the r=0.97 metric),
the mediated-false-edge rate (the direct-vs-mediated metric), and the floor
bias in bits.

This is the artifact that needs no permission, no lab, no vendor data: the
sealed standard you show Coleman in 10 minutes.
"""

from __future__ import annotations

import argparse
import sys

from kahlus_bench.methods.baselines import (
    ConditionalVARMethod,
    CorrelationMethod,
    PairwiseTransferEntropyMethod,
    PhaseAmplitudeCouplingMethod,
)
from kahlus_bench.scoring.sealed import SealedScorer, ScenarioAdapter
from kahlus_bench.synthetic.ground_truth import (
    ConfounderVAR,
    LinearGaussianVAR,
    MediatedVAR,
)


def _make_scenarios(n_samples: int, seed: int):
    return [
        ScenarioAdapter("LinearGaussianVAR@a=0.05",
                        LinearGaussianVAR(n_samples=n_samples, seed=seed, a=0.05).sample),
        ScenarioAdapter("LinearGaussianVAR@a=0.15",
                        LinearGaussianVAR(n_samples=n_samples, seed=seed, a=0.15).sample),
        ScenarioAdapter("LinearGaussianVAR@a=0.30",
                        LinearGaussianVAR(n_samples=n_samples, seed=seed, a=0.30).sample),
        ScenarioAdapter("ConfounderVAR",
                        ConfounderVAR(n_samples=n_samples, seed=seed).sample),
        ScenarioAdapter("MediatedVAR",
                        MediatedVAR(n_samples=n_samples, seed=seed).sample),
    ]


def _make_methods():
    methods = [
        CorrelationMethod(),
        PairwiseTransferEntropyMethod(),
        PhaseAmplitudeCouplingMethod(),
        ConditionalVARMethod(),
    ]
    # neuroforecast is an optional dependency -- the adapter raises a clear
    # ImportError at call time, not at import time, so the core benchmark
    # keeps working with numpy alone. Here we only add the method to the
    # scorecard if neuroforecast is actually importable, so the CLI never
    # raises on a fresh checkout.
    try:
        import neuroforecast  # noqa: F401
        from kahlus_bench.methods.neuroforecast_adapter import NeuroforecastMethod
        methods.append(NeuroforecastMethod(n_boot=400))
    except ImportError:
        pass
    return methods


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Kahlus-Bench scorecard.")
    p.add_argument("--n-draws", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-floor", type=float, default=1e-3,
                   help="Certification threshold in bits (default 1e-3).")
    args = p.parse_args(argv)

    header = (
        f"{'method':<20} {'scenario':<28} {'P':>5} {'R':>5} "
        f"{'nullCert':>9} {'medFalse':>9} {'floorBias':>10}"
    )
    print(header)
    print("-" * len(header))
    for scenario in _make_scenarios(args.n_samples, args.seed):
        scorer = SealedScorer(
            scenario, n_draws=args.n_draws,
            floor_floor=args.noise_floor,
        )
        for method in _make_methods():
            rep = scorer.score(method)
            print(
                f"{rep.method_name:<20} {rep.scenario_name:<28} "
                f"{rep.precision:>5.2f} {rep.recall:>5.2f} "
                f"{rep.null_certified_rate:>9.3f} "
                f"{rep.mediated_false_edge_rate:>9.3f} "
                f"{rep.floor_bias_bits:>+10.4f}"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
