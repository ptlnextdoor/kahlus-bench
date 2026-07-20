"""Adapter that wraps neuroforecast's certified directed-information graph as
a kahlus-bench ``Method``.

The whole benchmark payoff is this file: once neuroforecast is installed, the
benchmark can score its certified estimator on the same synthetic ground truth
as correlation, PAC, and pairwise TE -- and the pre-registered expectation is
that neuroforecast, which conditions on all other streams' pasts and certifies
a finite-sample LCB, passes ConfounderVAR and MediatedVAR *as well as or
better than* the lightweight ConditionalVARMethod baseline shipped here.

This adapter is deliberately an optional import. ``neuroforecast`` is not a
declared dependency of kahlus-bench; the core benchmark must remain runnable
with numpy alone. If neuroforecast is not importable, importing this module
raises a clear ``ImportError`` at use time, not at package import time, so the
rest of the benchmark keeps working.

To use:

    pip install git+https://github.com/ptlnextdoor/neuroforecast

Then:

    from kahlus_bench.methods.neuroforecast_adapter import NeuroforecastMethod
    scorer.score(NeuroforecastMethod())

The adapter performs a tiny bit of plumbing to translate
``ObservableDraw.series`` and ``clusters`` into neuroforecast's expected
arguments, and translates the returned (m, m) cdi/lcb matrices into the per-
edge ``EdgeReport`` dict that kahlus-bench's scorer consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kahlus_bench.contract import EdgeReport, MethodReport, ObservableDraw


@dataclass
class NeuroforecastMethod:
    """Wraps neuroforecast's certified directed-information graph as a
    kahlus-bench Method. Conditions on all other streams' pasts (so it should
    pass ConfounderVAR and MediatedVAR) with a subject-cluster bootstrap LCB
    in bits (so its certifications are finite-sample-honest)."""

    name: str = "Neuroforecast"
    lag: int = 1
    n_boot: int = 600

    def __call__(self, draw: ObservableDraw) -> MethodReport:
        try:
            from neuroforecast.graph import directed_information_graph  # type: ignore
        except ImportError as exc:  # pragma: no cover -- optional dep
            raise ImportError(
                "neuroforecast is not installed. Install with: "
                "pip install git+https://github.com/ptlnextdoor/neuroforecast"
            ) from exc

        series = np.asarray(draw.series, dtype=np.float64)
        m = series.shape[1]
        names = [f"s{i}" for i in range(m)]
        clusters = draw.clusters if draw.clusters is not None else None

        # neuroforecast expects clusters of length T - lag.
        if clusters is not None and len(clusters) == series.shape[0]:
            clusters = clusters[self.lag:]

        result = directed_information_graph(
            series=series,
            names=names,
            lag=self.lag,
            clusters=clusters,
            floor=0.0,
            n_boot=self.n_boot,
            seed=draw.spec.sample_seed,
        )

        edges: dict[tuple[int, int], EdgeReport] = {}
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                cdi_bits = float(result.cdi[i, j])
                lcb_bits = float(result.lcb[i, j])
                # neuroforecast conditions on the rest, so any edge it certifies
                # is reported as direct (direct_fraction = 1.0). Edges that
                # vanish after conditioning sit at lcb <= 0 and are not
                # certified by the scorer.
                edges[(i, j)] = EdgeReport(
                    cdi_bits=cdi_bits,
                    lcb95_bits=max(lcb_bits, 0.0),
                    direct_fraction=1.0 if cdi_bits > 0 else 0.0,
                )
        return MethodReport(edges=edges, claimed_floor_bits=0.0)
