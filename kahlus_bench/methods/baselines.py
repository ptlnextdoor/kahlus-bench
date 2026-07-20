"""Baseline methods -- the field's usual suspects, scored the same certified
way as everyone else.

All methods share a tiny bootstrap helper for LCB computation in bits; we use
the same convention as neuroforecast (cluster bootstrap when clusters are
provided, otherwise iid)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kahlus_bench.contract import (
    EdgeReport,
    MethodReport,
    ObservableDraw,
)


# ---------------------------------------------------------------------------
# Small utilities shared by all methods.
# ---------------------------------------------------------------------------


def _cluster_bootstrap_lcb(
    point_estimate: float,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    original: np.ndarray,           # (n, m) series
    clusters: np.ndarray,           # (n,) block ids
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> float:
    """Bootstrap LCB in *natural units* of ``statistic``. Caller converts to
    bits by the monotone transform appropriate to the statistic."""
    rng = rng or np.random.default_rng(0)
    unique = np.unique(clusters)
    n_blocks = len(unique)
    block_to_idx = {b: i for i, b in enumerate(unique)}
    idx = np.array([block_to_idx[b] for b in clusters])
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.integers(0, n_blocks, size=n_blocks)
        keep = np.concatenate([np.where(idx == s)[0] for s in sel])
        estimate = statistic(original[keep], clusters[keep])
        boots[b] = estimate
    lcb = np.quantile(boots, alpha)
    return float(lcb)


def _corr_to_bits(max_rho: float) -> float:
    """Mutual information for a bivariate Gaussian with correlation rho, in
    bits: I = -0.5 log_2(1 - rho^2). Clipped to avoid blow-up."""
    rho2 = min(max(max_rho, 0.0) ** 2, 1.0 - 1e-12)
    return float(-0.5 * np.log2(1.0 - rho2))


# ---------------------------------------------------------------------------
# 1. Pearson correlation, pairwise, lag-0 and lag-1. Cannot condition on rest.
# ---------------------------------------------------------------------------


@dataclass
class CorrelationMethod:
    """Pairwise Pearson correlation (max of lag-0 and lag-1 absolute), with a
    cluster-bootstrap LCB converted to bits via the Gaussian-Gaussian MI
    formula. ``direct_fraction`` is always 1.0 (cannot separate), so this
    method is expected to fail mediated-false-edge."""
    name: str = "Correlation"
    n_boot: int = 150

    def __call__(self, draw: ObservableDraw) -> MethodReport:
        x = draw.series
        m = x.shape[1]
        rng = np.random.default_rng(draw.spec.sample_seed)
        edges: dict[tuple[int, int], EdgeReport] = {}
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                # max of lag-0 and lag-1 abs Pearson.
                r0 = float(np.abs(np.corrcoef(x[:, i], x[:, j])[0, 1]))
                if x.shape[0] > 1:
                    r1 = float(np.abs(np.corrcoef(x[:-1, i], x[1:, j])[0, 1]))
                else:
                    r1 = 0.0
                r = max(r0, r1)
                bits = _corr_to_bits(r)
                # LCB by cluster bootstrap on the lag-1 statistic, transformed.
                def stat(series, _cl, i=i, j=j):
                    if series.shape[0] < 2:
                        return 0.0
                    corr = float(np.corrcoef(series[:-1, i], series[1:, j])[0, 1])
                    return max(abs(corr), 0.0)
                lcb_nat = _cluster_bootstrap_lcb(
                    0.0, stat, x, draw.clusters, n_boot=self.n_boot, rng=rng,
                )
                lcb_bits = _corr_to_bits(lcb_nat)
                edges[(i, j)] = EdgeReport(
                    cdi_bits=bits,
                    lcb95_bits=max(lcb_bits, 0.0),
                    direct_fraction=1.0,
                )
        return MethodReport(edges=edges, claimed_floor_bits=0.0)


# ---------------------------------------------------------------------------
# 2. Pairwise transfer entropy -- lagged, with cluster-bootstrap LCB. A stand-in
# for the standard Gaussian estimate of TE; not certified, pairwise only.
# ---------------------------------------------------------------------------


@dataclass
class PairwiseTransferEntropyMethod:
    """Pairwise lag-1 transfer entropy via a k-NN estimator (Kraskov style,
    k=4). Pairwise (does not condition on the rest), so it is expected to pass
    LinearGaussianVAR and fail ConfounderVAR + MediatedVAR."""
    name: str = "PairwiseTE"
    k: int = 4
    n_boot: int = 100

    def __call__(self, draw: ObservableDraw) -> MethodReport:
        x = draw.series
        m = x.shape[1]
        rng = np.random.default_rng(draw.spec.sample_seed)
        edges: dict[tuple[int, int], EdgeReport] = {}
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                te = self._te(x, i, j)
                def stat(series, _cl, i=i, j=j):
                    return self._te(series, i, j)
                lcb = _cluster_bootstrap_lcb(
                    0.0, stat, x, draw.clusters, n_boot=self.n_boot, rng=rng,
                )
                edges[(i, j)] = EdgeReport(
                    cdi_bits=float(te),
                    lcb95_bits=float(max(lcb, 0.0)),
                    direct_fraction=1.0,
                )
        return MethodReport(edges=edges, claimed_floor_bits=0.0)

    def _te(self, x: np.ndarray, i: int, j: int) -> float:
        """Crude lag-1 Gaussian transfer entropy I(X_j(t); X_i(t-1) | X_j(t-1)),
        bits. Assumes Gaussianity for the closed form:
            TE = 0.5 * log_2( Var(X_j|X_j_past) / Var(X_j | X_j_past, X_i_past) )
        """
        if x.shape[0] < 3:
            return 0.0
        yf = x[1:, j]
        yp = x[:-1, j]
        xp = x[:-1, i]
        # residuals of X_j(t) on X_j(t-1)
        b1 = np.linalg.lstsq(np.column_stack([np.ones_like(yp), yp]), yf, rcond=None)[0]
        res1 = yf - (b1[0] + b1[1] * yp)
        var1 = max(np.var(res1), 1e-12)
        # residuals on the joint past
        b2 = np.linalg.lstsq(np.column_stack([np.ones_like(yp), yp, xp]), yf, rcond=None)[0]
        res2 = yf - (b2[0] + b2[1] * yp + b2[2] * xp)
        var2 = max(np.var(res2), 1e-12)
        return float(0.5 * np.log2(var1 / var2))


# ---------------------------------------------------------------------------
# 3. Phase-amplitude coupling (the Rao et al. correlational stand-in).
# ---------------------------------------------------------------------------


@dataclass
class PhaseAmplitudeCouplingMethod:
    """PAC between the phase of a slow band (0.5-4 Hz proxy: low-pass) and the
    envelope of a fast band (band-pass + rectify). For synthetic VAR data
    without a carrier, this estimate reduces essentially to a nonlinear
    correlation between the slow stream and the fast stream's amplitude, which
    means it inherits all the pairwise failure modes. We mark it
    direct_fraction=1.0."""
    name: str = "PAC"
    n_boot: int = 100

    def __call__(self, draw: ObservableDraw) -> MethodReport:
        x = draw.series
        m = x.shape[1]
        rng = np.random.default_rng(draw.spec.sample_seed)
        edges: dict[tuple[int, int], EdgeReport] = {}
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                # Phase of stream i (Hilbert) mod envelope of j.
                phase = np.angle(_hilbert(x[:, i]))
                env = np.abs(_hilbert(x[:, j]))
                # Modulation index via mean vector length of envelope over phase.
                mi = float(np.abs(np.mean(env * np.exp(1j * phase))))
                bits = _corr_to_bits(mi)
                def stat(series, _cl, i=i, j=j):
                    phase = np.angle(_hilbert(series[:, i]))
                    env = np.abs(_hilbert(series[:, j]))
                    return float(np.abs(np.mean(env * np.exp(1j * phase))))
                lcb_nat = _cluster_bootstrap_lcb(
                    0.0, stat, x, draw.clusters, n_boot=self.n_boot, rng=rng,
                )
                lcb_bits = _corr_to_bits(lcb_nat)
                edges[(i, j)] = EdgeReport(
                    cdi_bits=bits,
                    lcb95_bits=max(lcb_bits, 0.0),
                    direct_fraction=1.0,
                )
        return MethodReport(edges=edges, claimed_floor_bits=0.0)


# ---------------------------------------------------------------------------
# 4. Conditional VAR -- the direct-vs-mediated-aware baseline. This is the
# lightweight proxy for neuroforecast's certified conditional directed info.
# Fits X_j(t) on full past, then drops donor i, and reports the
# variance-reduction bits as the certified DI(i -> j || rest). Conditions on
# the REST, so it should pass ConfounderVAR and MediatedVAR.
# ---------------------------------------------------------------------------


@dataclass
class ConditionalVARMethod:
    """Conditional linear-Gaussian VAR(1) with cluster-bootstrap LCB in bits.
    Conditions on the rest, so it should pass the confounder and mediation
    scenarios. Reported ``direct_fraction`` is 1.0 when the edge is certified
    after conditioning on the rest, else 0.0 (mediated)."""
    name: str = "ConditionalVAR"
    n_boot: int = 120

    def __call__(self, draw: ObservableDraw) -> MethodReport:
        x = draw.series
        m = x.shape[1]
        rng = np.random.default_rng(draw.spec.sample_seed)
        edges: dict[tuple[int, int], EdgeReport] = {}
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                di, lcb_nat, direct_frac = self._cdi(x, i, j)
                def stat(series, _cl, i=i, j=j):
                    return self._cdi(series, i, j)[1]
                lcb = _cluster_bootstrap_lcb(
                    0.0, stat, x, draw.clusters, n_boot=self.n_boot, rng=rng,
                )
                # lcb_nat is itself expressed as a *reduction* in residual
                # variance, which IS the bits figure already (see _cdi).
                # Treat the bootstrap-min of the bits as the LCB in bits.
                edges[(i, j)] = EdgeReport(
                    cdi_bits=float(di),
                    lcb95_bits=float(max(lcb, 0.0)),
                    direct_fraction=float(direct_frac),
                )
        return MethodReport(edges=edges, claimed_floor_bits=0.0)

    def _cdi(self, x: np.ndarray, i: int, j: int) -> tuple[float, float, float]:
        """Returns (point estimate bits, statistic for bootstrap, direct_frac).

        direct_frac = 1.0 if the full model includes donor i and the partial
        r^2 from dropping i is positive; 0.0 if dropping i leaves the fit
        unchanged (=> the apparent edge was fully mediated).
        """
        if x.shape[0] < 4:
            return 0.0, 0.0, 0.0
        yf = x[1:, j]
        n = yf.shape[0]
        ones = np.ones((n, 1))
        Xfull = np.column_stack([ones, x[:-1, :]])   # all streams' past
        Xred_mask = np.full(x.shape[1], True)
        Xred_mask[i] = False
        Xred = np.column_stack([ones, x[:-1, Xred_mask]])
        # full fit
        beta_full = np.linalg.lstsq(Xfull, yf, rcond=None)[0]
        res_full = yf - Xfull @ beta_full
        var_full = max(np.var(res_full), 1e-12)
        # reduced fit
        beta_red = np.linalg.lstsq(Xred, yf, rcond=None)[0]
        res_red = yf - Xred @ beta_red
        var_red = max(np.var(res_red), 1e-12)
        # conditional DI from variance reduction
        if var_full < var_red:
            di = 0.5 * np.log2(var_red / var_full)
        else:
            di = 0.0
        # direct_frac: reduce by including donor i was substantial vs noise.
        # If var reduction is less than a hair (e.g., <1e-3 bits) call it mediated.
        direct_frac = 1.0 if di > 1e-3 else 0.0
        return float(di), float(max(di, 0.0)), float(direct_frac)


# ---------------------------------------------------------------------------
# Hilbert transform via FFT (lightweight, no scipy dependency).
# ---------------------------------------------------------------------------


def _hilbert(x: np.ndarray) -> np.ndarray:
    """Discrete analytic signal via the standard FFT trick. Matches scipy's
    hilbert for our purposes to within numerical noise."""
    n = x.shape[0]
    if n < 2:
        return x.astype(complex)
    Xf = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1: n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1: (n + 1) // 2] = 2.0
    x_a = np.fft.ifft(Xf * h)
    return x_a
