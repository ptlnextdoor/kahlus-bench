"""Synthetic ground-truth generators.

Each scenario is a controlled source the scorer samples from. The scenario
knows the true directed graph and the true directed information per edge; it
hands the *method* only the observable series and clusters. Three scenarios
are enough to start, each stressing a different honesty axis:

  * ``LinearGaussianVAR`` -- a VAR(1) Gaussian system where the true directed
    information per edge is analytic (Quinn-Kiyavash-Coleman, 2015, IEEE TIT).
    Stresses basic detection and calibration. Ground truth = support of the
    transition matrix. Coupling strengths are graded so the scorer can sweep.

  * ``ConfounderVAR`` -- a system where two streams are driven by a common
    hidden driver and have NO direct edge between them, but a pairwise
    correlation or pairwise transfer entropy will light up. True direct DI is
    zero everywhere except the driver-output edges. This is the r=0.97 trap,
    encoded as a unit test the method must pass: certify no edge among the
    driven streams.

  * ``MediatedVAR`` -- X -> Z -> Y. Pairwise X->Y is nonzero, but conditioning
    on Z drives true DI(X->Y || rest) to zero. The benchmark's
    mediated_false_edge_rate lives here: pairwise estimators (correlation, PAC,
    pairwise TE) will fail it, neuroforecast must not.

The generators deliberately share a small numerical core: a stable VAR(1)
simulator with per-stream innovation noise, so leakage control is identical
across scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from kahlus_bench.contract import (
    ObservableDraw,
    ScenarioSpec,
    SealedDraw,
    TrueGraph,
)


# ---------------------------------------------------------------------------
# Core: stable VAR(1) simulator with leakage-sealed draws.
# ---------------------------------------------------------------------------


def _var1_simulate(
    A: np.ndarray,           # (m, m) transition; A[i, j] = effect of X_j(t-1) on X_i(t)
    sigma: np.ndarray,       # (m,) innovation std devs per stream
    n_samples: int,
    rng: np.random.Generator,
    burn_in: int = 200,
) -> np.ndarray:
    """Simulate a stable VAR(1). Returns (n_samples, m) float64.

    Stability is checked by the caller (we assert the spectral radius < 1).
    """
    m = A.shape[0]
    x = rng.standard_normal(m) * sigma
    for _ in range(burn_in):           # burn from any initial condition
        x = A @ x + rng.standard_normal(m) * sigma
    out = np.empty((n_samples, m), dtype=np.float64)
    for t in range(n_samples):
        x = A @ x + rng.standard_normal(m) * sigma
        out[t] = x
    return out


def _spectral_radius(A: np.ndarray) -> float:
    eig = np.linalg.eigvals(A)
    return float(np.max(np.abs(eig)))


# ---------------------------------------------------------------------------
# Analytic directed information for linear-Gaussian VAR(1), per edge.
# For X_i(t) = sum_j A[i, j] X_j(t-1) + eps_i, eps ~ N(0, Sigma), the
# conditional DI (i -> j || rest) is the KL between the predictive of X_j(t)
# given the full past vs. given the past excluding X_i. For a jointly Gaussian
# VAR(1) system this reduces to a closed form on the gramians; we
# compute it by a per-edge exclusion of the j-th column contribution and a
# 1-D Gaussian KL. This is the SAME analytic used by neuroforecast's
# ``analytic_cdi_gaussian`` and is the calibration anchor for the whole bench.
# ---------------------------------------------------------------------------


def _analytic_cdi_var1(A: np.ndarray, sigma_diag: np.ndarray) -> dict[tuple[int, int], float]:
    """Per-edge conditional DI (bits) for a VAR(1) Gaussian system.

    For the linear-Gaussian VAR(1) the conditional DI(i -> j || rest) equals the
    reduction in conditional entropy of X_j(t) from including X_i(t-1) given the
    rest of the past. Because the system is linear-Gaussian with diagonal
    innovations, this is the KL between two Gaussians with the same mean whose
    variances differ by removing the i-th predictor from the regression of
    X_j(t) on the past. That variance reduction is R^2 * Var(X_j | full past),
    and the per-edge DI is (1/2) log_2(1 / (1 - r^2_{j|full - i})) where
    r^2_{j|full - i} is the squared partial correlation of X_j(t) with X_i(t-1)
    after projecting out the other pasts. Computed via the covariance of
    [X(t), X(t-1)].
    """
    m = A.shape[0]
    # Stationary covariance Sigma solves: Sigma = A Sigma A^T + diag(sigma^2).
    # Solve the discrete Lyapunov equation.
    Q = np.diag(sigma_diag ** 2)
    Sigma = np.linalg.solve(
        np.eye(m * m) - np.kron(A, A),
        Q.reshape(m * m, order="F"),
    ).reshape(m, m, order="F")

    # Cross-cov Lag-1 is A Sigma (stationary).
    C01 = A @ Sigma

    # For each target j and donor i, partial r^2 of X_j(t) on X_i(t-1) given
    # the *other* past predictors. Compute via Schur complement of the gram.
    out: dict[tuple[int, int], float] = {}
    for j in range(m):
        # Predict X_j(t) from X(t-1) full past.
        Gxx = Sigma                       # (m, m), covariance of X(t-1)
        gxy = C01[j, :]                   # (m,), cov(X(t-1), X_j(t))
        var_y = Sigma[j, j]               # var(X_j(t))
        # Full regression R^2 (var explained).
        # Use pseudo-inverse for safety with singular Gxx.
        beta_full = np.linalg.lstsq(Gxx, gxy, rcond=None)[0]
        var_full = var_y - gxy @ beta_full     # residual variance, full model
        for i in range(m):
            if i == j:
                continue
            # Reduced model: drop predictor i.
            mask = np.full(m, True)
            mask[i] = False
            Gred = Gxx[np.ix_(mask, mask)]
            gred = gxy[mask]
            beta_red = np.linalg.lstsq(Gred, gred, rcond=None)[0]
            var_red = var_y - gred @ beta_red   # residual var, reduced model
            # Variance reduction from including i, given the rest.
            r2 = 1.0 - var_full / max(var_red, 1e-300)
            r2 = max(r2, 0.0)
            # per-edge conditional DI (bits).
            di = -0.5 * np.log2(max(1.0 - r2, 1e-300)) if r2 > 0 else 0.0
            out[(i, j)] = float(di)
    return out


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------


@dataclass
class _BaseScenario:
    n_samples: int = 4096
    lag: int = 4
    seed: int = 0
    _offset: int = 0           # internal draw counter

    def spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            n_samples=self.n_samples,
            n_streams=self._n_streams(),
            lag=self.lag,
            cluster_blocks=(self.n_samples,),   # single subject; subdivide for blocks
            sample_seed=self.seed + self._offset,
        )

    def _n_streams(self) -> int:  # pragma: no cover -- override
        raise NotImplementedError

    # subclasses fill these in.
    def _build(self) -> tuple[np.ndarray, np.ndarray, TrueGraph]:
        raise NotImplementedError

    def sample(self) -> SealedDraw:
        rng = np.random.default_rng(self.seed + self._offset)
        self._offset += 1
        A, sigma, truth = self._build()
        if _spectral_radius(A) >= 1.0:
            raise RuntimeError("scenario transition matrix is not stable")
        series = _var1_simulate(A, sigma, self.n_samples, rng)
        clusters = np.zeros(self.n_samples, dtype=np.int64)
        # Split into 16 blocks so a cluster-bootstrap has meaning.
        block = self.n_samples // 16
        for k in range(16):
            clusters[k * block: (k + 1) * block] = k
        obs = ObservableDraw(series=series, clusters=clusters, spec=self.spec())
        return SealedDraw(observable=obs, truth=truth)


# ---- 1. Linear-Gaussian VAR: known DI, basic detection+calibration axis ----


@dataclass
class LinearGaussianVAR(_BaseScenario):
    """Three-stream VAR(1) with two true direct edges and graded coupling.

    Default coupling a in (0.05, 0.35) sweeps from below-detection to clearly
    detectable. The analytic DI per edge is computed from the transition matrix
    via the closed form, so the scorer's floor is known exactly.
    """

    a: float = 0.15      # off-diagonal coupling strength
    snr: float = 1.0     # innovation sigma scaling

    def _n_streams(self) -> int:
        return 3

    def _build(self) -> tuple[np.ndarray, np.ndarray, TrueGraph]:
        A = np.array([
            [0.4, self.a, 0.0],   # X1 depends on X2(t-1): edge 2->1
            [0.0, 0.4, self.a],   # X2 depends on X3(t-1): edge 3->2
            [0.0, 0.0, 0.4],
        ], dtype=np.float64)
        sigma = np.full(3, 1.0 * self.snr, dtype=np.float64)
        truth_bits = _analytic_cdi_var1(A, sigma)
        direct = frozenset({(1, 0), (2, 1)})   # 2->1, 3->2 (0-indexed)
        all_edges = [(i, j) for i in range(3) for j in range(3) if i != j]
        null = frozenset(e for e in all_edges if e not in direct)
        truth = TrueGraph(
            direct_edges=direct,
            mediated_edges=frozenset(),
            null_edges=null,
            true_bits=truth_bits,
        )
        return A, sigma, truth


# ---- 2. Confounder VAR: the r=0.97 trap, encoded as a must-fail axis ----


@dataclass
class ConfounderVAR(_BaseScenario):
    """Hidden driver C feeds X and Y. True DI(X->Y || rest) = 0 because the
    driver is observable (passed as stream 0); conditioning on it kills the
    spurious X->Y edge. The pairwise correlation X<->Y is large and positive,
    so any method that does not condition on the rest must fail the
    null_certified_rate metric. neuroforecast conditions on rest and must pass.
    """

    a_driver_to_x: float = 0.6
    a_driver_to_y: float = 0.6
    snr: float = 0.5

    def _n_streams(self) -> int:
        return 3   # C, X, Y

    def _build(self) -> tuple[np.ndarray, np.ndarray, TrueGraph]:
        # C(t) = 0.4 C(t-1) + eps_c.
        # X(t) = a C(t-1) + eps_x ; Y(t) = a C(t-1) + eps_y.
        A = np.array([
            [0.4, 0.0, 0.0],
            [self.a_driver_to_x, 0.4, 0.0],
            [self.a_driver_to_y, 0.0, 0.4],
        ], dtype=np.float64)
        sigma = np.full(3, 1.0 * self.snr, dtype=np.float64)
        truth_bits = _analytic_cdi_var1(A, sigma)
        # True direct edges: C->X (donor 0 -> target 1), C->Y (0 -> 2) only.
        direct = frozenset({(0, 1), (0, 2)})
        all_edges = [(i, j) for i in range(3) for j in range(3) if i != j]
        null = frozenset(e for e in all_edges if e not in direct)
        truth = TrueGraph(
            direct_edges=direct,
            mediated_edges=frozenset(),   # X<->Y is pure confounding, not mediation
            null_edges=null,
            true_bits=truth_bits,
        )
        return A, sigma, truth


# ---- 3. Mediated VAR: direct-vs-mediated axis ---------------------------


@dataclass
class MediatedVAR(_BaseScenario):
    """Chain X -> Z -> Y. Pairwise X->Y is nonzero, but conditioning on Z drives
    true DI(X->Y || rest) to zero. The edge that a pairwise method will
    falsely certify as direct is (0, 2). neuroforecast must report it as
    mediated. This is where mediated_false_edge_rate is measured.
    """

    a_xz: float = 0.5
    a_zy: float = 0.5
    snr: float = 0.5

    def _n_streams(self) -> int:
        return 3   # X, Z, Y

    def _build(self) -> tuple[np.ndarray, np.ndarray, TrueGraph]:
        # X(t) = 0.4 X(t-1) + eps
        # Z(t) = a X(t-1) + 0.4 Z(t-1) + eps
        # Y(t) = a Z(t-1) + 0.4 Y(t-1) + eps
        A = np.array([
            [0.4, 0.0, 0.0],
            [self.a_xz, 0.4, 0.0],
            [0.0, self.a_zy, 0.4],
        ], dtype=np.float64)
        sigma = np.full(3, 1.0 * self.snr, dtype=np.float64)
        truth_bits = _analytic_cdi_var1(A, sigma)
        # True direct edges: X->Z (0, 1), Z->Y (1, 2).
        direct = frozenset({(0, 1), (1, 2)})
        # Mediated (pairwise nonzero, true DI = 0 after conditioning): X->Y.
        mediated = frozenset({(0, 2)})
        all_edges = [(i, j) for i in range(3) for j in range(3) if i != j]
        null = frozenset(
            e for e in all_edges
            if e not in direct and e not in mediated
        )
        truth = TrueGraph(
            direct_edges=direct,
            mediated_edges=mediated,
            null_edges=null,
            true_bits=truth_bits,
        )
        return A, sigma, truth


# Iterator over the three default scenarios, with a graded coupling sweep ----


def default_scenarios(n_samples: int = 4096, seed: int = 0) -> Iterator[tuple[str, _BaseScenario]]:
    """A small, opinionated default grid. Pair each scenario with a sweep of
    coupling strengths so the scorer can plot detection power vs truth.
    """
    yield ("LinearGaussianVAR", LinearGaussianVAR(n_samples=n_samples, seed=seed, a=0.05))
    yield ("LinearGaussianVAR", LinearGaussianVAR(n_samples=n_samples, seed=seed, a=0.15))
    yield ("LinearGaussianVAR", LinearGaussianVAR(n_samples=n_samples, seed=seed, a=0.30))
    yield ("ConfounderVAR", ConfounderVAR(n_samples=n_samples, seed=seed))
    yield ("MediatedVAR", MediatedVAR(n_samples=n_samples, seed=seed))
