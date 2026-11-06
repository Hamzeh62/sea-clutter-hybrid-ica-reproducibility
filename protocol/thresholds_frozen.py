"""Frozen threshold estimator (Phase 4C).

Implements ``THRESHOLD_ESTIMATOR_SPECIFICATION.md`` exactly:

* empirical Hyndman-Fan type-7 quantile when ``n*p >= 10``,
* otherwise the legacy exponential-tail extrapolation, labelled as such,
* cluster bootstrap over cells for the threshold uncertainty,
* hard errors instead of silent fallbacks.

No randomness is consumed by the estimator itself; the bootstrap takes an
explicit seed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

MIN_EVENTS_FOR_EMPIRICAL = 10
TAIL_QUANTILE = 0.995


@dataclass(frozen=True)
class Threshold:
    value: float
    method: str
    n: int
    design_pfa: float
    cal_exceedance: float
    bracket_lo: float | None
    bracket_hi: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _check(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("empty calibration vector")
    if not np.all(np.isfinite(s)):
        raise ValueError(f"{int(np.sum(~np.isfinite(s)))} non-finite calibration scores")
    return s


def extrapolate_tail(s: np.ndarray, pfa: float) -> float:
    """Legacy exponential-tail extrapolation above the 99.5 % quantile."""
    t0 = float(np.quantile(s, TAIL_QUANTILE))
    exc = s[s > t0]
    if exc.size < 2:
        raise ValueError("tail has fewer than two exceedances; cannot extrapolate")
    lam = exc.size / (exc.sum() - exc.size * t0 + 1e-30)
    c = exc.size / s.size
    return float(t0 + np.log(c / pfa) / lam)


def estimate(scores, design_pfa: float) -> Threshold:
    s = _check(scores)
    n = s.size
    if not (0.0 < design_pfa < 1.0):
        raise ValueError("design_pfa must lie in (0,1)")
    if n * design_pfa >= MIN_EVENTS_FOR_EMPIRICAL:
        t = float(np.quantile(s, 1.0 - design_pfa, method="linear"))
        pos = (n - 1) * (1.0 - design_pfa)
        lo_i, hi_i = int(np.floor(pos)), int(np.ceil(pos))
        ordered = np.sort(s)
        lo, hi = float(ordered[lo_i]), float(ordered[hi_i])
        method = "empirical"
    else:
        t = extrapolate_tail(s, design_pfa)
        lo = hi = None
        method = "extrapolated"
    return Threshold(value=t, method=method, n=n, design_pfa=float(design_pfa),
                     cal_exceedance=float(np.mean(s > t)), bracket_lo=lo, bracket_hi=hi)


def cluster_bootstrap(scores, cells, design_pfa: float, seed: int,
                      n_boot: int = 2000, stratified: bool = True) -> dict:
    """Resample range cells (optionally within acquisition) and re-estimate.

    ``cells`` must be an array of hashable cell labels aligned with ``scores``.
    When labels are tuples ``(acq, cell)`` the stratified variant resamples
    cells within each acquisition, preserving the acquisition composition.
    """
    s = _check(scores)
    labels = list(cells)
    if len(labels) != s.size:
        raise ValueError(f"cells ({len(labels)}) and scores ({s.size}) must have equal length")
    rng = np.random.default_rng(seed)
    draws = []
    by_label: dict[object, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    groups = {lab: np.asarray(idx) for lab, idx in by_label.items()}
    labels = sorted(by_label)
    if stratified:
        acqs = sorted({lab[0] for lab in labels if isinstance(lab, tuple)})
        if not acqs:
            raise ValueError("stratified bootstrap needs (acq, cell) labels")
        per_acq = {a: [lab for lab in labels if lab[0] == a] for a in acqs}
    for _ in range(n_boot):
        if stratified:
            picked = []
            for a in acqs:
                labs = per_acq[a]
                idx = rng.integers(0, len(labs), size=len(labs))
                picked.extend(labs[i] for i in idx)
        else:
            idx = rng.integers(0, len(labels), size=len(labels))
            picked = [labels[i] for i in idx]
        vals = np.concatenate([s[groups[lab]] for lab in picked])
        draws.append(estimate(vals, design_pfa).value)
    draws = np.asarray(draws)
    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "stratified": bool(stratified),
        "mean": float(draws.mean()),
        "rel_sd": float(draws.std() / max(abs(draws.mean()), 1e-30)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def realised_pfa(scores, threshold: float) -> dict:
    s = _check(scores)
    k = int(np.sum(s > threshold))
    return {"k": k, "n": int(s.size), "pfa": k / s.size}
