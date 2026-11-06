"""Phase 7 V3: reference definition and predictor self-cancellation.

Six variants of the AR(8)+NMF statistic on the *same* frozen trials, each thresholded at the
design Pfa from CAL-H0 scores built by the identical construction:

  deployed_g0      filter fitted on the clean VH of this CPI  (= the frozen protocol)
  deployed_g1      filter fitted on VH contaminated by the target at unit gain (the E3
                   convention, not the kappa-normalised V1 dSCR axis)
  target_channel   filter fitted on the CUT's own VV, target included: prediction from the
                   same channel the statistic scores
  temporal_oracle  filter fitted on the VH of the same cell +128 CPIs later: target-free,
                   simultaneous, still cross-polarised
  delay1           filter fitted on VV shifted by one sample: the target is present in the
                   reference, and no polarimetry is involved
  delay512         filter fitted on the previous non-overlapping CPI of VV: target-free,
                   one-polarisation temporal reference

`delay1` vs `delay512` is the polarimetry-free mechanism control: if `delay1` collapses while
`delay512` does not, the failure belongs to prediction-error filters fitted on target-bearing
data, not to the cross-polarised channel.  `deployed_g0` must reproduce the frozen AR(8) result
and `deployed_g1` the frozen E3 contamination result; both are verification gates.

Outputs: reports/PHASE7_V3_REFERENCE_RESULTS.csv, data/PHASE7_V3_{fold}.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
WORK7 = PROJECT / "results" / "phase7"      # released Phase 7 data (read only)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase5_common as C                            # noqa: E402
import phase5_runner as R                            # noqa: E402
from src.proposed.m4_detector import nmf_statistic   # noqa: E402

CFG = C.config()
DESIGN = C.DESIGN_PFA
BOOT = 20000
RNG = np.random.default_rng(20261031)
SHIFT = 128 * C.HOP          # temporal-oracle offset, in samples

VARIANTS = ("deployed_g0", "deployed_g1", "target_channel", "temporal_oracle",
            "delay1", "delay512")


def other_cpi(cube, start: int, cell: int) -> np.ndarray:
    """A non-overlapping CPI of the same cell: previous, else next, else the far end."""
    for s in (start - C.NP, start + C.NP, cube.shape[0] - 2 * C.NP):
        if s >= 0 and s + C.NP <= cube.shape[0] and abs(s - start) >= C.NP:
            return cube[s:s + C.NP, cell].astype(np.complex128)
    raise RuntimeError("no non-overlapping CPI available")


def variants(vv: np.ndarray, vh: np.ndarray, vv_other: np.ndarray, vh_other: np.ndarray,
             sig: np.ndarray | None, prep) -> dict[str, float]:
    vv_c = vv if sig is None else vv + sig
    sig0 = np.zeros(C.NP, complex) if sig is None else sig
    vh_g1 = vh + sig0
    out = {
        "deployed_g0": nmf_statistic(C.ar_residual(vv_c, C.ar_fit(vh)), prep),
        "deployed_g1": nmf_statistic(C.ar_residual(vv_c, C.ar_fit(vh_g1)), prep),
        "target_channel": nmf_statistic(C.ar_residual(vv_c, C.ar_fit(vv_c)), prep),
        "temporal_oracle": nmf_statistic(C.ar_residual(vv_c, C.ar_fit(vh_other)), prep),
        "delay1": nmf_statistic(C.ar_residual(vv_c, C.ar_fit(np.roll(vv_c, 1))), prep),
        "delay512": nmf_statistic(C.ar_residual(vv_c, C.ar_fit(vv_other)), prep),
    }
    return out


def collect(fold: str, limit: int | None) -> tuple[dict, dict, dict]:
    acqs = C.fold_acquisitions(CFG, fold)
    eng = R.Engine(CFG, fold, acqs["cal"])
    prep = eng.prep("ar8")
    units = C.pool_for("temporal", C.partition_units(fold, "CAL_H0", CFG), CFG, fold, "cal")
    if limit:
        units = units[:limit]
    s = eng.sampler
    cal: dict[str, list[float]] = {k: [] for k in VARIANTS}
    t0 = time.perf_counter()
    for i, u in enumerate(units):
        vv = s.data[u.acq][u.start:u.start + C.NP, u.cell].astype(np.complex128)
        vh = s.data_cross[u.acq][u.start:u.start + C.NP, u.cell].astype(np.complex128)
        vv_o = other_cpi(s.data[u.acq], u.start, u.cell)
        st = min(u.start + SHIFT, s.data_cross[u.acq].shape[0] - C.NP)
        vh_o = s.data_cross[u.acq][st:st + C.NP, u.cell].astype(np.complex128)
        for k, v in variants(vv, vh, vv_o, vh_o, None, prep).items():
            cal[k].append(v)
        if (i + 1) % 250 == 0:
            el = time.perf_counter() - t0
            print(f"  [cal] {i+1}/{len(units)} {el:.0f}s "
                  f"({1000*el/(i+1):.1f} ms/unit)", flush=True)

    eng = R.Engine(CFG, fold, acqs["test"])
    s = eng.sampler
    trials = C.h1_trials(fold, CFG)
    if limit:
        trials = trials[:limit]
    te: dict[str, list[float]] = {k: [] for k in VARIANTS}
    t0 = time.perf_counter()
    for i, t in enumerate(trials):
        sig, _, _, _ = C.make_target(t, s, eng.band(t.acq), 0.0)
        vv = s.data[t.acq][t.start:t.start + C.NP, t.cell].astype(np.complex128)
        vh = s.data_cross[t.acq][t.start:t.start + C.NP, t.cell].astype(np.complex128)
        vv_o = other_cpi(s.data[t.acq], t.start, t.cell)
        st = min(t.start + SHIFT, s.data_cross[t.acq].shape[0] - C.NP)
        vh_o = s.data_cross[t.acq][st:st + C.NP, t.cell].astype(np.complex128)
        for k, v in variants(vv, vh, vv_o, vh_o, sig, prep).items():
            te[k].append(v)
        if (i + 1) % 250 == 0:
            el = time.perf_counter() - t0
            print(f"  [test] {i+1}/{len(trials)} {el:.0f}s "
                  f"({1000*el/(i+1):.1f} ms/trial)", flush=True)
    meta = {"n_cal": len(units), "n_trial": len(trials),
            "acq": [t.acq for t in trials],
            "trial_ids": [t.trial_id for t in trials]}
    return cal, te, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=("fold1", "fold2"))
    ap.add_argument("--limit", type=int, default=0, help="probe only (not a result)")
    args = ap.parse_args(argv)
    fold = args.fold
    print(f"[v3] {fold} start {time.strftime('%H:%M:%S')}", flush=True)
    cal, te, meta = collect(fold, args.limit or None)

    rows, store = [], {}
    for k in VARIANTS:
        c, x = np.array(cal[k]), np.array(te[k])
        if len(c) < 100:
            print(f"[v3] {fold} {k}: n_cal={len(c)} too small for a Pfa estimate, skipped")
            continue
        thr = float(np.quantile(c, 1.0 - DESIGN, method="higher"))
        h = x > thr
        store[k] = {"hit": h.tolist(), "thr": thr}
        rows.append({"fold": fold, "variant": k, "n_cal": len(c), "n_trial": len(x),
                     "threshold": thr, "pd_0db": float(h.mean()),
                     "pd_lo": float(np.mean([h[np.array(meta["acq"]) == a].mean()
                                             for a in sorted(set(meta["acq"]))])),
                     "cal_pfa_at_own_thr": float(np.mean(c > thr)),
                     "cal_scores_median": float(np.median(c))})
        print(f"[v3] {fold} {k:16s} thr={thr:.4f} pd={rows[-1]['pd_0db']:.3f}", flush=True)

    summary = {"fold": fold, "n_cal": meta["n_cal"], "n_trial": meta["n_trial"],
               "variants": {r["variant"]: r["pd_0db"] for r in rows}, "contrasts": {}}
    acq = np.array(meta["acq"])
    for k in VARIANTS:
        if k == "deployed_g0" or k not in store:
            continue
        h = store["deployed_g0"]["hit"]
        r = store[k]["hit"]
        ha, ra = np.array(h), np.array(r)
        n01 = int(np.sum(ha & ~ra)); n10 = int(np.sum(~ha & ra))
        p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
        per = [float(ha[acq == a].mean() - ra[acq == a].mean()) for a in sorted(set(acq))]
        b = np.array([np.mean(RNG.choice(per, len(per), replace=True)) for _ in range(BOOT)])
        summary["contrasts"][f"g0_minus_{k}"] = {
            "delta_pd": float(ha.mean() - ra.mean()),
            "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)),
            "mcnemar_p": p, "n_a_only": n01, "n_b_only": n10,
            "n_acq_positive": int(sum(v > 0 for v in per)), "n_acq": len(per)}
        print(f"[v3] {fold} g0-{k:16s} d={summary['contrasts'][f'g0_minus_{k}']['delta_pd']:+.3f}"
              f" [{summary['contrasts'][f'g0_minus_{k}']['ci_lo']:+.3f},"
              f"{summary['contrasts'][f'g0_minus_{k}']['ci_hi']:+.3f}]"
              f" p={p:.2e}", flush=True)

    tag = "" if not args.limit else f"_probe{args.limit}"
    (WORK7 / f"data/PHASE7_V3_{fold}{tag}.json").write_text(
        json.dumps({**summary, "rows": rows}, indent=2), encoding="utf-8")
    with open(WORK7 / f"reports/PHASE7_V3_REFERENCE_RESULTS{tag}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[v3] {fold} done {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
