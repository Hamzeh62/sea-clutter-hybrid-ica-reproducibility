"""Report the E3 reference-contamination sensitivity.

Compares, on identical frozen trials and identical frozen thresholds, the detection
probability of the Hybrid, AR(8)+NMF and two-tap notch when the injected target is

* ``VV only`` - the frozen campaign convention (VH is a target-free reference), and
* ``VV + VH``  - the upper bound on reference leakage, this sensitivity.

Writes ``reports/MINIMAL_CAMPAIGN_E3_REFERENCE_CONTAMINATION.csv`` and a JSON summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import phase5_analyze_v2 as A
import phase5_common as C

ARMS = ("hybrid", "ar8", "notch")
SCR = 0.0
# (label, VH target gain in dB) - None means the frozen VV-only campaign
CONDITIONS = (("vv_only", None), ("vh_-10dB", -10.0), ("vh_0dB", 0.0))


def threshold(cal: np.ndarray, pfa: float = A.DESIGN_PFA) -> float:
    s = np.sort(cal)[::-1]
    import math
    return float(s[max(1, int(math.floor(len(s) * pfa))) - 1])


def main() -> int:
    cfg = C.config()
    run = C.run_dir(cfg)
    h0, h1 = A.load_scores(run)
    rows, summary = [], {"condition": {}, "contrasts": {}}
    for fold in ("fold1", "fold2"):
        thr = {a: threshold(h0[(fold, "cal")][a]["score"]) for a in ARMS}
        main_ids = {a: h1[(fold, A.ARM_GROUP[a])]["trial_ids"] for a in ARMS}
        ref_ids = main_ids["hybrid"]
        for a in ARMS:
            if not np.array_equal(main_ids[a], ref_ids):
                raise RuntimeError(f"{fold}: trial sets differ for {a}")
        base = {a: h1[(fold, A.ARM_GROUP[a])]["scores"][a][:, 6] > thr[a] for a in ARMS}
        masks = {"vv_only": base}
        for label, gain in CONDITIONS:
            if gain is None:
                continue
            p = run / "sensitivity" / f"{fold}__refcontam__scr{SCR:g}__vh{gain:g}dB.npz"
            with np.load(p) as z:
                sids = z["trial_ids"].astype(str)
                pos = {t: i for i, t in enumerate(sids)}
                idx = np.array([pos[t] for t in ref_ids], int)
                masks[label] = {a: z[f"arm_{a}"][idx] > thr[a] for a in ARMS}
                vh_ratio = float(np.median(z["vh_target_power_ratio"]))
            summary["condition"][f"{fold}|{label}"] = {
                "vh_target_to_clutter_median": vh_ratio}
        for a in ARMS:
            for label, _ in CONDITIONS:
                rows.append({"fold": fold, "arm": a, "condition": label,
                             "scr_db": SCR, "threshold": thr[a],
                             "n_trials": int(len(ref_ids)),
                             "pd": float(masks[label][a].mean()),
                             "pd_delta_vs_vv_only": float(masks[label][a].mean()
                                                          - base[a].mean()),
                             "n_detected": int(masks[label][a].sum())})
        a, b = "hybrid", "ar8"
        for label, _ in CONDITIONS:
            m = masks[label]
            n01 = int(np.sum(m[a] & ~m[b]))
            n10 = int(np.sum(~m[a] & m[b]))
            pv = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
            summary["contrasts"][f"{fold}|{label}|hybrid-ar8"] = {
                "pd_hybrid": float(m[a].mean()), "pd_ar8": float(m[b].mean()),
                "delta_pd": float(m[a].mean() - m[b].mean()),
                "a_only": n01, "b_only": n10, "mcnemar_p": pv}

    out = Path(__file__).resolve().parents[1] / "reports"
    keys = [k for k in rows[0] if not isinstance(rows[0][k], dict)]
    with open(out / "MINIMAL_CAMPAIGN_E3_REFERENCE_CONTAMINATION.csv", "w", newline="",
              encoding="utf-8") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.10g}" if isinstance(r[k], float) else r[k]) for k in keys})
    summary["rows"] = rows
    (out / "MINIMAL_CAMPAIGN_E3_REFERENCE_CONTAMINATION.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["contrasts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
