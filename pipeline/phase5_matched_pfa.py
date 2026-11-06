"""Pd at a common *realised* TEST-H0 false-alarm rate.

The frozen campaign compares arms at a common *design* PFA of 1e-3, each arm using its
own CAL order statistic.  The PFA audit shows those thresholds do not transfer equally
across folds (realised PFA from 0 to 9e-3), so a design-PFA comparison is not a
realised-PFA comparison.  Here each arm is thresholded on TEST-H0 itself at a requested
rate, which makes the false-alarm rate identical by construction and independent of
calibration; the price is that the threshold is no longer deployable, which is why this
is reported as a secondary, diagnostic comparison.

Writes ``reports/MINIMAL_CAMPAIGN_MATCHED_PFA.csv``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import phase5_analyze_v2 as A
import phase5_common as C

GRID = (1e-3, 2e-3, 5e-3, 1e-2)
SCR = 0.0


def main() -> int:
    cfg = C.config()
    run = C.run_dir(cfg)
    h0, h1 = A.load_scores(run)
    rows = []
    for arm, group in A.ARM_GROUP.items():
        for target in GRID:
            pds, pfas = [], []
            for fold in ("fold1", "fold2"):
                if (fold, "test_h0") not in h0 or (fold, group) not in h1:
                    continue
                h0s = np.sort(h0[(fold, "test_h0")][arm]["score"])[::-1]
                n0 = len(h0s)
                k = max(1, int(round(target * n0)))
                thr = h0s[k - 1] - 1e-15
                d = h1[(fold, group)]
                j = int(np.argmin(np.abs(d["scr_db"] - SCR)))
                y = d["scores"][arm][:, j] > thr
                pds.append(float(y.mean()))
                pfas.append(k / n0)
                rows.append({"arm": arm, "fold": fold, "requested_pfa": target,
                             "realised_pfa": k / n0, "n_test_h0": n0,
                             "threshold": float(thr), "pd": float(y.mean()),
                             "n_trials": int(len(y)), "n_detected": int(y.sum())})
            if pds:
                rows.append({"arm": arm, "fold": "pooled", "requested_pfa": target,
                             "realised_pfa": float(np.mean(pfas)),
                             "n_test_h0": int(sum(r["n_test_h0"] for r in rows
                                                  if r["arm"] == arm
                                                  and r["requested_pfa"] == target
                                                  and r["fold"] != "pooled")),
                             "threshold": float("nan"),
                             "pd": float(np.mean(pds)),
                             "n_trials": int(sum(r["n_trials"] for r in rows
                                                 if r["arm"] == arm
                                                 and r["requested_pfa"] == target
                                                 and r["fold"] != "pooled")),
                             "n_detected": int(sum(r["n_detected"] for r in rows
                                                   if r["arm"] == arm
                                                   and r["requested_pfa"] == target
                                                   and r["fold"] != "pooled"))})
    out = Path(__file__).resolve().parents[1] / "reports" / "MINIMAL_CAMPAIGN_MATCHED_PFA.csv"
    keys = list(rows[0])
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.10g}" if isinstance(r[k], float) else r[k]) for k in keys})
    rank = sorted({(r["arm"], r["pd"]) for r in rows
                   if r["fold"] == "pooled" and r["requested_pfa"] == 1e-3},
                  key=lambda t: -t[1])
    print(json.dumps({"ranking_at_realised_1e-3": rank}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
