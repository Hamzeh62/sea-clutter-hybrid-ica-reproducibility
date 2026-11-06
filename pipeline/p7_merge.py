"""Merge the two per-fold V3 records into one table, with the contrast against the clean G0 arm.

Both folds overwrite the same CSV when the runner finishes, so the merged table is rebuilt from
the per-fold JSON records rather than from the runner's last write.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORK7 = PROJECT / "results" / "phase7"      # released Phase 7 data (read only)
ORDER = ("deployed_g0", "deployed_g1", "target_channel", "delay1",
         "temporal_oracle", "delay512")


def main() -> int:
    rows = []
    for fold in ("fold1", "fold2"):
        d = json.loads((WORK7 / f"data/PHASE7_V3_{fold}.json").read_text())
        by = {r["variant"]: r for r in d["rows"]}
        for k in ORDER:
            r = by[k]
            c = d["contrasts"].get(f"g0_minus_{k}")
            rows.append({
                "fold": fold, "variant": k, "n_cal": r["n_cal"],
                "n_trial": r["n_trial"], "threshold": r["threshold"],
                "pd": r["pd_0db"], "cal_pfa_at_own_thr": r["cal_pfa_at_own_thr"],
                "delta_pd_vs_g0": c["delta_pd"] if c else 0.0,
                "ci_lo": c["ci_lo"] if c else 0.0,
                "ci_hi": c["ci_hi"] if c else 0.0,
                "mcnemar_p": c["mcnemar_p"] if c else 1.0,
                "n_acq_positive": c["n_acq_positive"] if c else "-",
                "n_acq": c["n_acq"] if c else "-",
            })
    out = PROJECT / "outputs" / "phase7" / "reports" / "PHASE7_V3_REFERENCE_RESULTS.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[merge] {len(rows)} rows -> {out}")
    for r in rows:
        print(f"{r['fold']} {r['variant']:16s} pd={r['pd']:.3f} "
              f"d={r['delta_pd_vs_g0']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
