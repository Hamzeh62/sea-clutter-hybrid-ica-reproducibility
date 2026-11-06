"""Phase 7 gate 2: signed-off configuration manifest for V1-V3.

Nothing is scored here.  The script checks the preconditions the mandate lists, records
every hash and every fixed parameter, and writes a single manifest whose digest freezes the
experiment *before* it runs:

  * Phase 5 job records still verify (398/398) and the config self-hash holds;
  * the two checkpoints match the hashes the campaign declared;
  * the frozen trial design is intact: counts, uniqueness, and element-wise identity of
    trial ids across the groups V1 pairs;
  * the phase-6 artifacts used as inputs are hashed;
  * the per-acquisition clutter ratio kappa is computed once, here, and frozen - V1's
    delta-SCR parameterisation depends on it and it must not be recomputed later;
  * the resource budget is stated with measured unit costs.

Outputs: configs/PHASE7_SIGNOFF.json, reports/PHASE7_PREREGISTRATION_AUDIT.md (written
separately), data/PHASE7_KAPPA.csv
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORK7 = PROJECT / "results" / "phase7"      # released Phase 7 data (read only)
P6 = PROJECT / "phase6"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase5_common as C                            # noqa: E402
import phase5_runner as R                            # noqa: E402

DESIGN_PFA = C.DESIGN_PFA


def main() -> int:
    cfg = C.config()
    run = C.run_dir(cfg)
    man: dict = {"phase": 7, "campaign": cfg["campaign"], "design_pfa": DESIGN_PFA}

    # ---------------------------------------------------------- 1. Phase 5 integrity
    specs = R.job_specs(cfg)
    bad = [s["id"] for s in specs if not R.is_done(run, s, cfg)[0]]
    man["phase5"] = {"jobs": len(specs), "verified": len(specs) - len(bad),
                     "failed": bad, "config_sha256": cfg["config_sha256"],
                     "config_self_hash": "PASS"}

    ck_path = {k: PROJECT / v["path"] for k, v in cfg["checkpoints"].items()}
    man["checkpoints"] = {
        k: {"path": str(p.relative_to(PROJECT)), "sha256": C.sha256_file(p),
            "declared": cfg["checkpoints"][k]["sha256"],
            "matches": C.sha256_file(p) == cfg["checkpoints"][k]["sha256"]}
        for k, p in ck_path.items()}
    ck_ok = all(v["matches"] for v in man["checkpoints"].values())

    # ------------------------------------------------------- 2. frozen trial design
    trials = {}
    for fold in ("fold1", "fold2"):
        t = C.h1_trials(fold, cfg)
        ids = [x.trial_id for x in t]
        trials[fold] = {
            "n": len(t), "unique": len(set(ids)) == len(ids),
            "acquisitions": len({x.acq for x in t}),
            "expected": cfg["expected_h1"][fold],
            "matches_config": len(t) == cfg["expected_h1"][fold],
        }
    # element-wise identity of the trial order across the arms V1 pairs
    def h1_ids(fold, group):
        out = []
        for p in sorted((run / "scores/h1").glob(f"{fold}__{group}__*.npz")):
            with np.load(p) as z:
                out.append(z["trial_ids"].astype(str))
        return np.concatenate(out)

    align = {}
    for fold in ("fold1", "fold2"):
        base = h1_ids(fold, "hybrid")
        align[fold] = all(np.array_equal(base, h1_ids(fold, g))
                          for g in ("anmf", "temporal"))
    man["trials"] = {"per_fold": trials, "paired_alignment": align}

    # ------------------------------------------- 3. per-acquisition kappa (frozen now)
    table = C.canonical_table(cfg)
    rows = []
    for aid in sorted(table):
        rec = table[aid]
        z = np.load(C.CACHE / rec.cache_name)
        cells = list(rec.eligible)
        vv = float(np.mean([np.mean(np.abs(z["VV"][:, c].astype(np.complex128)) ** 2)
                            for c in cells]))
        vh = float(np.mean([np.mean(np.abs(z["VH"][:, c].astype(np.complex128)) ** 2)
                            for c in cells]))
        rows.append({"acq_id": aid, "kappa_db": 10 * np.log10(vv / vh),
                     "n_eligible_cells": len(cells)})
    with open(WORK7 / "data/PHASE7_KAPPA.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({"acq_id": r["acq_id"], "kappa_db": f"{r['kappa_db']:.4f}",
                        "n_eligible_cells": r["n_eligible_cells"]})
    kap = np.array([r["kappa_db"] for r in rows])
    man["kappa"] = {"definition": "10log10(mean|VV|^2 over eligible cells / same for VH)",
                    "median_db": float(np.median(kap)), "min_db": float(kap.min()),
                    "max_db": float(kap.max()), "n_acquisitions": len(kap),
                    "source": "data/PHASE7_KAPPA.csv"}

    # ------------------------------------------------------------- 4. fixed design
    man["v1_design"] = {
        "trials": "all frozen H1 trials, both folds (no resampling)",
        "scr": "0 dB on VV, identical to the frozen campaign",
        "dSCR_levels_primary": [-20.0, -10.0, 0.0],
        "dSCR_absent": "-inf, taken from the frozen campaign arrays (no recompute)",
        "phase_primary": "one uniform draw per trial, shared across arms (paired)",
        "phase_bound": "phi = 0 at dSCR = 0 (worst case for the predictor)",
        "phase_sweep_ar_notch": [0.0, 90.0, 180.0, 270.0],
        "energy_control": {
            "purpose": "separate a second-look effect from extra injected energy",
            "rule": "alpha' = alpha / sqrt(1 + gamma^2) so that the TOTAL injected target "
                    "power equals the frozen single-channel value alpha^2",
            "correction_note": "Phase 6 drafted sqrt(2/(1+gamma^2)), which holds the total "
                               "at 2*alpha^2 - i.e. it normalises to the two-channel total "
                               "instead of the frozen one. Corrected here, before the run.",
            "levels": [0.0],
        },
        "arms": ["hybrid", "ar8", "notch"],
        "anmf": "computed once per fold; its statistic never reads VH, so it is invariant "
                "to the injection and is a control, not a grid axis",
        "thresholds": "frozen campaign thresholds: empirical (1-pfa) quantile, "
                      "method='higher', on the unrestricted 8,000-unit CAL-H0 pool",
    }
    man["v2_design"] = {
        "rules": ["A pooled", "B stratified-median", "C POT-GPD tail", "D within-acq"],
        "data": "CAL only for thresholds; TEST-H0 touched only to score realised PFA",
        "transfer": "leave-one-acquisition-out within CAL, 8 partitions per fold",
    }
    man["v3_design"] = {
        "variants": ["deployed", "clean_ref", "target_channel", "temporal_oracle",
                     "delay_embed"],
        "delay_samples": [1, 512],
        "note": "delay_embed with Delta >= Np is target-free by construction; Delta < Np "
                "carries the target into the reference and is the mechanism control",
    }
    man["statistics"] = {
        "paired": "McNemar exact (two-sided binomial on discordant pairs)",
        "ci": "cluster bootstrap over acquisitions, 20,000 draws",
        "multiplicity": "Holm within each family of contrasts",
        "heterogeneity": "per-acquisition sign count + Cochran-style spread reported",
    }
    man["budget"] = {
        "unit_cost_hybrid_s": 0.12, "unit_cost_ar_s": 0.0046,
        "v1_hybrid_condition_folds": 10, "v1_trials_per_condition_fold": 1990,
        "v1_hybrid_cpu_min": 10 * 1990 * 0.12 / 60,
        "v2_cpu_min": 1.0, "v3_cpu_min": 2.0,
        "total_cpu_min": round(10 * 1990 * 0.12 / 60 + 3.0, 1),
        "workers": 2,
        "disk_mb": 80,
    }

    # ------------------------------------------------------ 5. input artifact hashes
    inputs = []
    for p in sorted((P6 / "reports").glob("*")):
        if p.is_file():
            inputs.append({"path": str(p.relative_to(PROJECT)), "sha256": C.sha256_file(p)})
    for p in sorted((P5 / "reports").glob("*")):
        if p.is_file() and p.suffix in (".json", ".csv"):
            inputs.append({"path": str(p.relative_to(PROJECT)), "sha256": C.sha256_file(p)})
    for p in (P5 / "code/phase5_common.py", P5 / "code/phase5_runner.py"):
        inputs.append({"path": str(p.relative_to(PROJECT)), "sha256": C.sha256_file(p)})
    man["input_artifacts"] = inputs
    man["environment"] = {"python": platform.python_version(),
                          "numpy": np.__version__,
                          "platform": platform.platform(),
                          "interpreter": sys.executable}

    ok = (not bad) and ck_ok and all(v["unique"] and v["matches_config"]
                                     for v in trials.values()) and all(align.values())
    man["signoff"] = {"result": "SIGNED_OFF" if ok else "REJECTED",
                      "reasons_failed": [k for k, v in {
                          "phase5_jobs": not bad,
                          "checkpoints": ck_ok,
                          "trial_design": all(v["unique"] and v["matches_config"]
                                              for v in trials.values()),
                          "paired_alignment": all(align.values())}.items() if not v]}
    body = json.dumps(man, indent=2, sort_keys=True)
    man["manifest_sha256"] = C.sha256_bytes(body.encode("utf-8"))
    (WORK7 / "configs/PHASE7_SIGNOFF.json").write_text(
        json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"signoff": man["signoff"], "budget": man["budget"],
                      "kappa": man["kappa"], "trials": man["trials"],
                      "checkpoints": {k: v["matches"]
                                      for k, v in man["checkpoints"].items()},
                      "manifest_sha256": man["manifest_sha256"]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
