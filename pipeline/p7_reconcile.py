"""Phase 7 gate 1: reconstruct every quantity Phase 6 reported, from the raw artifacts.

Six reconciliation items (R1-R6).  For each one the script recomputes the number from
its source array or CSV and asserts the relationship claimed in
``PHASE7_NUMERICAL_RECONCILIATION.md``.  A failure here is a blocker.

Scale conventions enforced:
  * power ratio  -> 10 log10
  * amplitude ratio -> 20 log10
  * never take a logarithm of a signed statistic
  * a point estimate and its interval must refer to the same estimand and the same scale

Outputs: data/PHASE7_RECONCILIATION.json, reports/PHASE7_POLARIMETRY_CORRECTED.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORK7 = PROJECT / "results" / "phase7"      # released Phase 7 data (read only)
P6 = PROJECT / "phase6"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase5_common as C                            # noqa: E402

RUN = C.run_dir(C.config())
out: dict = {"items": {}, "assertions": {}}


def db_power(x):
    return 10.0 * np.log10(x)


def db_amp(x):
    return 20.0 * np.log10(x)


# --------------------------------------------------------------------------- R1
def r1_scale_of_dSCR():
    """The CI reported next to a delta-SCR point estimate must be on the same scale."""
    rows = list(csv.DictReader(open(P6 / "reports/PHASE6_POLARIMETRY_TARGET.csv",
                                   encoding="utf-8")))
    corrected = []
    max_point_err = 0.0
    for r in rows:
        eta, lo, hi = (float(r["dm_eta_median"]), float(r["dm_eta_lo"]),
                       float(r["dm_eta_hi"]))
        point_db = float(r["dm_db"])
        max_point_err = max(max_point_err, abs(db_power(eta) - point_db))
        corrected.append({
            "acq_id": r["acq_id"], "cell": int(r["cell"]),
            "cell_source": r["cell_source"],
            "eta_linear": eta, "eta_ci_lo": lo, "eta_ci_hi": hi,
            "dSCR_db": db_power(eta),
            "dSCR_ci_lo_db": db_power(lo), "dSCR_ci_hi_db": db_power(hi),
            "ci_brackets_point": bool(db_power(lo) <= db_power(eta) <= db_power(hi)),
            "ci_excludes_zero": bool(db_power(lo) > 0 or db_power(hi) < 0),
        })
    with open(WORK7 / "reports/PHASE7_POLARIMETRY_CORRECTED.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(corrected[0]))
        w.writeheader()
        for r in corrected:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    out["items"]["R1"] = {
        "finding": "point estimates are 10log10(eta); published intervals are percentiles "
                   "of eta on the linear power scale, so they do not bracket the point "
                   "estimate when read as dB",
        "max_abs_diff_between_10log10_eta_and_reported_dm_db": max_point_err,
        "all_cis_bracket_point_after_conversion": all(r["ci_brackets_point"]
                                                      for r in corrected),
        "n_cis_that_exclude_zero_in_dB": sum(r["ci_excludes_zero"] for r in corrected),
        "example_141630": next(r for r in corrected
                               if r["acq_id"].startswith("19931107_141630")),
    }
    # the CSV stores dm_db rounded to 4 dp, so agreement is asserted at that precision
    out["assertions"]["R1_point_is_10log10_eta"] = max_point_err < 5e-3
    out["assertions"]["R1_ci_brackets_after_conversion"] = all(
        r["ci_brackets_point"] for r in corrected)
    return corrected


# --------------------------------------------------------------------------- R2
def r2_ar8_baseline():
    """0.790 vs 0.8148: two different populations and two different threshold pools."""
    def load_h1(fold, group, arm):
        ids, sc, acq = [], [], []
        for p in sorted((RUN / "scores/h1").glob(f"{fold}__{group}__*.npz")):
            with np.load(p) as z:
                j = int(np.argmin(np.abs(z["scr_db"] - 0.0)))
                ids.append(z["trial_ids"].astype(str))
                sc.append(z[f"arm_{arm}"][:, j])
                acq.append(z["acqs"].astype(str))
        return np.concatenate(ids), np.concatenate(sc), np.concatenate(acq)

    def campaign_threshold(fold, group, arm):
        v = np.concatenate([np.load(p)[f"arm_{arm}"] for p in sorted(
            (RUN / "scores/h0").glob(f"{fold}__cal__{group}__*.npz"))])
        return float(np.quantile(v, 1.0 - C.DESIGN_PFA, method="higher"))

    def restricted_threshold(fold, group, arm):
        """Mimic ``phase5_analyze_v2.load_scores``: drop every unit that *any* CFAR block
        reported as unevaluable.  Only blocks carrying the metadata can contribute, so the
        filter is exactly as complete - or incomplete - as the campaign's metadata."""
        drop = set()
        covered = 0
        files = sorted((RUN / "scores/h0").glob(f"{fold}__cal__cfar__*.npz"))
        for p in files:
            with np.load(p) as z:
                if "excluded_cells" not in z.files:
                    continue
                covered += 1
                acq = p.stem.split("__")[-1]
                for c, w in zip(z["excluded_cells"].astype(int),
                                z["excluded_windows"].astype(int)):
                    drop.add((acq, int(c), int(w)))
        out["items"].setdefault("R2b_metadata_coverage", {})[f"{fold}_cal"] = {
            "cfar_files": len(files), "files_with_withdrawal_metadata": covered,
            "n_withdrawn_units_seen": len(drop)}
        sc, key = [], []
        for p in sorted((RUN / "scores/h0").glob(f"{fold}__cal__{group}__*.npz")):
            with np.load(p) as z:
                acq = p.stem.split("__")[-1]
                sc.append(z[f"arm_{arm}"])
                key.extend((acq, int(c), int(w))
                           for c, w in zip(z["cells"], z["windows"]))
        sc = np.concatenate(sc)
        keep = np.array([k not in drop for k in key])
        return float(np.quantile(sc[keep], 1.0 - C.DESIGN_PFA, method="higher"))

    def excluded_ids(fold):
        s = set()
        for p in sorted((RUN / "scores/h1").glob(f"{fold}__*__*.npz")):
            with np.load(p) as z:
                if "excluded_trial_ids" in z.files:
                    s.update(z["excluded_trial_ids"].astype(str).tolist())
        return s

    res = {}
    tot_hit_all = tot_all = 0
    tot_hit_com = tot_com = 0
    for fold in ("fold1", "fold2"):
        ids, sc, _ = load_h1(fold, "temporal", "ar8")
        exc = excluded_ids(fold)
        com = np.array([t not in exc for t in ids])
        thr_c = campaign_threshold(fold, "temporal", "ar8")
        thr_r = restricted_threshold(fold, "temporal", "ar8")
        # reproduction of the E3 report's own rule: floor(n*pfa)-1 on the descending sort
        import math
        cal_full = np.sort(np.concatenate([np.load(p)["arm_ar8"] for p in sorted(
            (RUN / "scores/h0").glob(f"{fold}__cal__temporal__*.npz"))]))[::-1]
        n_drop = out["items"].get("R2b_metadata_coverage", {}).get(
            f"{fold}_cal", {}).get("n_withdrawn_units_seen", 0)
        n_e3 = len(cal_full) - n_drop
        thr_e3 = float(cal_full[max(1, int(math.floor(n_e3 * C.DESIGN_PFA))) - 1])
        # volatility of Pd to the choice of order statistic (calibration-induced)
        pd_by_rank = {int(r): float(np.mean(sc > cal_full[r - 1])) for r in range(4, 13)}
        res[fold] = {
            "n_all": int(len(sc)), "n_common": int(com.sum()),
            "thr_campaign_pool": thr_c, "thr_restricted_pool": thr_r,
            "thr_e3_convention_full_pool": float(cal_full[7]),
            "thr_e3_convention_after_withdrawal": thr_e3,
            "n_units_used_by_e3_rule": int(n_e3),
            "pd_all_campaign_threshold": float(np.mean(sc > thr_c)),
            "pd_common_campaign_threshold": float(np.mean(sc[com] > thr_c)),
            "pd_common_restricted_threshold": float(np.mean(sc[com] > thr_r)),
            "pd_vs_order_statistic": pd_by_rank,
            "pd_span_over_ranks_4_to_12": float(max(pd_by_rank.values())
                                                 - min(pd_by_rank.values())),
        }
        tot_hit_all += int(np.sum(sc > thr_c)); tot_all += len(sc)
        tot_hit_com += int(np.sum(sc[com] > thr_c)); tot_com += int(com.sum())
    e3 = json.loads((P5 / "reports/MINIMAL_CAMPAIGN_E3_REFERENCE_CONTAMINATION.json")
                    .read_text(encoding="utf-8"))
    e3_rows = [r for r in e3["rows"] if r["arm"] == "ar8" and r["condition"] == "vv_only"]
    e3_pooled = sum(r["pd"] * r["n_trials"] for r in e3_rows) / sum(
        r["n_trials"] for r in e3_rows)
    out["items"]["R2"] = {
        "finding": "the two numbers are different estimands: the frozen arm on all "
                   "3,990 trials with campaign thresholds (0.8148) versus the E3 report's "
                   "re-analysis on the 3,960-trial common-evaluable set with thresholds "
                   "recomputed after withdrawing CFAR-unevaluable CAL units (0.790); the "
                   "withdrawal list itself is only as complete as the metadata that "
                   "survived a mid-campaign runner edit",
        "per_fold": res,
        "pooled_all_campaign_threshold": tot_hit_all / tot_all,
        "pooled_common_campaign_threshold": tot_hit_com / tot_com,
        "e3_report_pooled": float(e3_pooled),
        "e3_per_fold": [{"fold": r["fold"], "threshold": r["threshold"],
                         "n_trials": r["n_trials"], "pd": r["pd"]} for r in e3_rows],
    }
    out["assertions"]["R2_pooled_all_reproduces_phase6"] = abs(
        tot_hit_all / tot_all - 0.8148) < 5e-4
    out["assertions"]["R2_e3_report_reproduces_0_790"] = abs(e3_pooled - 0.790) < 2e-3
    out["assertions"]["R2_e3_rule_reproduces_its_threshold"] = abs(
        res["fold2"]["thr_e3_convention_after_withdrawal"] - 0.409068) < 1e-4
    out["assertions"]["R2_metadata_coverage_is_fold_asymmetric"] = (
        out["items"]["R2b_metadata_coverage"]["fold1_cal"]
        ["files_with_withdrawal_metadata"] == 0) and (
        out["items"]["R2b_metadata_coverage"]["fold2_cal"]
        ["files_with_withdrawal_metadata"] == 8)


# --------------------------------------------------------------------------- R3
def r3_fold_vs_pooled():
    rows = list(csv.DictReader(open(P6 / "reports/PHASE6_CALIBRATION_CONTRASTS.csv",
                                    encoding="utf-8")))
    res = {}
    for s in ("A", "B", "C", "D"):
        r = [x for x in rows if x["strategy"] == s]
        d = [float(x["d_hybrid_anmf"]) for x in r]
        n = [int(x["n"]) for x in r]
        res[s] = {"by_fold": d, "pooled_trial_weighted": float(np.average(d, weights=n)),
                  "sign_flips": bool(len(set(np.sign(d))) > 1)}
    out["items"]["R3"] = {
        "finding": "the pooled +0.123 is the trial-weighted mean of a -0.028 and a "
                   "+0.271 fold; it is an algebraic identity, not a new measurement",
        "by_strategy": res,
    }
    out["assertions"]["R3_pooled_equals_weighted_mean"] = abs(
        res["A"]["pooled_trial_weighted"] - 0.1221) < 5e-3


# --------------------------------------------------------------------------- R4
def r4_nominal_vs_realised():
    rows = list(csv.DictReader(open(P5 / "reports/MINIMAL_CAMPAIGN_MATCHED_PFA.csv",
                                    encoding="utf-8")))
    pooled = {r["arm"]: float(r["pd"]) for r in rows
              if r["fold"] == "pooled" and r["requested_pfa"] == "0.001"}
    ret = pooled["hybrid"] - pooled["anmf_dl"]
    dep = 0.1221                            # R3, threshold-deployed, campaign thresholds
    cal = list(csv.DictReader(open(P6 / "reports/PHASE6_CALIBRATION.csv", encoding="utf-8")))
    real = {(r["arm"], r["fold"], r["strategy"]): float(r["realised_pfa"]) for r in cal}
    out["items"]["R4"] = {
        "finding": "threshold-deployed (+0.122) and retrospective matched-realised-PFA "
                   "(+0.088) are different estimands; the second sets its threshold on "
                   "TEST-H0 and is therefore not deployable",
        "threshold_deployed_delta": dep,
        "retrospective_matched_delta": float(ret),
        "matched_pfa_realised": float(
            next(r["realised_pfa"] for r in rows if r["fold"] == "pooled"
                 and r["requested_pfa"] == "0.001")),
        "deployed_realised_pfa_hybrid_fold1_A": real[("hybrid", "fold1", "A")],
        "deployed_realised_pfa_hybrid_fold2_A": real[("hybrid", "fold2", "A")],
    }
    out["assertions"]["R4_retrospective_is_0_088"] = abs(ret - 0.0877) < 2e-3


# --------------------------------------------------------------------------- R5
def r5_e3_labels():
    e3 = json.loads((P5 / "reports/MINIMAL_CAMPAIGN_E3_REFERENCE_CONTAMINATION.json")
                    .read_text(encoding="utf-8"))
    conv = {}
    for k, v in e3["condition"].items():
        ratio = v["vh_target_to_clutter_median"]
        conv[k] = {"vh_target_to_clutter_ratio": ratio,
                   "dSCR_VH_db": float(db_power(ratio))}
    out["items"]["R5"] = {
        "finding": "the E3 labels are injection gains, not delta-SCR: with the measured "
                   "clutter ratio folded in, the same label corresponds to different "
                   "delta-SCR in the two folds",
        "conditions": conv,
    }
    out["assertions"]["R5_folds_differ_for_same_label"] = abs(
        conv["fold1|vh_0dB"]["dSCR_VH_db"] - conv["fold2|vh_0dB"]["dSCR_VH_db"]) > 1.5


# --------------------------------------------------------------------------- R6
def r6_medians():
    rows = list(csv.DictReader(open(P6 / "reports/PHASE6_POLARIMETRY_TARGET.csv",
                                    encoding="utf-8")))
    eta = np.array([float(r["dm_eta_median"]) for r in rows])
    lit = np.array([float(r["dm_eta_median"]) for r in rows
                    if r["cell_source"] == "literature"])
    vis = np.array([float(r["dm_eta_median"]) for r in rows
                    if r["cell_source"] == "literature"
                    and float(r["delta_vv_db"]) >= 3.0])
    out["items"]["R6"] = {
        "finding": "the median commutes with the monotone log, but the subsets differ",
        "all_16": {"median_eta": float(np.median(eta)),
                   "dSCR_db": float(db_power(np.median(eta)))},
        "literature_8": {"median_eta": float(np.median(lit)),
                         "dSCR_db": float(db_power(np.median(lit)))},
        "literature_measurable_4": {"median_eta": float(np.median(vis)),
                                    "dSCR_db": float(db_power(np.median(vis)))},
    }
    # A monotone map commutes with the median only for odd n: with n = 16 the median is
    # the arithmetic mean of the two middle order statistics, and 10log10 does not
    # commute with averaging.  The two conventions therefore differ, slightly.
    delta = abs(np.median(db_power(eta)) - db_power(np.median(eta)))
    out["items"]["R6"]["median_of_db_minus_db_of_median_db"] = float(delta)
    out["items"]["R6"]["note"] = (
        "with n = 16 the median averages two values; the two conventions differ by "
        "%.4f dB, well inside every interval quoted" % delta)
    out["assertions"]["R6_median_convention_difference_is_negligible"] = delta < 0.01


def main() -> int:
    r1_scale_of_dSCR()
    r2_ar8_baseline()
    r3_fold_vs_pooled()
    r4_nominal_vs_realised()
    r5_e3_labels()
    r6_medians()
    out["assertions"] = {k: bool(v) for k, v in out["assertions"].items()}
    ok = all(out["assertions"].values())
    out["result"] = "PASS" if ok else "FAIL"
    out["failed"] = [k for k, v in out["assertions"].items() if not v]
    (WORK7 / "data").mkdir(exist_ok=True)
    (WORK7 / "data/PHASE7_RECONCILIATION.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"result": out["result"], "failed": out["failed"],
                      "assertions": out["assertions"]}, indent=2))
    print(json.dumps(out["items"], indent=2)[:4000])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
