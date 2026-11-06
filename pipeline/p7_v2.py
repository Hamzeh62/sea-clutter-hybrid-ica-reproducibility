"""Phase 7 V2: calibration transfer, threshold uncertainty and acquisition clustering.

All thresholds come from CAL only; TEST-H0 is used solely to *score* the realised rate.
Four rules, as preregistered:

  A pooled          one empirical quantile of the arm's whole CAL-H0 sample
  B stratified      per-acquisition CAL quantile, median across acquisitions
  C robust tail     generalised-Pareto (peaks-over-threshold) extrapolation
  D within-acq      pooled standardised quantile with each acquisition's own
                    location/scale from its CAL segment in the opposite fold

Reported, per detector x acquisition x fold: the realised PFA, the threshold, whether the
acquisition owns any of the top-k CAL scores (tail clustering), and the leave-one-
acquisition-out transfer ratio.  Plus a bootstrap of the CAL pool giving the threshold
distribution and the implied TEST-H0 rate distribution.

Outputs: reports/PHASE7_V2_CALIBRATION_RESULTS.csv,
         data/PHASE7_V2_DETAIL.csv, data/PHASE7_V2.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORK7 = PROJECT / "results" / "phase7"      # released Phase 7 data (read only)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase5_common as C                            # noqa: E402

CFG = C.config()
RUN = C.run_dir(CFG)
DESIGN = C.DESIGN_PFA
RNG = np.random.default_rng(20261031)

ARMS = (("hybrid", "hybrid", "hybrid"), ("anmf_dl", "anmf", "anmf_dl"),
        ("ar8", "temporal", "ar8"), ("notch", "temporal", "notch"),
        ("ca", "cfar", "ca"), ("os", "cfar", "os"))


def load(fold: str, part: str, group: str, arm: str) -> dict[str, np.ndarray]:
    out = {}
    for p in sorted((RUN / "scores/h0").glob(f"{fold}__{part}__{group}__*.npz")):
        acq = p.stem.split("__")[-1]
        with np.load(p) as z:
            out[acq] = z[f"arm_{arm}"].astype(float)
    return out


def q(x: np.ndarray, pfa: float = DESIGN) -> float:
    return float(np.quantile(x, 1.0 - pfa, method="higher"))


def gpd_thr(x: np.ndarray, pfa: float = DESIGN, tail: float = 0.05) -> float:
    from scipy.stats import genpareto
    u = float(np.quantile(x, 1.0 - tail))
    exc = x[x > u] - u
    rate = len(exc) / len(x)
    target = pfa / rate
    if len(exc) < 30 or target >= 1:
        return q(x, pfa)
    try:
        xi, _, beta = genpareto.fit(exc, floc=0.0)
        if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0):
            raise ValueError
        return float(u + (beta / xi) * (target ** (-xi) - 1.0) if abs(xi) > 1e-9
                     else u - beta * np.log(target))
    except Exception:
        return float(u - exc.mean() * np.log(target))


def thresholds(rule: str, cal: dict, cal_o: dict, targets: list[str]) -> dict[str, float]:
    allv = np.concatenate(list(cal.values()))
    if rule == "A":
        return {a: q(allv) for a in targets}
    if rule == "B":
        t = float(np.median([q(v) for v in cal.values()]))
        return {a: t for a in targets}
    if rule == "C":
        t = gpd_thr(allv)
        return {a: t for a in targets}
    if rule == "D":
        med = {a: float(np.median(v)) for a, v in cal.items()}
        iqr = {a: max(float(np.subtract(*np.percentile(v, [75, 25]))), 1e-12)
               for a, v in cal.items()}
        z = np.concatenate([(v - med[a]) / iqr[a] for a, v in cal.items()])
        tz = q(z)
        out = {}
        for a in targets:
            if a in cal_o:
                mo = float(np.median(cal_o[a]))
                so = max(float(np.subtract(*np.percentile(cal_o[a], [75, 25]))), 1e-12)
                out[a] = mo + so * tz
            else:
                out[a] = q(allv)
        return out
    raise ValueError(rule)


def main() -> int:
    rows, detail, summary = [], [], {"design_pfa": DESIGN, "rules": {}}
    for arm, group, armkey in ARMS:
        for fold in ("fold1", "fold2"):
            other = "fold2" if fold == "fold1" else "fold1"
            cal = load(fold, "cal", group, armkey)
            te = load(fold, "test_h0", group, armkey)
            cal_o = load(other, "cal", group, armkey)
            allv = np.concatenate(list(cal.values()))
            top = np.sort(allv)[::-1]
            # which acquisitions own the 8 scores that will set a 1e-3 threshold
            th8 = top[7]
            owners = {a: int(np.sum(v >= th8)) for a, v in cal.items()}
            for rule in ("A", "B", "C", "D"):
                thr = thresholds(rule, cal, cal_o, list(te))
                flags = {a: v > thr[a] for a, v in te.items()}
                n_tot = sum(len(v) for v in te.values())
                n_hit = sum(int(f.sum()) for f in flags.values())
                pooled = n_hit / n_tot
                worst = max((float(f.mean()), a) for a, f in flags.items())
                rows.append({
                    "arm": arm, "fold": fold, "rule": rule,
                    "n_cal": int(len(allv)), "n_test_h0": int(n_tot),
                    "threshold_median": float(np.median(list(thr.values()))),
                    "realised_pfa": pooled, "realised_over_design": pooled / DESIGN,
                    "n_exceedances": int(n_hit),
                    "worst_acq": worst[1], "worst_acq_pfa": worst[0],
                    "worst_acq_over_design": worst[0] / DESIGN,
                    "acqs_over_2x": int(sum(1 for f in flags.values() if f.mean() > 2 * DESIGN)),
                })
                # leave-one-acquisition-out transfer inside CAL
                loo = []
                for held in sorted(cal):
                    sub = {a: v for a, v in cal.items() if a != held}
                    t = thresholds(rule, sub, cal_o, [held])[held]
                    loo.append(float(np.mean(cal[held] > t)))
                detail.append({
                    "arm": arm, "fold": fold, "rule": rule,
                    "loo_median": float(np.median(loo)),
                    "loo_min": float(np.min(loo)), "loo_max": float(np.max(loo)),
                    "loo_over_design_median": float(np.median(loo)) / DESIGN,
                    "tail_owners": ";".join(f"{a}:{c}" for a, c in owners.items() if c),
                    "n_tail_owners": int(sum(1 for c in owners.values() if c)),
                })
                # bootstrap the CAL pool -> threshold distribution and implied rate
                boot_t, boot_p = [], []
                for _ in range(400):
                    pick = RNG.choice(list(cal), len(cal), replace=True)
                    samp = np.concatenate([cal[a] for a in pick])
                    t = q(samp)
                    boot_t.append(t)
                    boot_p.append(float(np.mean(np.concatenate(
                        list(te.values())) > t)))
                detail[-1].update({
                    "boot_thr_lo": float(np.percentile(boot_t, 2.5)),
                    "boot_thr_hi": float(np.percentile(boot_t, 97.5)),
                    "boot_rate_lo": float(np.percentile(boot_p, 2.5)),
                    "boot_rate_hi": float(np.percentile(boot_p, 97.5)),
                })
        print(f"{arm} done", flush=True)

    with open(WORK7 / "reports/PHASE7_V2_CALIBRATION_RESULTS.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    with open(WORK7 / "data/PHASE7_V2_DETAIL.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail[0]))
        w.writeheader()
        for r in detail:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    for rule in ("A", "B", "C", "D"):
        sub = [r for r in rows if r["rule"] == rule]
        summary["rules"][rule] = {
            "worst_realised_over_design": max(r["realised_over_design"] for r in sub),
            "median_realised_over_design": float(np.median(
                [r["realised_over_design"] for r in sub])),
            "worst_acq_over_design": max(r["worst_acq_over_design"] for r in sub),
            "n_cases_over_2x": int(sum(r["worst_acq_over_design"] > 2 for r in sub)),
            "loo_median_over_design": float(np.median(
                [d["loo_over_design_median"] for d in detail if d["rule"] == rule])),
            "n_tail_owners_median": float(np.median(
                [d["n_tail_owners"] for d in detail if d["rule"] == rule])),
        }
    (WORK7 / "data/PHASE7_V2.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
