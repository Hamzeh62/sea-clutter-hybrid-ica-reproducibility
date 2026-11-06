"""Phase 7 V1 analysis: the polarimetric mechanism curve, deployment and matched operating points.

Two operating points are reported for every condition, and they answer different questions:

``deployment``
    the frozen CAL-derived threshold.  It is what a fielded detector would use, so it carries the
    calibration error measured in V2.  All arms share one condition-invariant threshold because
    H0 units carry no target, so only the signal-present distribution changes along the curve.

``matched``
    a common retrospective Pfa read off the TEST-H0 scores of the same fold.  It removes
    threshold-placement error and compares *separability*; it is not deployable and is labelled
    as such everywhere it appears.  Two levels are reported: 5e-3 (20 expected exceedances in
    4000 H0 units, the better-resolved primary) and the design 1e-3 (4 expected, a sensitivity).

The ANMF-DL statistic never reads the cross-polarised channel, so its row is exact at every
dSCR and is not recomputed (deviation D8 in the V1 record).

Outputs
  reports/PHASE7_V1_POLARIMETRIC_RESULTS.csv   deployment rows
  reports/PHASE7_V1_MATCHED.csv                matched rows
  reports/PHASE7_V1_CONTRASTS.csv              paired contrasts, all modes
  data/PHASE7_ANALYSIS.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
WORK7 = PROJECT / "results" / "phase7"      # released Phase 7 data (read only)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase5_common as C                            # noqa: E402

CFG = C.config()
RUN = C.run_dir(CFG)
V1 = PROJECT / "outputs" / "phase7" / "data" / "v1"
RNG = np.random.default_rng(20261101)
BOOT = 20000
ARMS = ("hybrid", "ar8", "notch")
MATCHED_PFAS = (5e-3, 1e-3)
GROUP = {"hybrid": "hybrid", "ar8": "temporal", "notch": "temporal", "anmf_dl": "anmf"}
KEY = {"hybrid": "hybrid", "ar8": "ar8", "notch": "notch", "anmf_dl": "anmf_dl"}


def mcnemar(h: np.ndarray, r: np.ndarray) -> tuple[float, int, int]:
    n01 = int(np.sum(h & ~r)); n10 = int(np.sum(~h & r))
    p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return p, n01, n10


def cluster_ci(per_acq: list[float]) -> tuple[float, float, float]:
    b = np.array([np.mean(RNG.choice(per_acq, len(per_acq), replace=True))
                  for _ in range(BOOT)])
    return float(np.mean(per_acq)), float(np.percentile(b, 2.5)), float(
        np.percentile(b, 97.5))


def h0_scores(fold: str, part: str, arm: str) -> np.ndarray:
    return np.concatenate([np.load(p)[f"arm_{KEY[arm]}"] for p in sorted(
        (RUN / "scores/h0").glob(f"{fold}__{part}__{GROUP[arm]}__*.npz"))])


def frozen() -> dict:
    """Raw H1 scores and identities of the frozen campaign at SCR = 0 dB."""
    out = {}
    for fold in ("fold1", "fold2"):
        for arm in (*ARMS, "anmf_dl"):
            ids, sc, acq = [], [], []
            for p in sorted((RUN / "scores/h1").glob(f"{fold}__{GROUP[arm]}__*.npz")):
                with np.load(p) as z:
                    j = int(np.argmin(np.abs(z["scr_db"] - 0.0)))
                    ids.append(z["trial_ids"].astype(str))
                    sc.append(z[f"arm_{KEY[arm]}"][:, j])
                    acq.append(z["acqs"].astype(str))
            out[(fold, arm)] = {"ids": np.concatenate(ids),
                                "score": np.concatenate(sc),
                                "acq": np.concatenate(acq)}
    return out


def load_cond(fold: str, tag: str) -> dict | None:
    p = V1 / f"{fold}__{tag}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    d["thr"] = dict(zip(d["threshold_arms"].astype(str), d["thresholds"]))
    return d


def dump(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})


def main() -> int:
    fz = frozen()

    # ---- operating points: deployment from CAL, matched from TEST-H0 (retrospective)
    thr: dict[tuple[str, str, str], float] = {}
    for fold in ("fold1", "fold2"):
        for arm in (*ARMS, "anmf_dl"):
            thr[(fold, arm, "deployment")] = float(np.quantile(
                h0_scores(fold, "cal", arm), 1.0 - C.DESIGN_PFA, method="higher"))
            for pf in MATCHED_PFAS:
                thr[(fold, arm, f"matched_{pf:g}")] = float(np.quantile(
                    h0_scores(fold, "test_h0", arm), 1.0 - pf, method="higher"))
    modes = ("deployment",) + tuple(f"matched_{pf:g}" for pf in MATCHED_PFAS)

    deploy_rows, match_rows, contrasts = [], [], []
    for fold in ("fold1", "fold2"):
        for arm in (*ARMS, "anmf_dl"):
            d = fz[(fold, arm)]
            for mode in modes:
                h = d["score"] > thr[(fold, arm, mode)]
                row = {"fold": fold, "condition": "vh_absent", "dSCR_dB": "-inf",
                       "phi": "-", "arm": arm, "mode": mode, "n": int(len(h)),
                       "pd": float(h.mean()), "threshold": thr[(fold, arm, mode)],
                       "realised_test_pfa": float(np.mean(
                           h0_scores(fold, "test_h0", arm) > thr[(fold, arm, mode)])),
                       "source": "frozen campaign"}
                (deploy_rows if mode == "deployment" else match_rows).append(row)

    for fold in ("fold1", "fold2"):
        for tag in sorted({p.stem.split("__", 1)[1] for p in V1.glob(f"{fold}__*.npz")}):
            if "smoke" in tag or "probe" in tag:
                continue
            d = load_cond(fold, tag)
            if d is None:
                continue
            ds = float(tag.split("__")[0].replace("dSCR", ""))
            phi = tag.split("__")[1]
            for arm in d["threshold_arms"].astype(str):
                sc = d[f"arm_{arm}"]
                for mode in modes:
                    t = d["thr"][arm] if mode == "deployment" else thr[(fold, arm, mode)]
                    row = {"fold": fold, "condition": tag, "dSCR_dB": f"{ds:g}",
                           "phi": phi, "arm": arm, "mode": mode, "n": int(len(sc)),
                           "pd": float(np.mean(sc > t)), "threshold": t,
                           "realised_test_pfa": float(np.mean(
                               h0_scores(fold, "test_h0", arm) > t)),
                           "source": "V1"}
                    (deploy_rows if mode == "deployment" else match_rows).append(row)
                if phi == "randphi":
                    fz[(fold, ds, arm)] = {"score": sc, "acq": d["acqs"].astype(str),
                                           "ids": d["trial_ids"].astype(str)}

    def cell(fold: str, ds: float, arm: str) -> dict:
        # the frozen arrays serve both the VH-absent row and the VH-invariant ANMF arm
        if ds == -np.inf or arm == "anmf_dl":
            return fz[(fold, arm)]
        return fz[(fold, ds, arm)]

    for mode in modes:
        for fold in ("fold1", "fold2"):
            for ds in (-np.inf, -20.0, -10.0, 0.0):
                hd = cell(fold, ds, "hybrid")
                acq = hd["acq"]
                h = hd["score"] > thr[(fold, "hybrid", mode)]
                for other in ("ar8", "notch", "anmf_dl"):
                    od = cell(fold, ds, other)
                    if not np.array_equal(od["ids"], hd["ids"]):
                        same = np.isin(hd["ids"], od["ids"]).all()
                        if not same:
                            print(f"[skip] {fold} {ds} hybrid-{other}: trial sets differ "
                                  f"({len(hd['ids'])} vs {len(od['ids'])})")
                            continue
                        pos = {t: i for i, t in enumerate(od["ids"])}
                        od = {"score": od["score"][[pos[t] for t in hd["ids"]]],
                              "acq": od["acq"][[pos[t] for t in hd["ids"]]],
                              "ids": hd["ids"]}
                    r = od["score"] > thr[(fold, other, mode)]
                    p, n01, n10 = mcnemar(h, r)
                    per = [float(h[acq == a].mean() - r[acq == a].mean())
                           for a in sorted(set(acq))]
                    _m, lo, hi = cluster_ci(per)
                    contrasts.append({
                        "mode": mode, "fold": fold,
                        "dSCR_dB": "-inf" if ds == -np.inf else f"{ds:g}",
                        "contrast": f"hybrid-{other}", "n": int(len(h)),
                        "delta_pd": float(h.mean() - r.mean()),
                        "ci_lo": lo, "ci_hi": hi, "mcnemar_p": p,
                        "a_only": n01, "b_only": n10,
                        "n_acq_positive": int(sum(v > 0 for v in per)),
                        "n_acq": len(per)})

    # Holm within each operating-point family (2 folds x 4 conditions x 3 contrasts = 24)
    for mode in modes:
        fam = [c for c in contrasts if c["mode"] == mode]
        order = sorted(range(len(fam)), key=lambda i: fam[i]["mcnemar_p"])
        m = len(fam)
        running = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, (m - rank) * fam[i]["mcnemar_p"])
            running = max(running, adj)
            fam[i]["p_holm"] = running
        for c in fam:
            c.setdefault("p_holm", 1.0)

    dump(WORK7 / "reports/PHASE7_V1_POLARIMETRIC_RESULTS.csv", deploy_rows)
    dump(WORK7 / "reports/PHASE7_V1_MATCHED.csv", match_rows)
    dump(WORK7 / "reports/PHASE7_V1_CONTRASTS.csv", contrasts)

    summary = {"design_pfa": C.DESIGN_PFA, "matched_pfas": list(MATCHED_PFAS),
               "thresholds": {f"{k[0]}|{k[1]}": {"deployment": thr[(k[0], k[1], "deployment")],
                                                 "matched_5e-3": thr[(k[0], k[1], "matched_0.005")],
                                                 "matched_1e-3": thr[(k[0], k[1], "matched_0.001")]}
                              for k in thr if k[2] == "deployment"},
               "contrasts": contrasts}
    (WORK7 / "data/PHASE7_ANALYSIS.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")

    for mode in modes:
        print(f"\n=== {mode} ===")
        for r in contrasts:
            if r["mode"] != mode:
                continue
            print(f"{r['fold']} dSCR={r['dSCR_dB']:>5} {r['contrast']:15s} "
                  f"d={r['delta_pd']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                  f"acq+={r['n_acq_positive']}/{r['n_acq']} only={r['a_only']}/{r['b_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
