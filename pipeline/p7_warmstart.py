"""Phase 7: final consistency check of the warm-start and physics-prior claims (saved evidence).

Uses the frozen Phase 5 arrays only; no network is retrained or re-run.  Each contrast is paired
by trial identity at SCR = 0 dB and reported at the deployment threshold (that group's own CAL
quantile) and at a matched retrospective operating point (common TEST-H0 rate of the same fold).

Groups: ``hybrid`` (warm), ``cold`` (same architecture, cold ICA init), ``ica_only`` (no
network), ``phys`` (physics-prior training), ``nophys`` (no physics prior).
"""
from __future__ import annotations

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
RNG = np.random.default_rng(20261102)
BOOT = 20000
MATCH_PFA = 5e-3
# (group_a, arm_a, group_b, arm_b).  The physics prior was never run as its own group in the
# frozen campaign; ``nophys`` is the no-prior variant of the same architecture, so the physics
# contrast is available only as the archived Phase 5 nominal figure and is not recomputed here.
PAIRS = (("hybrid", "hybrid", "cold", "cold"), ("hybrid", "hybrid", "ica", "ica_only"),
         ("hybrid", "hybrid", "nophys", "nophys"))


def scores(fold: str, part: str, group: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    ids, sc = [], []
    for p in sorted((RUN / "scores/h0").glob(f"{fold}__{part}__{group}__*.npz")):
        with np.load(p) as z:
            ids.append(z["cells"].astype(int).astype(str) + "_" +
                       z["windows"].astype(int).astype(str))
            sc.append(z[f"arm_{arm}"])
    return np.concatenate(sc), np.concatenate(ids)


def h1(fold: str, group: str, arm: str) -> dict:
    ids, sc, acq = [], [], []
    for p in sorted((RUN / "scores/h1").glob(f"{fold}__{group}__c*.npz")):
        with np.load(p) as z:
            j = int(np.argmin(np.abs(z["scr_db"] - 0.0)))
            ids.append(z["trial_ids"].astype(str))
            sc.append(z[f"arm_{arm}"][:, j])
            acq.append(z["acqs"].astype(str))
    return {"ids": np.concatenate(ids), "score": np.concatenate(sc),
            "acq": np.concatenate(acq)}


def main() -> int:
    out = {"contrasts": {}, "operating_points": {}}
    for a_grp, a_arm, b_grp, b_arm in PAIRS:
        for fold in ("fold1", "fold2"):
            A, B = h1(fold, a_grp, a_arm), h1(fold, b_grp, b_arm)
            pos = {t: i for i, t in enumerate(B["ids"])}
            if not set(A["ids"]) <= set(pos):
                print(f"[skip] {fold} {a_grp}-{b_grp}: trial sets differ")
                continue
            idx = np.array([pos[t] for t in A["ids"]], int)
            B = {"score": B["score"][idx], "acq": B["acq"][idx]}
            ops = {}
            for mode in ("deployment", f"matched_{MATCH_PFA:g}"):
                pa = "cal" if mode == "deployment" else "test_h0"
                pf = C.DESIGN_PFA if mode == "deployment" else MATCH_PFA
                ta = np.quantile(scores(fold, pa, a_grp, a_arm)[0], 1 - pf,
                                 method="higher")
                tb = np.quantile(scores(fold, pa, b_grp, b_arm)[0], 1 - pf,
                                 method="higher")
                h, r = A["score"] > ta, B["score"] > tb
                n01 = int(np.sum(h & ~r)); n10 = int(np.sum(~h & r))
                p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
                per = [float(h[A["acq"] == x].mean() - r[A["acq"] == x].mean())
                       for x in sorted(set(A["acq"]))]
                boot = np.array([np.mean(RNG.choice(per, len(per), replace=True))
                                 for _ in range(BOOT)])
                ops[mode] = {"thr_a": float(ta), "thr_b": float(tb),
                             "pd_a": float(h.mean()), "pd_b": float(r.mean()),
                             "delta_pd": float(h.mean() - r.mean()),
                             "ci_lo": float(np.percentile(boot, 2.5)),
                             "ci_hi": float(np.percentile(boot, 97.5)),
                             "mcnemar_p": p, "a_only": n01, "b_only": n10,
                             "n_acq_positive": int(sum(v > 0 for v in per)),
                             "n_acq": len(per), "n": int(len(h))}
                print(f"{a_grp:8s}-{b_grp:7s} {fold} {mode:14s} "
                      f"d={ops[mode]['delta_pd']:+.3f} "
                      f"[{ops[mode]['ci_lo']:+.3f},{ops[mode]['ci_hi']:+.3f}] "
                      f"p={p:.1e} acq+={ops[mode]['n_acq_positive']}/{ops[mode]['n_acq']}")
            out["contrasts"][f"{a_grp}-{b_grp}|{fold}"] = ops
    (WORK7 / "data/PHASE7_WARMSTART.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
