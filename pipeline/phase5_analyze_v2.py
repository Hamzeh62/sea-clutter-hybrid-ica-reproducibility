"""Phase 5 analysis: thresholds, Pd/PFA tables, clustered bootstrap, contrasts.

Reads only the frozen score arrays written by ``phase5_runner.py`` and the frozen
manifests; writes the campaign result CSVs and the integrity record.  Every number
in the report is produced here, so the report has no hand-entered statistics.

Statistical rules are those of ``phase4/reports/STATISTICAL_ANALYSIS_PLAN.md``:

* threshold = empirical (1 - pfa) order statistic of the arm's own CAL scores;
* realised PFA measured on TEST-H0, never on CAL;
* inference from a two-stage cluster bootstrap (acquisitions, then range cells),
  with the *same* resample indices for every detector (pairing preserved);
* exact McNemar on discordant pairs, Holm across the three pre-specified contrasts;
* binomial (Clopper-Pearson) intervals labelled as lower bounds on uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import phase5_common as C

DESIGN_PFA = 1e-3
B_BOOT = 2000
B_THRESH = 1000
PRIMARY_SCR = 0.0
REPORT = C.WORK / "reports"
STATS = C.WORK / "statistics"
FIGS = C.WORK / "figures"

ARM_GROUP = {arm: g for g, arms in C.ARMS.items() for arm in arms}
PRIMARY = [("hybrid", "anmf_dl"), ("hybrid", "ar8"), ("nophys", "hybrid")]
SECONDARY = [("hybrid", "anmf_unloaded"), ("hybrid", "glrt"), ("hybrid", "os"),
             ("hybrid", "ca"), ("hybrid", "notch"), ("hybrid", "notch_clean"),
             ("ar8", "ar8_vvfit"), ("notch", "notch_clean"),
             ("cold", "hybrid"), ("ica_only", "hybrid")]
COMMON_GRID = (-10.0, -5.0, 0.0, 5.0, 10.0)


# --------------------------------------------------------------------- loading
def load_scores(run: Path):
    """Concatenate every H0 block of a (fold, partition) into one arm vector.

    A block is one acquisition, so an arm's partition vector is the concatenation of
    its per-acquisition blocks; assigning instead of appending would silently threshold
    on the last acquisition alphabetically.
    """
    acc_h0: dict[tuple[str, str], dict[str, dict[str, list]]] = defaultdict(dict)
    for path in sorted((run / "scores/h0").glob("*.npz")):
        fold, part, group, acq = path.stem.split("__")
        with np.load(path) as z:
            cells = z["cells"].astype(int)
            windows = z["windows"].astype(int)
            for arm in C.ARMS[group]:
                slot = acc_h0[(fold, part)].setdefault(
                    arm, {"score": [], "cell": [], "window": [], "acq": []})
                slot["score"].append(z[f"arm_{arm}"].astype(float))
                slot["cell"].append(cells)
                slot["window"].append(windows)
                slot["acq"].append(np.array([acq] * len(cells)))
    # ---- common evaluable set ------------------------------------------------
    # A unit withdrawn by one arm because its statistic is undefined there (CFAR
    # guard consuming the whole range window at a clean-band edge) is withdrawn from
    # *every* arm, so thresholds, PFA audits and pooled Pd all use one unit set.
    excl_h0: dict[tuple[str, str], set] = defaultdict(set)
    excl_h1: dict[str, set] = defaultdict(set)
    for path in sorted((run / "scores/h0").glob("*.npz")):
        fold, part, group, acq = path.stem.split("__")
        with np.load(path) as z:
            if "excluded_cells" not in z.files:
                continue
            for c, w in zip(z["excluded_cells"].astype(int),
                            z["excluded_windows"].astype(int)):
                excl_h0[(fold, part)].add((acq, int(c), int(w)))
    h0: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for key, arms in acc_h0.items():
        drop = excl_h0.get(key, set())
        for arm, slot in arms.items():
            score = np.concatenate(slot["score"])
            cell = np.concatenate(slot["cell"])
            window = np.concatenate(slot["window"])
            acq = np.concatenate(slot["acq"])
            keep = np.ones(len(score), bool)
            if drop:
                keep = np.array([(a, int(c), int(w)) not in drop
                                 for a, c, w in zip(acq, cell, window)])
            h0[key][arm] = {"score": score[keep], "cell": cell[keep],
                            "window": window[keep], "acq": acq[keep].tolist(),
                            "n_excluded": int((~keep).sum())}
    # The withdrawal list must be collected over *every* group before any group is
    # filtered.  Collecting it inside the per-group loop made the trial set depend on
    # the alphabetical order of the groups: the group that reported the withdrawal
    # (``cfar``) kept 1990 trials while groups loaded earlier kept all 2000, which then
    # silenced every paired contrast that touched them.
    for path in sorted((run / "scores/h1").glob("*.npz")):
        fold = path.stem.split("__")[0]
        with np.load(path) as z:
            if "excluded_trial_ids" in z.files:
                excl_h1[fold].update(z["excluded_trial_ids"].astype(str).tolist())

    h1: dict[tuple[str, str], dict] = {}
    for fold in ("fold1", "fold2"):
        for group in C.GROUPS:
            chunks = sorted((run / "scores/h1").glob(f"{fold}__{group}__c*.npz"))
            if not chunks:
                continue
            acc = {arm: [] for arm in C.ARMS[group]}
            meta = {k: [] for k in ("trial_ids", "cells", "acqs", "windows")}
            grid = None
            for p in chunks:
                with np.load(p) as z:
                    for arm in C.ARMS[group]:
                        acc[arm].append(z[f"arm_{arm}"].astype(float))
                    for k in meta:
                        meta[k].append(z[k])
                    grid = z["scr_db"].astype(float)
            ids = np.concatenate(meta["trial_ids"]).astype(str)
            keep = (np.ones(len(ids), bool) if not excl_h1[fold]
                    else ~np.isin(ids, sorted(excl_h1[fold])))
            h1[(fold, group)] = {
                "scores": {a: np.concatenate(v)[keep] for a, v in acc.items()},
                "trial_ids": ids[keep],
                **{k: np.concatenate(v)[keep] for k, v in meta.items() if k != "trial_ids"},
                "scr_db": grid,
                "n_excluded_trials": int((~keep).sum())}
    return h0, h1


# ----------------------------------------------------------------- thresholding
def threshold(cal: np.ndarray, pfa: float = DESIGN_PFA) -> tuple[float, int]:
    s = np.sort(cal)[::-1]
    k = max(1, int(math.floor(len(s) * pfa)))
    return float(s[k - 1]), k


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# -------------------------------------------------------------------- bootstrap
def cell_table(acqs, cells, binary: np.ndarray | None = None):
    """Per-cell totals and (optional) per-column successes; cells are the clusters."""
    keys = list(zip(list(acqs), list(cells)))
    index, order = {}, []
    for k in keys:
        if k not in index:
            index[k] = len(order)
            order.append(k)
    idx = np.array([index[k] for k in keys], int)
    ncell = len(order)
    tot = np.zeros(ncell)
    np.add.at(tot, idx, 1.0)
    succ = None
    if binary is not None:
        succ = np.zeros((ncell, binary.shape[1]))
        np.add.at(succ, idx, binary.astype(float))
    uniq = sorted({k[0] for k in order})
    acq_id = np.array([uniq.index(k[0]) for k in order], int)
    return {"uniq_acq": uniq, "acq_id": acq_id, "tot": tot, "succ": succ,
            "cells": order}


def draw_indices(rng, table) -> np.ndarray:
    """Two-stage resample: acquisitions, then cells within each drawn acquisition."""
    sizes = np.array([np.sum(table["acq_id"] == i)
                      for i in range(len(table["uniq_acq"]))])
    offs = np.concatenate([[0], np.cumsum(sizes)])
    picks = []
    for d in rng.integers(0, len(sizes), len(sizes)):
        picks.append(offs[d] + rng.integers(0, sizes[d], sizes[d]))
    return np.concatenate(picks)


def boot_binary(table, rng, b: int) -> np.ndarray:
    out = np.empty((b, table["succ"].shape[1]))
    for i in range(b):
        idx = draw_indices(rng, table)
        out[i] = table["succ"][idx].sum(0) / table["tot"][idx].sum()
    return out


def ci(boot: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------- report
def main(argv=None) -> int:
    global B_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run directory (default: campaign)")
    ap.add_argument("--out", default=None, help="output directory (default: phase5)")
    ap.add_argument("--boot", type=int, default=B_BOOT)
    args = ap.parse_args(argv)
    B_BOOT = args.boot
    cfg = C.config()
    run = Path(args.run) if args.run else C.run_dir(cfg)
    report_dir = Path(args.out) / "reports" if args.out else REPORT
    stats_dir = Path(args.out) / "statistics" if args.out else STATS
    report_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    h0, h1 = load_scores(run)
    folds = ["fold1", "fold2"]
    integrity = {"checks": []}

    def rec(check: str, ok: bool, detail) -> None:
        integrity["checks"].append({"check": check, "status": "PASS" if ok else "FAIL",
                                    "detail": detail})

    # ---- 1. structural integrity -------------------------------------------
    seen = defaultdict(set)
    for (fold, part), arms in h0.items():
        for arm, d in arms.items():
            keys = set(zip(d["acq"], d["cell"], d["window"]))
            rec(f"h0_unique_units|{fold}|{part}|{arm}", len(keys) == len(d["score"]),
                len(d["score"]))
            rec(f"h0_finite|{fold}|{part}|{arm}",
                bool(np.all(np.isfinite(d["score"]))), float(np.nanmax(d["score"])))
            seen[(fold, part, arm)] |= keys
    for fold in folds:
        for part in ("cal", "test_h0"):
            sizes = {len(seen[(fold, part, a)]) for a in ARM_GROUP if (fold, part, a) in seen}
            rec(f"h0_same_pool_all_arms|{fold}|{part}", len(sizes) == 1, sorted(sizes))
            sets = [frozenset(seen[(fold, part, a)]) for a in ARM_GROUP
                    if (fold, part, a) in seen]
            rec(f"h0_identical_unit_sets|{fold}|{part}", len(set(sets)) == 1,
                {"n_units": len(sets[0]) if sets else 0,
                 "n_withdrawn": sum(h0[(fold, part)][a]["n_excluded"] for a in ARM_GROUP
                                    if (fold, part, a) in seen and "n_excluded" in h0[(fold, part)][a])})
        rec(f"h0_cal_test_disjoint|{fold}",
            not (seen[(fold, "cal", "hybrid")] & seen[(fold, "test_h0", "hybrid")]), None)
    for fold in folds:
        ids = {a: list(h1[(fold, g)]["trial_ids"]) for (f, g) in h1 if f == fold
               for a in C.ARMS[g]}
        rec(f"h1_unique_trial_ids|{fold}",
            all(len(set(v)) == len(v) for v in ids.values()),
            {a: len(v) for a, v in ids.items()})
        first = next(iter(ids.values()))
        rec(f"h1_same_trial_set_all_arms|{fold}",
            all(v == first for v in ids.values()), len(first))
        for (f, g), d in h1.items():
            if f != fold:
                continue
            ok = all(np.all(np.isfinite(d["scores"][a])) for a in C.ARMS[g])
            rec(f"h1_finite|{fold}|{g}", bool(ok),
                list(d["scores"][C.ARMS[g][0]].shape))

    # ---- 2. thresholds and PFA audit ---------------------------------------
    thr: dict[tuple[str, str], float] = {}
    pfa_rows = []
    for fold in folds:
        for arm in ARM_GROUP:
            cal = h0[(fold, "cal")][arm]
            tst = h0[(fold, "test_h0")][arm]
            t, rank = threshold(cal["score"])
            thr[(fold, arm)] = t
            k_cal = int(np.sum(cal["score"] > t))
            k_t = int(np.sum(tst["score"] > t))
            n_t = len(tst["score"])
            p_hat = k_t / n_t
            tab = cell_table(tst["acq"], tst["cell"],
                             (tst["score"] > t).astype(float)[:, None])
            b = boot_binary(tab, np.random.default_rng(
                cfg["seeds"]["bootstrap"] + int(fold[-1])), B_BOOT)[:, 0]
            lo, hi = ci(b)
            cp = clopper_pearson(k_t, n_t)
            rec(f"pfa_ci_ordered|{fold}|{arm}", lo <= hi, [lo, hi])
            rec(f"pfa_ci_contains_point|{fold}|{arm}", lo - 1e-12 <= p_hat <= hi + 1e-12,
                [lo, p_hat, hi])
            pfa_rows.append({
                "arm": arm, "fold": fold, "design_pfa": DESIGN_PFA, "threshold": t,
                "cal_rank": rank, "cal_n": len(cal["score"]), "cal_exceed": k_cal,
                "test_n": n_t, "test_exceed": k_t, "realised_pfa": p_hat,
                "cp_lo": cp[0], "cp_hi": cp[1], "boot_lo": lo, "boot_hi": hi,
                "pfa_ratio_to_design": p_hat / DESIGN_PFA,
            })

    # ---- 3. Pd tables (per fold) ------------------------------------------
    pd_rows, pd_curves = [], {}
    for (fold, group), d in h1.items():
        for arm in C.ARMS[group]:
            pd = (d["scores"][arm] > thr[(fold, arm)]).astype(float)
            pd_curves[(fold, arm)] = pd
            for j, scr in enumerate(d["scr_db"]):
                tab = cell_table(d["acqs"], d["cells"], pd[:, j:j + 1])
                b = boot_binary(tab, np.random.default_rng(
                    cfg["seeds"]["bootstrap"] + 1000 * int(fold[-1]) + j), B_BOOT)[:, 0]
                lo, hi = ci(b)
                rec(f"pd_ci_ordered|{fold}|{arm}|{scr:g}", lo <= hi, [lo, hi])
                pd_rows.append({"arm": arm, "group": group, "fold": fold, "scr_db": scr,
                                "pd": float(pd[:, j].mean()), "boot_lo": lo,
                                "boot_hi": hi, "n_trials": int(pd.shape[0]),
                                "n_detected": int(pd[:, j].sum())})
    grids = {arm: sorted({r["scr_db"] for r in pd_rows if r["arm"] == arm})
             for arm in ARM_GROUP}

    pooled: dict[tuple[str, float], float] = {}
    for arm in ARM_GROUP:
        for scr in grids.get(arm, []):
            num = sum(r["pd"] * r["n_trials"] for r in pd_rows
                      if r["arm"] == arm and r["scr_db"] == scr)
            den = sum(r["n_trials"] for r in pd_rows
                      if r["arm"] == arm and r["scr_db"] == scr)
            pooled[(arm, scr)] = float(num / den) if den else float("nan")

    # ---- 4. paired contrasts ----------------------------------------------
    def paired_contrast(arm_a: str, arm_b: str, scr: float) -> dict | None:
        ga, gb = grids.get(arm_a, []), grids.get(arm_b, [])
        if not any(abs(x - scr) < 1e-9 for x in ga):
            return None
        if not any(abs(x - scr) < 1e-9 for x in gb):
            return None
        rows = []
        for fold in folds:
            g_a, g_b = ARM_GROUP[arm_a], ARM_GROUP[arm_b]
            if (fold, g_a) not in h1 or (fold, g_b) not in h1:
                return None
            da, db = h1[(fold, g_a)], h1[(fold, g_b)]
            if not np.array_equal(da["trial_ids"], db["trial_ids"]):
                return None
            ja = int(np.argmin(np.abs(da["scr_db"] - scr)))
            jb = int(np.argmin(np.abs(db["scr_db"] - scr)))
            a = (da["scores"][arm_a][:, ja] > thr[(fold, arm_a)]).astype(float)
            b = (db["scores"][arm_b][:, jb] > thr[(fold, arm_b)]).astype(float)
            rows.append((fold, da, a, b))
        n = sum(len(a) for _, _, a, b in rows)
        tot_a = sum(float(a.sum()) for _, _, a, b in rows)
        tot_b = sum(float(b.sum()) for _, _, a, b in rows)
        a_only = sum(int(np.sum((a > 0) & (b == 0))) for _, _, a, b in rows)
        b_only = sum(int(np.sum((a == 0) & (b > 0))) for _, _, a, b in rows)
        p = (float(stats.binomtest(a_only, a_only + b_only, 0.5).pvalue)
             if a_only + b_only else 1.0)
        rng = np.random.default_rng(cfg["seeds"]["bootstrap"] + 77 + int(round(scr * 10)))
        boots = np.empty(B_BOOT)
        tabs = [(h1_table_pair(d, a, b)) for _, d, a, b in rows]
        for i in range(B_BOOT):
            num_a, num_b, den = 0.0, 0.0, 0
            for ta, tb in tabs:
                idx = draw_indices(rng, ta)
                num_a += ta["succ"][idx].sum()
                num_b += tb["succ"][idx].sum()
                den += ta["tot"][idx].sum()
            boots[i] = (num_a - num_b) / den
        lo, hi = ci(boots)
        per_acq = {}
        for fold, d, a, b in rows:
            for acq in sorted(set(d["acqs"])):
                m = d["acqs"] == acq
                dlt = float(a[m].mean() - b[m].mean())
                per_acq[acq] = dlt
        return {"arm_a": arm_a, "arm_b": arm_b, "scr_db": scr,
                "pd_a": tot_a / n, "pd_b": tot_b / n, "delta_pd": (tot_a - tot_b) / n,
                "boot_lo": lo, "boot_hi": hi, "n": n, "a_only": a_only, "b_only": b_only,
                "mcnemar_p": p,
                "acq_favouring_a": sum(1 for v in per_acq.values() if v > 0),
                "acq_favouring_b": sum(1 for v in per_acq.values() if v < 0),
                "n_acq": len(per_acq), "per_acq_delta": per_acq}

    def h1_table_pair(d, a, b):
        ta = cell_table(d["acqs"], d["cells"], a[:, None])
        tb = cell_table(d["acqs"], d["cells"], b[:, None])
        return ta, tb

    contrasts = []
    for arm_a, arm_b in PRIMARY:
        for scr in sorted(set(grids.get(arm_a, [])) & set(grids.get(arm_b, []))):
            c = paired_contrast(arm_a, arm_b, scr)
            if c:
                contrasts.append(c)
    at0 = [c for c in contrasts if abs(c["scr_db"] - PRIMARY_SCR) < 1e-9]
    order = sorted(range(len(at0)), key=lambda i: at0[i]["mcnemar_p"])
    m, running = len(at0), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * at0[i]["mcnemar_p"]))
        at0[i]["holm_p"] = float(running)
    for c in contrasts:
        c.setdefault("holm_p", float("nan"))

    secondary = []
    for arm_a, arm_b in SECONDARY:
        for scr in sorted(set(grids.get(arm_a, [])) & set(grids.get(arm_b, []))):
            c = paired_contrast(arm_a, arm_b, scr)
            if c:
                c["holm_p"] = float("nan")
                secondary.append(c)

    # ---- 5. per-acquisition and heterogeneity ------------------------------
    per_acq_rows = []
    for arm in ARM_GROUP:
        for fold in folds:
            group = ARM_GROUP[arm]
            if (fold, group) not in h1:
                continue
            d = h1[(fold, group)]
            j = int(np.argmin(np.abs(d["scr_db"] - PRIMARY_SCR)))
            y = pd_curves[(fold, arm)][:, j]
            for acq in sorted(set(d["acqs"])):
                msk = d["acqs"] == acq
                per_acq_rows.append({"arm": arm, "fold": fold, "acq": acq,
                                     "scr_db": PRIMARY_SCR,
                                     "n_trials": int(msk.sum()),
                                     "n_cells": int(len(set(d["cells"][msk]))),
                                     "pd": float(y[msk].mean())})

    def discordant_by_acquisition(arm_a, arm_b, scr: float = PRIMARY_SCR) -> dict:
        """Matched binary outcomes per acquisition: (n01, n10).

        Cochran's Q applied to per-acquisition *detection counts* is undefined here: the
        acquisitions have different trial counts, so the usual ``k*sum(a+b) - sum(a+b)^2``
        denominator is not the variance of anything and turns negative.  The paired
        binary outcomes are the right unit, and they give two interpretable statistics:
        a stratified McNemar test for the average effect and an acquisition-homogeneity
        Q for whether that effect is the same in every acquisition.
        """
        out = {}
        for fold in folds:
            g_a, g_b = ARM_GROUP[arm_a], ARM_GROUP[arm_b]
            if (fold, g_a) not in h1 or (fold, g_b) not in h1:
                continue
            da, db = h1[(fold, g_a)], h1[(fold, g_b)]
            if not np.array_equal(da["trial_ids"], db["trial_ids"]):
                continue
            ja = int(np.argmin(np.abs(da["scr_db"] - scr)))
            jb = int(np.argmin(np.abs(db["scr_db"] - scr)))
            a = da["scores"][arm_a][:, ja] > thr[(fold, arm_a)]
            b = db["scores"][arm_b][:, jb] > thr[(fold, arm_b)]
            for acq in sorted(set(da["acqs"])):
                m = da["acqs"] == acq
                out[acq] = (int(np.sum(a[m] & ~b[m])), int(np.sum(~a[m] & b[m])),
                            int(m.sum()))
        return out

    q_rows = []
    for arm_a, arm_b in PRIMARY:
        disc = discordant_by_acquisition(arm_a, arm_b)
        n01 = np.array([v[0] for v in disc.values()], float)
        n10 = np.array([v[1] for v in disc.values()], float)
        tot = n01 + n10
        k = len(disc)
        d = n01 - n10
        if k >= 2 and tot.sum() > 0:
            q = float(np.sum(d ** 2 / np.where(tot > 0, tot, np.nan)) - d.sum() ** 2 / tot.sum())
            p_q = float(stats.chi2.sf(q, k - 1)) if np.isfinite(q) else float("nan")
        else:
            q, p_q = float("nan"), float("nan")
        z = float(d.sum() / math.sqrt(tot.sum())) if tot.sum() > 0 else 0.0
        p_strat = float(2 * stats.norm.sf(abs(z))) if tot.sum() > 0 else 1.0
        q_rows.append({"arm_a": arm_a, "arm_b": arm_b, "scr_db": PRIMARY_SCR,
                       "stratified_mcnemar_z": z, "stratified_mcnemar_p": p_strat,
                       "homogeneity_q": q, "df": k - 1, "homogeneity_p": p_q,
                       "n_acquisitions": k,
                       "acq_favouring_a": int(np.sum(d > 0)),
                       "acq_favouring_b": int(np.sum(d < 0)),
                       "acq_tied": int(np.sum(np.abs(d) < 1e-12)),
                       "n01_total": int(n01.sum()), "n10_total": int(n10.sum())})

    # ---- 6b. threshold-uncertainty of the paired contrast --------------------
    # The absolute-Pd sensitivity above cannot answer the question that matters: the
    # comparison uses two different thresholds, so a shift in the calibration tail
    # moves both arms.  Here the CAL units and the TEST trials are resampled with the
    # *same* indices for both arms and each arm's threshold is re-estimated in the same
    # replicate, which is the only way to see whether the ranking is a calibration
    # artefact.  With ~80 clusters and a rank-8 tail the interval is expected to be
    # wide; that width is the result, not a failure.
    def kth_largest(x: np.ndarray, pfa: float = DESIGN_PFA) -> float:
        s = np.sort(x)[::-1]
        return float(s[max(1, int(math.floor(len(s) * pfa))) - 1])

    redraw = []
    for arm_a, arm_b in PRIMARY:
        rows = []
        for fold in folds:
            g_a, g_b = ARM_GROUP[arm_a], ARM_GROUP[arm_b]
            if (fold, g_a) not in h1 or (fold, g_b) not in h1:
                continue
            da, db = h1[(fold, g_a)], h1[(fold, g_b)]
            if not np.array_equal(da["trial_ids"], db["trial_ids"]):
                continue
            ca, cb = h0[(fold, "cal")][arm_a], h0[(fold, "cal")][arm_b]
            if len(ca["score"]) != len(cb["score"]):
                continue
            ja = int(np.argmin(np.abs(da["scr_db"] - PRIMARY_SCR)))
            jb = int(np.argmin(np.abs(db["scr_db"] - PRIMARY_SCR)))
            rows.append((fold, ca, cb, da["scores"][arm_a][:, ja],
                         db["scores"][arm_b][:, jb],
                         cell_table(ca["acq"], ca["cell"]),
                         cell_table(da["acqs"], da["cells"])))
        if not rows:
            continue
        rng = np.random.default_rng(cfg["seeds"]["bootstrap"] + 5150)
        vals, ths = np.empty(B_BOOT), np.empty(B_BOOT)
        for i in range(B_BOOT):
            pa, pb, na, nb = 0.0, 0.0, 0, 0
            for fold, ca, cb, sa, sb, tcal, ttest in rows:
                ic = draw_indices(rng, tcal)
                ta, tb = kth_largest(ca["score"][ic]), kth_largest(cb["score"][ic])
                it = draw_indices(rng, ttest)
                pa += float(np.sum(sa[it] > ta))
                pb += float(np.sum(sb[it] > tb))
                na += len(it)
                nb += len(it)
                if len(rows) == 1:
                    ths[i] = ta / (tb if tb else 1.0)
            vals[i] = pa / na - pb / nb
        fixed = [c for c in contrasts
                 if c["arm_a"] == arm_a and c["arm_b"] == arm_b
                 and abs(c["scr_db"] - PRIMARY_SCR) < 1e-9]
        redraw.append({
            "arm_a": arm_a, "arm_b": arm_b, "scr_db": PRIMARY_SCR,
            "delta_pd_frozen": fixed[0]["delta_pd"] if fixed else float("nan"),
            "redraw_median": float(np.median(vals)),
            "redraw_lo": float(np.percentile(vals, 2.5)),
            "redraw_hi": float(np.percentile(vals, 97.5)),
            "p_a_worse_than_b": float(np.mean(vals < 0)),
            "p_sign_flip": float(min(np.mean(vals < 0), np.mean(vals > 0))),
            "n_boot": B_BOOT})

    # ---- 6. threshold-uncertainty sensitivity ------------------------------
    sens = []
    for arm in ("hybrid", "anmf_dl", "ar8"):
        for fold in folds:
            group = ARM_GROUP[arm]
            cal = h0[(fold, "cal")][arm]
            d = h1[(fold, group)]
            j = int(np.argmin(np.abs(d["scr_db"] - PRIMARY_SCR)))
            y = (d["scores"][arm][:, j] > thr[(fold, arm)]).astype(float)
            tab = cell_table(d["acqs"], d["cells"], y[:, None])
            # trial indices belonging to every cell, in the same cell order
            keys = list(zip(list(d["acqs"]), list(d["cells"])))
            order: list = []
            index: dict = {}
            for k in keys:
                if k not in index:
                    index[k] = len(order)
                    order.append(k)
            cell_idx = np.array([index[k] for k in keys], int)
            trials_by_cell = [np.where(cell_idx == c)[0] for c in range(len(order))]
            cal_tab = cell_table(cal["acq"], cal["cell"])
            rng = np.random.default_rng(cfg["seeds"]["bootstrap"] + 999 + int(fold[-1]))
            pds = np.empty(B_THRESH)
            for i in range(B_THRESH):
                idx_c = draw_indices(rng, cal_tab)
                cs = np.sort(cal["score"][idx_c])[::-1]
                thr_b = cs[max(1, int(len(cs) * DESIGN_PFA)) - 1]
                idx_t = draw_indices(rng, tab)
                sel = np.concatenate([trials_by_cell[c] for c in idx_t])
                pds[i] = float(np.mean(d["scores"][arm][sel, j] > thr_b))
            lo, hi = ci(pds)
            sens.append({"arm": arm, "fold": fold, "scr_db": PRIMARY_SCR,
                         "pd_fixed_threshold": float(y.mean()),
                         "pd_lo_threshold_redrawn": lo, "pd_hi_threshold_redrawn": hi,
                         "n_boot": int(np.sum(~np.isnan(pds)))})

    # ---- 7. runtime --------------------------------------------------------
    manifest = C.read_json(run / "campaign_manifest.json")
    by_group = defaultdict(lambda: {"n": 0, "wall": 0.0, "rss": 0.0})
    for j in manifest["jobs"]:
        g = j["group"] or j["kind"]
        by_group[g]["n"] += 1
        by_group[g]["wall"] += j["wall_s"] or 0.0
        by_group[g]["rss"] = max(by_group[g]["rss"], j["peak_rss_mb"] or 0.0)
    runtime = [{"group": g, "n_jobs": v["n"], "cpu_hours": v["wall"] / 3600.0,
                "peak_rss_mb": v["rss"]} for g, v in sorted(by_group.items())]
    runtime.append({"group": "TOTAL", "n_jobs": manifest["n_jobs"],
                    "cpu_hours": sum(v["wall"] for v in by_group.values()) / 3600.0,
                    "peak_rss_mb": max(v["rss"] for v in by_group.values())})

    # ---- 8. write ----------------------------------------------------------
    def write(name: str, rows: list[dict]) -> None:
        path = report_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = [k for k in rows[0] if not isinstance(rows[0][k], dict)]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{r[k]:.10g}" if isinstance(r[k], float) else r[k])
                            for k in keys})

    write("MINIMAL_CAMPAIGN_RESULTS.csv", pd_rows)
    write("MINIMAL_CAMPAIGN_PFA_AUDIT.csv", pfa_rows)
    write("MINIMAL_CAMPAIGN_ACQUISITION_RESULTS.csv", per_acq_rows)
    write("MINIMAL_CAMPAIGN_ABLATIONS.csv",
          [r for r in contrasts + secondary
           if r["arm_a"] in ("nophys", "cold", "ica_only", "ar8", "ar8_vvfit", "notch")
           or r["arm_b"] in ("nophys", "cold", "ica_only", "ar8", "ar8_vvfit", "notch")])
    write("MINIMAL_CAMPAIGN_CONTRASTS.csv", contrasts + secondary)
    write("MINIMAL_CAMPAIGN_RUNTIME.csv", runtime)
    write("MINIMAL_CAMPAIGN_HETEROGENEITY.csv", q_rows)
    write("MINIMAL_CAMPAIGN_THRESHOLD_SENSITIVITY.csv", sens)
    write("MINIMAL_CAMPAIGN_THRESHOLD_REDRAW.csv", redraw)

    summary = {
        "generated": C.utc_now(), "campaign": cfg["campaign"],
        "config_sha256": cfg["config_sha256"], "design_pfa": DESIGN_PFA,
        "bootstrap_B": B_BOOT, "threshold_bootstrap_B": B_THRESH,
        "primary_scr_db": PRIMARY_SCR,
        "n_h0_windows": {f"{f}/{p}": len(h0[(f, p)]["hybrid"]["score"])
                         for f in folds for p in ("cal", "test_h0")},
        "n_h1_trials": {f: len(h1[(f, "hybrid")]["trial_ids"]) for f in folds
                        if (f, "hybrid") in h1},
        "common_evaluable_set": {
            "h0_units": {f"{f}/{p}": len(h0[(f, p)]["hybrid"]["score"])
                         for f in folds for p in ("cal", "test_h0")
                         if (f, p) in h0},
            "h1_withdrawn_trials": {f: int(h1[(f, "hybrid")]["n_excluded_trials"])
                                    for f in folds if (f, "hybrid") in h1}},
        "cfar_withdrawn_units": {f"{f}/{p}": sorted({h0[(f, p)][a]["n_excluded"]
                                                      for a in ARM_GROUP
                                                      if (f, p, a) in seen})
                                 for f in folds for p in ("cal", "test_h0")
                                 if (f, p) in h0},
        "thresholds": {f"{k[0]}|{k[1]}": v for k, v in thr.items()},
        "pooled_pd": {f"{a}|{s:g}": v for (a, s), v in pooled.items()},
        "ranking_at_0db": sorted([(a, pooled[(a, PRIMARY_SCR)]) for a in ARM_GROUP
                                  if (a, PRIMARY_SCR) in pooled], key=lambda t: -t[1]),
        "ranking_common_grid": sorted(
            [(a, float(np.mean([pooled[(a, s)] for s in COMMON_GRID])))
             for a in ARM_GROUP if all((a, s) in pooled for s in COMMON_GRID)],
            key=lambda t: -t[1]),
        "primary_contrasts_at_0db": {f"{c['arm_a']}-{c['arm_b']}": {
            "delta_pd": c["delta_pd"], "ci": [c["boot_lo"], c["boot_hi"]],
            "mcnemar_p": c["mcnemar_p"], "holm_p": c["holm_p"]} for c in at0},
        "contrasts_with_per_acq": {f"{c['arm_a']}-{c['arm_b']}|{c['scr_db']:g}":
                                   c["per_acq_delta"] for c in contrasts + secondary},
        "heterogeneity": q_rows,
        "threshold_sensitivity": sens,
        "threshold_redraw_contrast": redraw,
        "integrity": integrity,
        "runtime_total_cpu_hours": runtime[-1]["cpu_hours"],
    }
    (stats_dir / "ANALYSIS_SUMMARY.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    n_fail = sum(c["status"] == "FAIL" for c in integrity["checks"])
    print(json.dumps({"integrity_checks": len(integrity["checks"]), "failed": n_fail,
                      "ranking_0db": summary["ranking_at_0db"][:8],
                      "ranking_common": summary["ranking_common_grid"][:8],
                      "primary_0db": summary["primary_contrasts_at_0db"],
                      "cpu_hours": summary["runtime_total_cpu_hours"]}, indent=1,
                     default=float))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
