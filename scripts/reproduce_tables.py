#!/usr/bin/env python
"""Recompute every number published in the manuscript's tables from released data.

The expected values in ``manifests/PUBLISHED_VALUES.json`` were transcribed from the
manuscript, not from these files.  This script therefore tests the released artifacts
*against the paper*: a failure means the repository does not support a published number.

Usage
-----
    python scripts/reproduce_tables.py            # human-readable table
    python scripts/reproduce_tables.py --json     # machine-readable, for CI
    python scripts/reproduce_tables.py --tol 1e-3 # loosen the rounding tolerance

Exit status is 0 only if every published value is reproduced.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPECTED = ROOT / "manifests" / "PUBLISHED_VALUES.json"


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(x) -> float:
    return float(x)


# --------------------------------------------------------------------- Table 1
def table1() -> dict:
    """Measured cross-polarised target contrast (4 literature cells + the 16-cell summary)."""
    corrected = rows(RESULTS / "phase7/PHASE7_POLARIMETRY_CORRECTED.csv")
    target = {r["acq_id"]: r for r in rows(RESULTS / "phase6/PHASE6_POLARIMETRY_TARGET.csv")}
    out: dict[str, float] = {}
    lit = [r for r in corrected if r["cell_source"] == "literature"]
    for r in lit:
        a = r["acq_id"]
        out[f"{a}.delta_vv_db"] = f(target[a]["delta_vv_db"])
        out[f"{a}.eta"] = f(r["eta_linear"])
        out[f"{a}.eta_lo"] = f(r["eta_ci_lo"])
        out[f"{a}.eta_hi"] = f(r["eta_ci_hi"])
        out[f"{a}.dscr_db"] = f(r["dSCR_db"])
    etas = sorted(f(r["eta_linear"]) for r in corrected)
    n = len(etas)
    out["all.median_eta"] = (etas[n // 2] if n % 2 else 0.5 * (etas[n // 2 - 1] + etas[n // 2]))
    out["all.dscr_min"] = min(f(r["dSCR_db"]) for r in corrected)
    out["all.dscr_max"] = max(f(r["dSCR_db"]) for r in corrected)
    out["all.n_ci_excluding_zero"] = float(
        sum(r["ci_excludes_zero"].strip().lower() == "true" for r in corrected))
    out["all.n_cells"] = float(len(corrected))
    return out


# --------------------------------------------------------------------- Table 2
CONDS = ("vh_absent", "dSCR-10__randphi", "dSCR0__randphi")
ARMS = ("hybrid", "ar8", "notch", "anmf_dl")


def table2() -> dict:
    """Detection probability at the matched operating point (5e-3), VV SCR = 0 dB."""
    mat = rows(RESULTS / "phase7/PHASE7_V1_MATCHED.csv")
    out = {}
    for fold in ("fold1", "fold2"):
        for arm in ARMS:
            for cond in CONDS:
                v = [r for r in mat if r["fold"] == fold and r["arm"] == arm
                     and r["condition"] == cond and r["mode"] == "matched_0.005"]
                if not v:
                    # ANMF-DL reads only the co-polarised channel, so it has a single
                    # VH-independent row per fold.  The manuscript prints that value in
                    # the whole row and says so in the caption; propagating it here is
                    # therefore a property of the detector, not a substitution.
                    v = [r for r in mat if r["fold"] == fold and r["arm"] == arm
                         and r["condition"] == "vh_absent" and r["mode"] == "matched_0.005"]
                    if len(v) != 1:
                        raise SystemExit(f"no VH-invariant fallback for {fold} {arm}")
                if len(v) != 1:
                    raise SystemExit(f"matched row not unique: {fold} {arm} {cond} {len(v)}")
                out[f"{arm}.{fold}.{cond}"] = f(v[0]["pd"])
    return out


# --------------------------------------------------------------------- Table 3
VARIANTS = ("deployed_g0", "deployed_g1", "target_channel", "delay1",
            "temporal_oracle", "delay512")


def table3() -> dict:
    """Reference-definition controls for AR(8)+NMF, deployment thresholds."""
    v3 = rows(RESULTS / "phase7/PHASE7_V3_REFERENCE_RESULTS.csv")
    out = {}
    for fold in ("fold1", "fold2"):
        base = [r for r in v3 if r["fold"] == fold and r["variant"] == "deployed_g0"][0]
        for var in VARIANTS:
            r = [x for x in v3 if x["fold"] == fold and x["variant"] == var]
            if len(r) != 1:
                raise SystemExit(f"V3 row missing: {fold} {var}")
            r = r[0]
            out[f"{fold}.{var}.pd"] = f(r["pd"])
            # signed difference of the variant against the frozen clean-reference arm
            out[f"{fold}.{var}.delta"] = round(f(r["pd"]) - f(base["pd"]), 6)
    return out


# --------------------------------------------------------------------- Table 4
T4_CONTRASTS = ("hybrid-ar8", "hybrid-notch", "hybrid-anmf_dl")


def table4() -> dict:
    """Paired contrasts at the matched operating point, 0 dB."""
    c = rows(RESULTS / "phase7/PHASE7_V1_CONTRASTS.csv")
    out = {}
    for fold in ("fold1", "fold2"):
        for con in T4_CONTRASTS:
            r = [x for x in c if x["mode"] == "matched_0.005" and x["dSCR_dB"] == "0"
                 and x["fold"] == fold and x["contrast"] == con]
            if len(r) != 1:
                raise SystemExit(f"contrast row missing: {fold} {con}")
            r = r[0]
            out[f"{fold}.{con}.delta"] = f(r["delta_pd"])
            out[f"{fold}.{con}.lo"] = f(r["ci_lo"])
            out[f"{fold}.{con}.hi"] = f(r["ci_hi"])
            out[f"{fold}.{con}.p_holm"] = f(r["p_holm"])
    return out


# --------------------------------------------------------------------- Table 5
T5_CONTRASTS = {"warm": "hybrid-cold", "net": "hybrid-ica", "phys": "hybrid-nophys"}
T5_MODES = ("matched_0.005", "deployment")


def table5() -> dict:
    """Ablations: warm start, network, physics prior; both operating points."""
    d = json.loads((RESULTS / "phase7/data/PHASE7_WARMSTART.json").read_text(encoding="utf-8"))
    out = {}
    for label, con in T5_CONTRASTS.items():
        for fold in ("fold1", "fold2"):
            blk = d["contrasts"][f"{con}|{fold}"]
            for mode in T5_MODES:
                r = blk[mode]
                tag = f"{label}.{fold}.{'matched' if mode == 'matched_0.005' else 'deploy'}"
                out[f"{tag}.delta"] = f(r["delta_pd"])
                out[f"{tag}.lo"] = f(r["ci_lo"])
                out[f"{tag}.hi"] = f(r["ci_hi"])
    return out


# --------------------------------------------------------------------- Table 6
RULES = ("A", "B", "C", "D")


def table6() -> dict:
    """Realised-to-nominal false-alarm ratio at 1e-3, four clutter-only calibration rules."""
    v2 = rows(RESULTS / "phase7/PHASE7_V2_CALIBRATION_RESULTS.csv")
    out = {}
    for rule in RULES:
        sel = [r for r in v2 if r["rule"] == rule]
        out[f"{rule}.worst_pooled"] = max(f(r["realised_over_design"]) for r in sel)
        out[f"{rule}.worst_acq"] = max(f(r["worst_acq_over_design"]) for r in sel)
        out[f"{rule}.n_acq_gt2"] = float(max(int(r["acqs_over_2x"]) for r in sel))
    return out


TABLES = {"T1": table1, "T2": table2, "T3": table3, "T4": table4,
          "T5": table5, "T6": table6}


def tolerance_for(table: dict, key: str, base: float) -> float:
    """Half a unit in the last digit published for this table, with per-key overrides."""
    tol = max(base, float(table.get("abs_tol", base)))
    for suffix, override in table.get("key_tolerance", {}).items():
        if key.rsplit(".", 1)[-1] == suffix:
            tol = max(tol, float(override))
    return tol


def compare(tol: float) -> list[dict]:
    spec = json.loads(EXPECTED.read_text(encoding="utf-8"))
    report = []
    for name, fn in TABLES.items():
        got = fn()
        tbl = spec["tables"][name]
        want = tbl["values"]
        for key, exp in want.items():
            if key not in got:
                report.append({"table": name, "key": key, "expected": exp,
                               "recomputed": None, "status": "MISSING"})
                continue
            val = got[key]
            if exp is None:
                ok = False
            else:
                ok = abs(val - float(exp)) <= tolerance_for(tbl, key, tol)
            report.append({"table": name, "key": key, "expected": exp,
                           "recomputed": val, "status": "PASS" if ok else "FAIL"})
        extra = set(got) - set(want)
        for key in sorted(extra):
            report.append({"table": name, "key": key, "expected": None,
                           "recomputed": got[key], "status": "UNLISTED"})
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tol", type=float, default=6e-4,
                    help="absolute tolerance on a value published to 3 decimals")
    ap.add_argument("--show-unlisted", action="store_true")
    args = ap.parse_args(argv)

    rep = compare(args.tol)
    published = [r for r in rep if r["status"] != "UNLISTED"]
    extra = [r for r in rep if r["status"] == "UNLISTED"]
    bad = [r for r in published if r["status"] != "PASS"]
    n_pass = sum(r["status"] == "PASS" for r in published)

    if args.json:
        print(json.dumps({"published_values": len(published), "reproduced": n_pass,
                          "failed": bad, "unlisted_recomputed": len(extra),
                          "base_tolerance": args.tol}, indent=1))
        return 0 if not bad else 1

    for t in TABLES:
        sub = [r for r in published if r["table"] == t]
        ok = sum(r["status"] == "PASS" for r in sub)
        print(f"\n{t}  {ok}/{len(sub)} published values reproduced")
        for r in sub:
            if r["status"] != "PASS":
                print(f"   {r['status']:8s} {r['key']:40s} expected {r['expected']} "
                      f"recomputed {r['recomputed']}")
    print(f"\n{n_pass}/{len(published)} published values reproduced "
          f"(base tolerance {args.tol:g}); {len(bad)} failing")
    if args.show_unlisted:
        print(f"{len(extra)} further values were recomputed but are not printed in the paper:")
        for r in extra:
            print(f"   {r['table']} {r['key']} = {r['recomputed']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
