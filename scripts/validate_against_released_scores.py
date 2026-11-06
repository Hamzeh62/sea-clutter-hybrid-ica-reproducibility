#!/usr/bin/env python
"""Level-2 check: re-run one released campaign job and compare it with the released scores.

A campaign job is identified by its id, e.g. ``h1|fold1|hybrid|c0``.  The script

1. rebuilds the detector's calibration preparation (``prep|<fold>|<variant>``) from the IPIX
   cube cache, so nothing is taken on trust from the archive;
2. runs the requested scoring job through the released runner;
3. compares every array with the released ``.npz`` for the same job.

Requirements: the IPIX cube cache (``data/README.md``), the released checkpoints
(``checkpoints/README.md``), and the extracted results archive that contains
``runs/minimal_v1_20261029/scores/``.

Thread count matters.  The campaign ran each worker with one BLAS and one torch thread; at
that setting the comparison is bit-exact.  With more threads the hybrid statistic moves by
about 1e-6 relative because float32 reductions are reassociated, so the script pins the
threads and says so.

Usage
-----
    python scripts/validate_against_released_scores.py \
        --released /path/to/paper1_results_v1.0.0/runs/minimal_v1_20261029 \
        --job "h1|fold1|hybrid|c0"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "pipeline" / "phase5_runner.py"

#: Arrays that must match exactly because they encode the frozen design, not a computation.
EXACT = {"trial_ids", "cells", "windows", "acqs", "start_samples", "fd", "phi", "scr_db"}
#: Tolerance on a computed statistic at one thread.  Bit-exact was observed; the small
#: allowance is for a different BLAS build, and it is reported, not hidden.
TOL = {"arm_hybrid": 1e-9, "arm_ar8": 1e-9, "arm_notch": 1e-9,
       "arm_anmf_dl": 1e-9, "arm_anmf_unloaded": 1e-9, "arm_glrt": 1e-9,
       "arm_ca": 1e-9, "arm_os": 1e-9, "arm_cold": 1e-9, "arm_ica_only": 1e-9}

PREP_OF_GROUP = {"hybrid": "hybrid", "nophys": "nophys", "cold": "cold", "ica": "ica",
                 "temporal": None, "fairness": None, "anmf": None, "cfar": None}


def run_job(job: str, outdir: Path, extra_env: dict) -> dict:
    env = {**os.environ, "PAPER1_OUTPUTS": str(outdir),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PHASE5_TORCH_THREADS": "1", **extra_env}
    r = subprocess.run([sys.executable, str(RUNNER), "run", "--job", job],
                       env=env, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"job {job} failed:\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def out_path(outdir: Path, job: str, cfg_name: str) -> Path:
    kind = job.split("|")[0]
    if kind == "prep":
        _, fold, variant = job.split("|")
        return outdir / "runs" / cfg_name / "preps" / f"{fold}__{variant}.npz"
    _, fold, group, chunk = job.split("|")
    # the runner writes the chunk *number*, zero-padded (c00), while the job id carries it
    # unpadded (c0)
    return outdir / "runs" / cfg_name / "scores" / "h1" / (
        f"{fold}__{group}__c{int(chunk.lstrip('c')):02d}.npz")


def compare(got: Path, want: Path) -> tuple[bool, list[dict]]:
    a, b = np.load(got), np.load(want)
    ok = sorted(a.files) == sorted(b.files)
    rows = []
    for k in sorted(a.files):
        if k not in b.files:
            rows.append({"array": k, "status": "MISSING"})
            ok = False
            continue
        if a[k].size == 0 and b[k].size == 0:
            rows.append({"array": k, "status": "EMPTY-IN-BOTH"})
            continue
        if a[k].dtype.kind in "US" or k in EXACT:
            same = a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
            rows.append({"array": k, "status": "EXACT" if same else "DIFFERS",
                         "max_abs_diff": 0.0 if same else None})
            ok &= same
            continue
        if a[k].shape != b[k].shape:
            rows.append({"array": k, "status": "SHAPE", "got": a[k].shape,
                         "want": b[k].shape})
            ok = False
            continue
        # keep complex arrays complex: casting to float would silently drop the imaginary
        # part of the ICA matrices and let a wrong preparation pass
        d = float(np.max(np.abs(a[k] - b[k])))
        rel = d / max(1e-30, float(np.max(np.abs(b[k]))))
        tol = TOL.get(k, 1e-9)
        passed = d <= tol
        rows.append({"array": k, "status": "PASS" if passed else "FAIL",
                     "max_abs_diff": d, "rel_diff": rel, "tolerance": tol})
        ok &= passed
    return ok, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True,
                    help="the released run directory holding preps/ and scores/")
    ap.add_argument("--job", default="h1|fold1|hybrid|c0")
    ap.add_argument("--keep", action="store_true", help="keep the temporary output tree")
    args = ap.parse_args(argv)

    released = Path(args.released)
    cfg = json.loads((ROOT / "configs" / "campaign.json").read_text(encoding="utf-8"))
    name = cfg["campaign"]

    if not args.job.startswith("h1|"):
        raise SystemExit("only h1 scoring jobs are comparable with released scores")
    _, fold, group, chunk = args.job.split("|")
    # the runner names the file after the chunk *number* (c00), while the job id carries
    # the number itself (c0); normalise the file name from the number
    want = released / "scores" / "h1" / f"{fold}__{group}__c{int(chunk.lstrip('c')):02d}.npz"
    if not want.is_file():
        raise SystemExit(f"released scores for {args.job} not found at {want}")

    tmp = Path(tempfile.mkdtemp(prefix="paper1_level2_"))
    print(f"working directory: {tmp}")
    try:
        variant = PREP_OF_GROUP.get(group)
        if variant:
            prep_job = f"prep|{fold}|{variant}"
            print(f"[1/3] rebuilding the calibration prep: {prep_job}")
            print("      ", run_job(prep_job, tmp, {}))
            rel_prep = released / "preps" / f"{fold}__{variant}.npz"
            if rel_prep.is_file():
                okp, rows = compare(out_path(tmp, prep_job, name), rel_prep)
                for r in rows:
                    print(f"      prep {r['array']:10s} {r['status']:14s} "
                          f"max|d|={r.get('max_abs_diff')}")
                if not okp:
                    print("      prep differs from the released preparation")
        else:
            print(f"[1/3] {group} arms use a prep built from CAL residual/notch rows; run "
                  f"'prep|<fold>|ar8' or 'prep|<fold>|notch' first if you need it")

        print(f"[2/3] running {args.job}")
        print("      ", run_job(args.job, tmp, {}))

        print("[3/3] comparing with the released scores")
        ok, rows = compare(out_path(tmp, args.job, name), want)
        for r in rows:
            extra = ""
            if r.get("rel_diff") is not None:
                extra = f"  rel={r['rel_diff']:.3e}  tol={r['tolerance']:.0e}"
            print(f"      {r['array']:16s} {r['status']:14s} "
                  f"max|d|={r.get('max_abs_diff')}{extra}")
        print("\nLEVEL 2 REPRODUCTION", "EXACT" if ok else "DIFFERS")
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
