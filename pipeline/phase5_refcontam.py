"""E3 sensitivity: the injected target is present in *both* polarisations.

The frozen protocol injects the Swerling-0 target into VV only.  The AR(8) and
two-tap-notch arms estimate their reference from the CUT's own VH channel, which is
therefore exactly target-free; a real target would raise VH as well.  That asymmetry
could manufacture the classical arms' advantage, so Phase 4 E3 asks for the
"reference contaminated by the injected target" upper bound.

This script re-scores the Hybrid, AR(8)+NMF and two-tap notch on the frozen H1 trials
of one fold at a single SCR, adding the *same* target instance to VV and VH.  Thresholds
are untouched (they come from CAL, which is H0 and therefore unaffected), so the
comparison to the frozen campaign is like-for-like on identical trials.

Usage:  python phase5_refcontam.py --fold fold1 --scr 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import phase5_common as C
import phase5_runner as R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=("fold1", "fold2"))
    ap.add_argument("--scr", type=float, default=0.0)
    ap.add_argument("--vh-gain-db", type=float, default=0.0,
                    help="target amplitude in VH relative to VV (0 = full leakage bound)")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = C.config()
    run = C.run_dir(cfg)
    fold = args.fold
    acqs = C.fold_acquisitions(cfg, fold)["test"]
    engine = R.Engine(cfg, fold, acqs)
    s = engine.sampler
    trials = C.h1_trials(fold, cfg)
    t0 = time.perf_counter()

    det = engine.det("hybrid")
    p_ar = engine.prep("ar8")
    p_no = engine.prep("notch")

    out = {a: np.empty(len(trials)) for a in ("hybrid", "ar8", "notch")}
    fds = np.empty(len(trials))
    sig_frac_vh = np.empty(len(trials))

    for lo in range(0, len(trials), args.chunk):
        hi = min(lo + args.chunk, len(trials))
        chunk = trials[lo:hi]
        xs = []
        for i, t in enumerate(chunk, start=lo):
            sig, fd, _phi, p_cl = C.make_target(t, s, engine.band(t.acq), args.scr)
            fds[i] = fd
            vv = s.data[t.acq][t.start:t.start + C.NP, t.cell].astype(np.complex128)
            vh = s.data_cross[t.acq][t.start:t.start + C.NP, t.cell].astype(np.complex128)
            # same injected instance in both channels; --vh-gain-db scales it in VH
            sig_vh = sig * 10.0 ** (args.vh_gain_db / 20.0)
            sig_frac_vh[i] = float(np.mean(np.abs(sig_vh) ** 2)
                                   / max(np.mean(np.abs(vh) ** 2), 1e-30))
            xs.append(np.stack([vv + sig, vh + sig_vh]))
        xs = np.stack(xs)
        nets = det.network_out_batch(xs, chunk=32)
        from src.proposed.m4_detector import nmf_statistic
        p_hy = engine.prep("hybrid")
        for k, (x, o) in enumerate(zip(xs, nets)):
            i = lo + k
            out["hybrid"][i] = det.statistic_from_net(x, o, prep_sep=p_hy,
                                                       warm_start=True)[0]
            vv, vh = x[0], x[1]
            out["ar8"][i] = nmf_statistic(C.ar_residual(vv, C.ar_fit(vh)), p_ar)
            out["notch"][i] = nmf_statistic(C.notch_vector(vv).conj() @ x, p_no)
        print(f"{fold} {hi}/{len(trials)}  {time.perf_counter() - t0:.0f}s", flush=True)

    dst = Path(args.out) if args.out else (run / "sensitivity")
    dst.mkdir(parents=True, exist_ok=True)
    tag = f"{fold}__refcontam__scr{args.scr:g}__vh{args.vh_gain_db:g}dB"
    npz = dst / f"{tag}.npz"
    payload = {f"arm_{a}": v for a, v in out.items()}
    payload.update({
        "trial_ids": np.array([t.trial_id for t in trials]),
        "cells": np.array([t.cell for t in trials], int),
        "windows": np.array([t.window for t in trials], int),
        "acqs": np.array([t.acq for t in trials]),
        "fd": fds,
        "scr_db": np.array([args.scr], float),
        "vh_target_power_ratio": sig_frac_vh,
        "condition": np.array([f"target_in_vv_and_vh@{args.vh_gain_db:g}dB"]),
    })
    digest = C.atomic_write_npz(npz, **payload)
    (dst / f"{tag}.done").write_text(json.dumps({
        "fold": fold, "scr_db": args.scr, "vh_gain_db": args.vh_gain_db,
        "n_trials": len(trials),
        "sha256": digest, "elapsed_s": round(time.perf_counter() - t0, 1),
        "generated": C.utc_now(), "config_sha256": cfg["config_sha256"],
    }, indent=2), encoding="utf-8")
    print(json.dumps({"fold": fold, "n": len(trials),
                      "elapsed_s": round(time.perf_counter() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
