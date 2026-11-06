"""Phase 7 V1: the mechanism curve as a function of delta-SCR, at fixed VV SCR.

Design is frozen in ``configs/PHASE7_SIGNOFF.json`` (``v1_design``).  In short:

  x_VV = c_VV + alpha s
  x_VH = c_VH + gamma e^{j phi} alpha s ,    gamma = 10^((dSCR - kappa_t)/20)

with ``kappa_t`` the trial's own measured VV/VH clutter ratio, so a given ``dSCR`` is the
same physical condition in every acquisition.  VV SCR is held at 0 dB exactly as in the
frozen campaign, so the VV-only condition is the campaign's own array rather than a rerun.

Conditions per fold (``--fold`` runs one fold, resumable):

  dSCR  -20 / -10 / 0  with one uniform phi draw per trial   (primary; all arms)
  dSCR    0  with phi = 0                                     (predictor's worst case)
  dSCR    0  energy control: alpha' = alpha/sqrt(1+gamma^2)   (total power = alpha^2)
  dSCR    0 / -10 with phi = 90 / 180 / 270 deg               (AR and notch only)
  ANMF once (its statistic never reads VH: a control, not a grid axis)

Outputs under ``phase7/data/v1/``: one npz per condition with per-trial scores, the realised
per-trial dSCR, and the frozen thresholds used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORK7 = PROJECT / "results" / "phase7"      # released Phase 7 data (read only)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase5_common as C                            # noqa: E402
import phase5_runner as R                            # noqa: E402

OUT = PROJECT / "outputs" / "phase7" / "data" / "v1"
CFG = C.config()
RUN = C.run_dir(CFG)
KAPPA = {r.split(",")[0]: float(r.split(",")[1]) for r in
         (WORK7 / "data/PHASE7_KAPPA.csv").read_text().splitlines()[1:] if r}

PRIMARY_DS = (-20.0, -10.0, 0.0)
SWEEP_PHASES = (90.0, 180.0, 270.0)
SWEEP_DS = (0.0, -10.0)


def phi_of(trial_id: str, mode: str) -> float:
    """Deterministic per-trial relative polarimetric phase."""
    if mode == "zero":
        return 0.0
    if mode.startswith("fixed"):
        return np.deg2rad(float(mode[5:]))
    h = hashlib.sha256(f"{trial_id}|v1|{mode}".encode()).digest()
    return 2 * np.pi * (int.from_bytes(h[:8], "little") % (2 ** 53)) / (2 ** 53)


def frozen_threshold(fold: str, group: str, arm: str) -> float:
    v = np.concatenate([np.load(p)[f"arm_{arm}"] for p in sorted(
        (RUN / "scores/h0").glob(f"{fold}__cal__{group}__*.npz"))])
    return float(np.quantile(v, 1.0 - C.DESIGN_PFA, method="higher"))


def arm_of(variant: str) -> tuple[str, str]:
    return {"hybrid": ("hybrid", "hybrid"), "ar8": ("temporal", "ar8"),
            "notch": ("temporal", "notch"), "anmf_dl": ("anmf", "anmf_dl")}[variant]


def run_condition(fold: str, tag: str, dscr: float | None, phi_mode: str,
                  energy: bool, arms: tuple[str, ...]) -> dict:
    t0 = time.perf_counter()
    acqs = C.fold_acquisitions(CFG, fold)["test"]
    eng = R.Engine(CFG, fold, acqs)
    s = eng.sampler
    trials = C.h1_trials(fold, CFG)
    n = len(trials)
    thr = {a: frozen_threshold(fold, *arm_of(a)) for a in arms}

    xs = np.empty((n, 2, C.NP), complex)
    realised = np.full(n, np.nan)
    vh_ratio = np.full(n, np.nan)
    vv_scr = np.full(n, np.nan)
    phis = np.zeros(n)
    gammas = np.ones(n)
    for i, t in enumerate(trials):
        sig, fd, _phi0, p_cl = C.make_target(t, s, eng.band(t.acq), 0.0)
        vv = s.data[t.acq][t.start:t.start + C.NP, t.cell].astype(np.complex128)
        vh = s.data_cross[t.acq][t.start:t.start + C.NP, t.cell].astype(np.complex128)
        if dscr is None:                       # ANMF / VH-invariant arms
            xs[i] = np.stack([vv, vh])
            gammas[i] = 0.0
            continue
        kappa_t = 10 * np.log10(np.mean(np.abs(vv) ** 2) / np.mean(np.abs(vh) ** 2))
        gamma = 10 ** ((dscr - kappa_t) / 20.0)
        phi = phi_of(t.trial_id, phi_mode)
        gammas[i], phis[i] = gamma, phi
        a = sig / np.sqrt(1.0 + gamma ** 2) if energy else sig
        x_vv, x_vh = vv + a, vh + gamma * np.exp(1j * phi) * a
        xs[i] = np.stack([x_vv, x_vh])
        # every quantity below is measured against the CLEAN clutter of this CPI, so the
        # condition is verified rather than restated from the injection formula
        p_vv, p_vh = float(np.mean(np.abs(vv) ** 2)), float(np.mean(np.abs(vh) ** 2))
        p_t = float(np.mean(np.abs(a) ** 2))
        vh_ratio[i] = gamma ** 2 * p_t / p_vh
        vv_scr[i] = 10 * np.log10(p_t / p_vv)
        realised[i] = 10 * np.log10(vh_ratio[i]) - vv_scr[i]

    out = {"trial_ids": np.array([t.trial_id for t in trials]),
           "acqs": np.array([t.acq for t in trials]),
           "cells": np.array([t.cell for t in trials], int),
           "windows": np.array([t.window for t in trials], int),
           "dSCR_design": np.array([np.nan if dscr is None else dscr]),
           "dSCR_realised": realised, "gamma": gammas, "phi_rad": phis,
           "vh_target_to_clutter": vh_ratio, "vv_scr_db": vv_scr,
           "energy_control": np.array([energy])}

    if "hybrid" in arms:
        det = eng.det("hybrid")
        prep = eng.prep("hybrid")
        nets = det.network_out_batch(xs, chunk=64)
        sc = np.empty(n)
        for i, (x, o) in enumerate(zip(xs, nets)):
            sc[i] = det.statistic_from_net(x, o, prep_sep=prep, warm_start=True)[0]
        out["arm_hybrid"] = sc
    if "ar8" in arms or "notch" in arms:
        from src.proposed.m4_detector import nmf_statistic
        if "ar8" in arms:
            p_ar = eng.prep("ar8")
            out["arm_ar8"] = np.array([
                nmf_statistic(C.ar_residual(x[0], C.ar_fit(x[1])), p_ar) for x in xs])
        if "notch" in arms:
            p_no = eng.prep("notch")
            out["arm_notch"] = np.array([
                nmf_statistic(C.notch_vector(x[0]).conj() @ x, p_no) for x in xs])
    if "anmf_dl" in arms:
        P = eng.P
        sc = np.empty(n)
        for i, t in enumerate(trials):
            sec = C.anmf_secondary_windows(s, t.acq, t.cell, t.window, C.K_SEC, 4)
            prep_a = C.anmf_preps(sec, P)
            if prep_a is None:
                raise RuntimeError(f"ANMF prep unavailable at {t.trial_id}")
            base = s.data[t.acq][t.start:t.start + C.NP, t.cell].astype(np.complex128)
            sig, _, _, _ = C.make_target(t, s, eng.band(t.acq), 0.0)
            sc[i] = C.anmf_scores(base + sig, prep_a)[0]
        out["arm_anmf_dl"] = sc

    out["thresholds"] = np.array([thr[a] for a in arms])
    out["threshold_arms"] = np.array(list(arms))
    digest = C.atomic_write_npz(OUT / f"{fold}__{tag}.npz", **out)
    (OUT / f"{fold}__{tag}.done").write_text(json.dumps({
        "fold": fold, "tag": tag, "dSCR": dscr, "phi_mode": phi_mode,
        "energy_control": energy, "arms": list(arms), "n": n,
        "thresholds": thr, "sha256": digest,
        "realised_dSCR_median": float(np.nanmedian(realised)) if np.isfinite(
            realised).any() else None,
        "realised_dSCR_max_abs_dev": float(np.nanmax(np.abs(realised - dscr)))
        if dscr is not None and np.isfinite(realised).any() else None,
        "wall_s": round(time.perf_counter() - t0, 1),
        "config_sha256": CFG["config_sha256"],
        "signoff_sha256": json.loads((WORK7 / "configs/PHASE7_SIGNOFF.json")
                                     .read_text(encoding="utf-8"))["manifest_sha256"],
    }, indent=2), encoding="utf-8")
    return {"tag": tag, "wall_s": round(time.perf_counter() - t0, 1), "thresholds": thr}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=("fold1", "fold2"))
    ap.add_argument("--smoke", type=int, default=0,
                    help="restrict to N trials and skip the resumable skip-check")
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    fold = args.fold

    jobs = []
    for d in PRIMARY_DS:
        jobs.append((f"dSCR{d:g}__randphi", d, "rand", False, ("hybrid", "ar8", "notch")))
    jobs.append(("dSCR0__phi0", 0.0, "zero", False, ("hybrid", "ar8", "notch")))
    jobs.append(("dSCR0__energy", 0.0, "rand", True, ("hybrid", "ar8", "notch")))
    for d in SWEEP_DS:
        for ph in SWEEP_PHASES:
            jobs.append((f"dSCR{d:g}__phi{ph:g}", d, f"fixed{ph:g}", False, ("ar8", "notch")))
    jobs.append(("anmf__scr0", None, "n/a", False, ("anmf_dl",)))

    for tag, d, mode, energy, arms in jobs:
        if args.only and args.only != tag:
            continue
        if (OUT / f"{fold}__{tag}.done").is_file() and not args.smoke:
            print(f"[skip] {fold} {tag}", flush=True)
            continue
        if args.smoke:
            global C_h1
            r = run_condition(fold, tag + "__smoke", d, mode, energy, arms)
        else:
            r = run_condition(fold, tag, d, mode, energy, arms)
        print(f"[done] {fold} {r['tag']} {r['wall_s']}s thr={r['thresholds']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
