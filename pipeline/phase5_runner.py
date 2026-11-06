"""Phase 5 minimal-campaign runner.

One process executes exactly one job.  Jobs are enumerated deterministically from the
frozen Phase 4 manifests and the campaign config, so the supervisor never invents work
and a rerun never changes trial identity.

Job families
------------
``prep``  build a detector's null covariance from 250 CAL windows (per fold, per variant)
``h0``    score one (fold, partition, group, acquisition) block of eligible windows
``h1``    score all frozen paired H1 trials of one (fold, group), in 250-trial chunks

Every output is written atomically as ``.npz`` + ``.done`` (JSON with the SHA-256 of the
array file and the hashed job specification).  A job whose ``.done`` record verifies is
never recomputed; a job whose record fails verification is a hard error, not a rebuild.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import phase5_common as C

CHUNK = 250


# --------------------------------------------------------------------- job spec
def job_specs(cfg: dict) -> list[dict]:
    specs: list[dict] = []
    grid = {g: cfg["e1"]["scr_db"] for g in ("hybrid", "anmf", "cfar", "temporal")}
    grid["fairness"] = cfg["e3"]["scr_db"]
    for g in ("nophys", "cold", "ica"):
        grid[g] = cfg["e2"]["scr_db"]

    for fold in ("fold1", "fold2"):
        for variant in C.PREP_VARIANTS:
            specs.append(dict(kind="prep", fold=fold, variant=variant))
    for fold in ("fold1", "fold2"):
        acqs = C.fold_acquisitions(cfg, fold)
        for part, key in (("cal", "cal"), ("test_h0", "test")):
            for group in C.GROUPS:
                for acq in sorted(acqs[key]):
                    specs.append(dict(kind="h0", fold=fold, partition=part,
                                      group=group, acq=acq))
    for fold in ("fold1", "fold2"):
        n_trials = len(C.h1_trials(fold, cfg))
        n_chunks = int(np.ceil(n_trials / CHUNK))
        for group in C.GROUPS:
            for k in range(n_chunks):
                specs.append(dict(kind="h1", fold=fold, group=group, chunk=k,
                                  scr_db=grid[group]))
    for i, s in enumerate(specs):
        s["index"] = i
        s["id"] = job_id(s)
    return specs


def job_id(spec: dict) -> str:
    k = spec["kind"]
    if k == "prep":
        return f"prep|{spec['fold']}|{spec['variant']}"
    if k == "h0":
        return f"h0|{spec['fold']}|{spec['partition']}|{spec['group']}|{spec['acq']}"
    return f"h1|{spec['fold']}|{spec['group']}|c{spec['chunk']}"


def safe(name: str) -> str:
    return name.replace("|", "__")


def out_paths(run: Path, spec: dict) -> tuple[Path, Path]:
    if spec["kind"] == "prep":
        p = run / "preps" / f"{spec['fold']}__{spec['variant']}.npz"
    elif spec["kind"] == "h0":
        p = run / "scores" / "h0" / (f"{spec['fold']}__{spec['partition']}__"
                                    f"{spec['group']}__{spec['acq']}.npz")
    else:
        p = run / "scores" / "h1" / (f"{spec['fold']}__{spec['group']}__"
                                    f"c{spec['chunk']:02d}.npz")
    return p, p.with_suffix(".done")


def spec_hash(spec: dict, cfg: dict) -> str:
    body = {k: v for k, v in spec.items() if k not in ("index",)}
    return C.sha256_bytes(json.dumps({"spec": body, "config": cfg["config_sha256"]},
                                     sort_keys=True).encode())


def is_done(run: Path, spec: dict, cfg: dict) -> tuple[bool, str]:
    npz, done = out_paths(run, spec)
    if not done.is_file() or not npz.is_file():
        return False, "missing"
    rec = C.read_json(done)
    if rec.get("spec_hash") != spec_hash(spec, cfg):
        return False, "spec_changed"
    if rec.get("sha256") != C.sha256_file(npz):
        return False, "hash_mismatch"
    return True, "ok"


# ------------------------------------------------------------------- detection
class Engine:
    """Lazily built scoring machinery for the groups a job needs."""

    def __init__(self, cfg: dict, fold: str, acqs: list[str]):
        self.cfg, self.fold = cfg, fold
        self.table = C.canonical_table(cfg)
        self.sampler = C.load_sampler(sorted(acqs), self.table)
        self.run = C.run_dir(cfg)
        self._dets: dict[str, object] = {}
        self._P = None
        self._bands: dict[str, tuple[float, float]] = {}
        self._preps: dict[str, dict] = {}

    # -- lazy pieces -------------------------------------------------------
    @property
    def P(self):
        if self._P is None:
            from src.baselines.anmf import steering_bank
            self._P = steering_bank(C.NP)
        return self._P

    def det(self, variant: str):
        if variant not in self._dets:
            import torch
            torch.set_num_threads(int(os.environ.get("PHASE5_TORCH_THREADS", "1")))
            from src.proposed.infer import HybridDetector
            self._dets[variant] = HybridDetector(C.CHECKPOINTS[
                "nophys" if variant == "nophys" else "hybrid"])
        return self._dets[variant]

    def band(self, acq: str):
        if acq not in self._bands:
            self._bands[acq] = C.clear_band(self.sampler, acq)
        return self._bands[acq]

    def prep(self, variant: str) -> dict:
        if variant in self._preps:
            return self._preps[variant]
        path = self.run / "preps" / f"{self.fold}__{variant}.npz"
        if not path.is_file():
            raise RuntimeError(f"missing prep {path}; run the prep job first")
        with np.load(path) as z:
            p = {k: z[k] for k in z.files}
        self._preps[variant] = p
        return p

    def x_of(self, acq: str, cell: int, w0: int, inj=None) -> np.ndarray:
        vv = self.sampler.data[acq][w0:w0 + C.NP, cell].astype(np.complex128)
        vh = self.sampler.data_cross[acq][w0:w0 + C.NP, cell].astype(np.complex128)
        if inj is not None:
            vv = vv + inj
        return np.stack([vv, vh])


# ---------------------------------------------------------------------- scoring
def score_h0(engine: Engine, group: str, units: list[C.Unit]) -> dict[str, np.ndarray]:
    s = engine.sampler
    n = len(units)
    out = {a: np.empty(n) for a in C.ARMS[group]}
    arms = C.ARMS[group]

    if group in ("hybrid", "nophys", "cold", "ica"):
        variant = {"hybrid": "hybrid", "nophys": "nophys", "cold": "cold", "ica": "ica"}[group]
        det = engine.det(variant)
        prep = engine.prep(variant)
        xs = np.stack([engine.x_of(u.acq, u.cell, u.start) for u in units])
        if group == "ica":
            for i, x in enumerate(xs):
                out["ica_only"][i] = det.statistic(x, prep_sep=prep, use_network=False,
                                                   warm_start=False)[0]
        else:
            outs = det.network_out_batch(xs, chunk=64)
            for i, (x, o) in enumerate(zip(xs, outs)):
                out[arms[0]][i] = det.statistic_from_net(
                    x, o, prep_sep=prep, warm_start=(group != "cold"))[0]
        return out

    if group == "anmf":
        for i, u in enumerate(units):
            sec = C.anmf_secondary_windows(s, u.acq, u.cell, u.window, C.K_SEC, 4)
            prep = C.anmf_preps(sec, engine.P)
            if prep is None:
                raise RuntimeError(f"ANMF prep unavailable at {u}")
            y = s.data[u.acq][u.start:u.start + C.NP, u.cell].astype(np.complex128)
            dl, un, gl = C.anmf_scores(y, prep)
            out["anmf_dl"][i], out["anmf_unloaded"][i], out["glrt"][i] = dl, un, gl
        return out

    if group == "cfar":
        from src.baselines.cfar import ca_cfar_stat, os_cfar_stat
        for i, u in enumerate(units):
            cells, ci = s.range_block(u.acq, u.cell, u.start, C.CF["guard_r"] + C.CF["train_r"],
                                      inject_cell=u.cell)
            kw = dict(train_r=C.CF["train_r"], guard_r=C.CF["guard_r"],
                      train_f=C.CF["train_f"], guard_f=C.CF["guard_f"])
            out["ca"][i] = ca_cfar_stat(cells, ci, **kw)
            out["os"][i] = os_cfar_stat(cells, ci, **kw)
        return out

    if group in ("temporal", "fairness"):
        from src.proposed.m4_detector import nmf_statistic
        p_ar = engine.prep("ar8") if "ar8" in arms else None
        p_arvv = engine.prep("ar8vv") if "ar8_vvfit" in arms else None
        p_no = engine.prep("notch") if ("notch" in arms or "notch_clean" in arms) else None
        for i, u in enumerate(units):
            x = engine.x_of(u.acq, u.cell, u.start)
            vv, vh = x[0], x[1]
            if "ar8" in arms:
                out["ar8"][i] = nmf_statistic(C.ar_residual(vv, C.ar_fit(vh)), p_ar)
            if "ar8_vvfit" in arms:
                out["ar8_vvfit"][i] = nmf_statistic(C.ar_residual(vv, C.ar_fit(vv)), p_arvv)
            if "notch" in arms:
                out["notch"][i] = nmf_statistic(C.notch_vector(vv).conj() @ x, p_no)
            if "notch_clean" in arms:
                out["notch_clean"][i] = nmf_statistic(C.notch_vector(vv).conj() @ x, p_no)
        return out

    raise ValueError(group)


def score_h1(engine: Engine, group: str, trials: list[C.Trial],
             scr_grid: list[float]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Scores for every trial at every SCR of the group's grid."""
    s = engine.sampler
    n, ns = len(trials), len(scr_grid)
    out = {a: np.empty((n, ns)) for a in C.ARMS[group]}
    fds = np.empty(n)
    phis = np.empty(n)

    def target(t, scr):
        return C.make_target(t, s, engine.band(t.acq), scr)

    arms = C.ARMS[group]

    if group in ("hybrid", "nophys", "cold", "ica"):
        variant = {"hybrid": "hybrid", "nophys": "nophys", "cold": "cold", "ica": "ica"}[group]
        det = engine.det(variant)
        prep = engine.prep(variant)
        for j, scr in enumerate(scr_grid):
            xs, meta = [], []
            for i, t in enumerate(trials):
                sig, fd, phi, _ = target(t, scr)
                if j == 0:
                    fds[i], phis[i] = fd, phi
                xs.append(engine.x_of(t.acq, t.cell, t.start, inj=sig))
            xs = np.stack(xs)
            if group == "ica":
                for i, x in enumerate(xs):
                    out["ica_only"][i, j] = det.statistic(
                        x, prep_sep=prep, use_network=False, warm_start=False)[0]
            else:
                outs = det.network_out_batch(xs, chunk=64)
                for i, (x, o) in enumerate(zip(xs, outs)):
                    out[arms[0]][i, j] = det.statistic_from_net(
                        x, o, prep_sep=prep, warm_start=(group != "cold"))[0]
        return out, fds, phis

    if group == "anmf":
        for i, t in enumerate(trials):
            sig, fd, phi, _ = target(t, scr_grid[0])
            fds[i], phis[i] = fd, phi
            sec = C.anmf_secondary_windows(s, t.acq, t.cell, t.window, C.K_SEC, 4)
            prep = C.anmf_preps(sec, engine.P)
            if prep is None:
                raise RuntimeError(f"ANMF prep unavailable at {t.trial_id}")
            base = s.data[t.acq][t.start:t.start + C.NP, t.cell].astype(np.complex128)
            for j, scr in enumerate(scr_grid):
                s2, _, _, _ = target(t, scr)
                y = base + s2
                dl, un, gl = C.anmf_scores(y, prep)
                out["anmf_dl"][i, j], out["anmf_unloaded"][i, j], out["glrt"][i, j] = dl, un, gl
        return out, fds, phis

    if group == "cfar":
        from src.baselines.cfar import ca_cfar_stat, os_cfar_stat
        kw = dict(train_r=C.CF["train_r"], guard_r=C.CF["guard_r"],
                  train_f=C.CF["train_f"], guard_f=C.CF["guard_f"])
        for i, t in enumerate(trials):
            sig0, fd, phi, _ = target(t, scr_grid[0])
            fds[i], phis[i] = fd, phi
            for j, scr in enumerate(scr_grid):
                sig = sig0 * (10 ** ((scr - scr_grid[0]) / 20.0))
                cells, ci = s.range_block(t.acq, t.cell, t.start,
                                          C.CF["guard_r"] + C.CF["train_r"],
                                          inject_cell=t.cell, inject=lambda c: c + sig)
                out["ca"][i, j] = ca_cfar_stat(cells, ci, **kw)
                out["os"][i, j] = os_cfar_stat(cells, ci, **kw)
        return out, fds, phis

    if group in ("temporal", "fairness"):
        from src.proposed.m4_detector import nmf_statistic
        p_ar = engine.prep("ar8") if "ar8" in arms else None
        p_arvv = engine.prep("ar8vv") if "ar8_vvfit" in arms else None
        p_no = engine.prep("notch") if ("notch" in arms or "notch_clean" in arms) else None
        for i, t in enumerate(trials):
            x0 = engine.x_of(t.acq, t.cell, t.start)
            vv0, vh = x0[0], x0[1]
            a8 = C.ar_fit(vh) if "ar8" in arms else None
            w_no = C.notch_vector(vv0) if "notch" in arms else None
            w_cl = C.notch_vector(vv0) if "notch_clean" in arms else None
            for j, scr in enumerate(scr_grid):
                sig, fd, phi, _ = target(t, scr)
                if j == 0:
                    fds[i], phis[i] = fd, phi
                vv = vv0 + sig
                if "ar8" in arms:
                    out["ar8"][i, j] = nmf_statistic(C.ar_residual(vv, a8), p_ar)
                if "ar8_vvfit" in arms:
                    out["ar8_vvfit"][i, j] = nmf_statistic(
                        C.ar_residual(vv, C.ar_fit(vv)), p_arvv)
                if "notch" in arms:
                    out["notch"][i, j] = nmf_statistic(
                        C.notch_vector(vv).conj() @ np.stack([vv, vh]), p_no)
                if "notch_clean" in arms:
                    out["notch_clean"][i, j] = nmf_statistic(
                        w_cl.conj() @ np.stack([vv, vh]), p_no)
        return out, fds, phis

    raise ValueError(group)


# --------------------------------------------------------------------- prep job
def run_prep(cfg: dict, spec: dict) -> dict:
    from src.proposed.m4_detector import nmf_statistic, prep_sep_from_windows
    fold, variant = spec["fold"], spec["variant"]
    acqs = C.fold_acquisitions(cfg, fold)
    units_all = C.partition_units(fold, "CAL_H0", cfg)
    n = int(cfg["c_sep"]["n_windows"])
    picked = C.stratified_subsample(units_all, n, cfg["seeds"]["csep"] + int(fold[-1]))
    engine = Engine(cfg, fold, acqs["cal"])
    s = engine.sampler
    rows = []
    if variant in ("hybrid", "nophys", "cold", "ica"):
        det = engine.det(variant)
        for u in picked:
            x = engine.x_of(u.acq, u.cell, u.start)
            if variant == "ica":
                rows.append(det.statistic(x, use_network=False, warm_start=False)[1]["s_T"])
            else:
                rows.append(det.statistic(x, warm_start=(variant != "cold"))[1]["s_T"])
    elif variant in ("ar8", "ar8vv"):
        for u in picked:
            x = engine.x_of(u.acq, u.cell, u.start)
            ref = x[1] if variant == "ar8" else x[0]
            rows.append(C.ar_residual(x[0], C.ar_fit(ref)))
    elif variant == "notch":
        for u in picked:
            x = engine.x_of(u.acq, u.cell, u.start)
            rows.append(C.notch_vector(x[0]).conj() @ x)
    else:
        raise ValueError(variant)
    prep = prep_sep_from_windows(np.stack(rows))
    return {"prep": prep, "n_windows": len(picked),
            "units": np.array([[u.cell, u.start] for u in picked], int),
            "acqs": np.array([u.acq for u in picked])}


# -------------------------------------------------------------------- execution
def run_job(cfg: dict, spec: dict, run: Path) -> dict:
    t0 = time.perf_counter()
    base = {"job_id": spec["id"], "spec": spec, "host_pid": os.getpid(),
            "started": C.utc_now()}
    if spec["kind"] == "prep":
        payload = run_prep(cfg, spec)
    elif spec["kind"] == "h0":
        units = C.partition_units(spec["fold"], spec["partition"], cfg)
        pool = C.pool_for(spec["group"], units, cfg, spec["fold"], spec["partition"])
        mine = [u for u in pool if u.acq == spec["acq"]]
        if not mine:
            raise RuntimeError(f"empty pool for {spec}")
        acqs = [spec["acq"]]
        engine = Engine(cfg, spec["fold"], acqs)
        dropped: list[C.Unit] = []
        if spec["group"] == "cfar":
            # a cell whose training window is consumed by the guard has no CA/OS
            # denominator at all; it is withdrawn from every arm's pool by the
            # analysis, so the compared unit set stays identical
            keep = [u for u in mine if C.cfar_evaluable(engine.sampler, u.acq, u.cell)]
            dropped = [u for u in mine if not C.cfar_evaluable(engine.sampler, u.acq, u.cell)]
            if not keep:
                raise RuntimeError(f"no CFAR-evaluable unit in {spec['id']}")
            mine = keep
        scores = score_h0(engine, spec["group"], mine)
        payload = {"scores": scores,
                   "cells": np.array([u.cell for u in mine], int),
                   "windows": np.array([u.window for u in mine], int),
                   "n_pool_total": np.array([len(pool)], int),
                   "n_partition": np.array([len(units)], int),
                   "excluded_cells": np.array([u.cell for u in dropped], int),
                   "excluded_windows": np.array([u.window for u in dropped], int)}
    elif spec["kind"] == "h1":
        trials = C.h1_trials(spec["fold"], cfg)
        lo, hi = spec["chunk"] * CHUNK, min((spec["chunk"] + 1) * CHUNK, len(trials))
        chunk = trials[lo:hi]
        acqs = C.fold_acquisitions(cfg, spec["fold"])["test"]
        engine = Engine(cfg, spec["fold"], acqs)
        dropped_ids: list[str] = []
        if spec["group"] == "cfar":
            keep_t = [t for t in chunk if C.cfar_evaluable(engine.sampler, t.acq, t.cell)]
            dropped_ids = [t.trial_id for t in chunk if t not in keep_t]
            if not keep_t:
                raise RuntimeError(f"no CFAR-evaluable trial in {spec['id']}")
            chunk = keep_t
        scores, fds, phis = score_h1(engine, spec["group"], chunk, spec["scr_db"])
        payload = {"scores": scores, "fd": fds, "phi": phis,
                   "trial_ids": np.array([t.trial_id for t in chunk]),
                   "cells": np.array([t.cell for t in chunk], int),
                   "windows": np.array([t.window for t in chunk], int),
                   "start_samples": np.array([t.start for t in chunk], int),
                   "acqs": np.array([t.acq for t in chunk]),
                   "scr_db": np.array(spec["scr_db"], float),
                   "excluded_trial_ids": np.array(dropped_ids)}
    else:
        raise ValueError(spec["kind"])

    flat: dict[str, np.ndarray] = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            for arm, arr in v.items():
                key = arm if spec["kind"] == "prep" else f"arm_{arm}"
                flat[key] = np.asarray(arr)
                if not np.all(np.isfinite(flat[key])):
                    raise FloatingPointError(f"non-finite value in {spec['id']} {key}")
        else:
            flat[k] = np.asarray(v)

    npz, done = out_paths(run, spec)
    digest = C.atomic_write_npz(npz, **flat)
    elapsed = time.perf_counter() - t0
    record = {**base, "sha256": digest, "spec_hash": spec_hash(spec, cfg),
              "finished": C.utc_now(), "wall_s": round(elapsed, 2),
              "peak_rss_mb": round(C.peak_rss_mb(), 1),
              "n_units": int(len(flat.get("cells", []))),
              "arrays": {k: list(v.shape) for k, v in flat.items()}}
    C.atomic_write_json(done, record)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "list", "status", "verify"])
    ap.add_argument("--job", help="job id for run")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = C.config()
    run = C.run_dir(cfg)
    specs = job_specs(cfg)

    if args.command == "list":
        cols = ["index", "id", "kind"]
        if args.json:
            print(json.dumps([{k: s[k] for k in cols} for s in specs], indent=1))
        else:
            for s in specs:
                print(f"{s['index']:4d}  {s['id']}")
        return 0

    if args.command == "status":
        rows = []
        for s in specs:
            ok, why = is_done(run, s, cfg)
            rows.append({"index": s["index"], "id": s["id"], "done": ok, "why": why})
        n_done = sum(r["done"] for r in rows)
        print(json.dumps({"done": n_done, "total": len(rows),
                          "pending": [r for r in rows if not r["done"]][:10]}, indent=1)
                    if args.json else
                    f"{n_done}/{len(rows)} jobs complete")
        return 0

    if args.command == "verify":
        bad = []
        for s in specs:
            ok, why = is_done(run, s, cfg)
            if not ok:
                bad.append({"id": s["id"], "why": why})
        print(json.dumps({"verified": len(specs) - len(bad), "failed": bad[:20],
                          "n_failed": len(bad)}, indent=1))
        return 0 if not bad else 1

    if args.command == "run":
        if not args.job:
            raise SystemExit("--job is required")
        match = [s for s in specs if s["id"] == args.job]
        if not match:
            raise SystemExit(f"unknown job id {args.job!r}")
        spec = match[0]
        ok, why = is_done(run, spec, cfg)
        if ok:
            print(json.dumps({"job": spec["id"], "status": "already_done"}))
            return 0
        lock = run / "logs" / (safe(spec["id"]) + ".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print(json.dumps({"job": spec["id"], "status": "locked"}))
            return 3
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            log = run / "logs" / (safe(spec["id"]) + ".log")
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"\n=== {C.utc_now()} start {spec['id']} pid {os.getpid()}\n")
            try:
                rec = run_job(cfg, spec, run)
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(f"done wall={rec['wall_s']}s rss={rec['peak_rss_mb']}MB "
                             f"n={rec['n_units']}\n")
                print(json.dumps({"job": spec["id"], "status": "done",
                                  "wall_s": rec["wall_s"]}))
                return 0
            except Exception:
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(traceback.format_exc())
                raise
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
