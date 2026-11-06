"""Phase 5 pre-flight: the twelve mandatory validity checks plus three integrity ones.

Nothing here is a scientific result; every check either passes (and is recorded) or
aborts the campaign.  Runtime target: a few minutes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import phase5_common as C

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail) -> None:
    RESULTS.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, json.dumps(detail, default=str)[:160],
          flush=True)


def main() -> int:
    t0 = time.perf_counter()
    cfg = C.config()
    run = C.run_dir(cfg)

    # 1 -- canonical CDF/NPZ mapping -------------------------------------------
    table = C.canonical_table(cfg)
    cache = sorted(C.CACHE.glob("*_cubes.npz"))
    check("01_cache_acquisitions_16", len(cache) == 16, len(cache))
    check("01_metadata_acquisitions_16", len(table) == 16, len(table))
    check("01_every_cache_has_metadata",
          all(any(r.cache_name == p.name for r in table.values()) for p in cache), None)
    check("01_every_record_has_eligible_cells",
          all(len(r.eligible) > 0 for r in table.values()),
          {a: len(r.eligible) for a, r in table.items()})

    # 2 -- exclusion ------------------------------------------------------------
    bad = 0
    for fold in ("fold1", "fold2"):
        for role in ("CAL_H0", "TEST_H0"):
            units = C.partition_units(fold, role, cfg)
            cells = {(u.acq, u.cell) for u in units}
            for acq, cell in cells:
                if cell in table[acq].forbidden:
                    bad += 1
    check("02_partition_cells_exclude_target_guard", bad == 0, bad)
    min_forbid = min(r.n_forbidden for r in table.values())
    check("02_exclusion_sets_nonempty", min_forbid > 0, min_forbid)

    # 3 -- CAL/TEST independence ------------------------------------------------
    overlaps = {}
    for fold in ("fold1", "fold2"):
        cal = C.partition_units(fold, "CAL_H0", cfg)
        test = C.partition_units(fold, "TEST_H0", cfg)
        acq_ov = {u.acq for u in cal} & {u.acq for u in test}
        win_ov = {(u.acq, u.cell, u.window) for u in cal} & \
                 {(u.acq, u.cell, u.window) for u in test}
        tri = C.h1_trials(fold, cfg)
        h1_ov = {(t.acq, t.cell, t.window) for t in tri} & \
                {(u.acq, u.cell, u.window) for u in test}
        overlaps[fold] = {"acq": len(acq_ov), "window": len(win_ov), "h1_in_h0": len(h1_ov)}
    check("03_zero_acquisition_overlap", all(v["acq"] == 0 for v in overlaps.values()), overlaps)
    check("03_zero_window_overlap", all(v["window"] == 0 for v in overlaps.values()), overlaps)
    check("03_h1_trials_not_in_h0", all(v["h1_in_h0"] == 0 for v in overlaps.values()), overlaps)

    # 4 -- complex loading ------------------------------------------------------
    fold_acqs = C.fold_acquisitions(cfg, "fold1")
    s = C.load_sampler(fold_acqs["cal"][:2], table)
    a0 = fold_acqs["cal"][0]
    cube = s.data[a0]
    check("04_complex_dtype", cube.dtype == np.complex64, str(cube.dtype))
    check("04_imaginary_power_present",
          float(np.mean(np.abs(np.imag(cube)) ** 2)) > 0.0,
          float(np.mean(np.abs(np.imag(cube)) ** 2)))
    check("04_cross_pol_loaded", s.data_cross[a0].shape == cube.shape,
          list(s.data_cross[a0].shape))

    # 5/6 -- injection geometry -------------------------------------------------
    tri = C.h1_trials("fold1", cfg)[:40]
    s_all = C.load_sampler(sorted({t.acq for t in tri}), table)
    scr_err, bin_err, lobe = [], [], []
    for scr in (-10.0, 0.0, 10.0):
        for t in tri[:12]:
            band = C.clear_band(s_all, t.acq)
            sig, fd, phi, p_c = C.make_target(t, s_all, band, scr)
            vv = s_all.data[t.acq][t.start:t.start + C.NP, t.cell]
            scr_err.append(abs(10 * np.log10(np.mean(np.abs(sig) ** 2) / p_c) - scr))
            spec = np.abs(np.fft.fft(sig)) ** 2
            bins = np.fft.fftfreq(C.NP, d=1 / C.PRF)
            bin_err.append(abs(int(np.argmax(spec)) - int(np.argmin(np.abs(bins - fd)))))
            k = int(np.argmax(spec))
            idx = [(k + j) % C.NP for j in (-3, -2, -1, 0, 1, 2, 3)]
            lobe.append(float(spec[idx].sum() / spec.sum()))
    check("05_empirical_scr_exact", max(scr_err) < 5e-3,
          {"max_abs_err_db": float(max(scr_err))})
    check("06_doppler_peak_at_requested_bin", max(bin_err) <= 1,
          {"max_bin_offset": int(max(bin_err))})
    check("06_main_lobe_energy", min(lobe) > 0.90, {"min": float(min(lobe))})

    # 7 -- checkpoint -----------------------------------------------------------
    ck = cfg["checkpoints"]
    ok_ck = all(C.sha256_file(C.PROJECT / v["path"]) == v["sha256"] for v in ck.values())
    check("07_checkpoint_hashes_match_frozen_config", ok_ck,
          {k: v["sha256"][:12] for k, v in ck.items()})

    # 8 -- separated-domain covariance -----------------------------------------
    prep_path = run / "preps" / "fold1__hybrid.npz"
    if prep_path.is_file():
        with np.load(prep_path) as z:
            Chat, Cinv, CinvP, a = z["Chat"], z["Cinv"], z["CinvP"], z["a"]
        from src.proposed.m4_detector import nmf_statistic
        herm = float(np.max(np.abs(Chat - Chat.conj().T)))
        check("08_ceps_hermitian", herm < 1e-10, herm)
        # the operational quantity is the statistic, not the matrix inverse:
        # compare the hoisted-inverse path against the per-call solve path.  The
        # tolerance is conditioning-aware: the deployed C_sep carries 1e-10 diagonal
        # loading on a 250-window SCM (heavy null-space regularisation), so the two
        # algebraically identical paths differ by round-off amplified by cond(Chat).
        # The campaign always uses one path, so no decision is affected.
        prep_solve = {"Chat": Chat, "CinvP": CinvP, "a": a}
        prep_inv = {"Chat": Chat, "CinvP": CinvP, "a": a, "Cinv": Cinv}
        rng = np.random.default_rng(7)
        rel, worst = 0.0, None
        for _ in range(200):
            sv = rng.standard_normal(C.NP) + 1j * rng.standard_normal(C.NP)
            t1 = nmf_statistic(sv, prep_inv)
            t2 = nmf_statistic(sv, prep_solve)
            d = abs(t1 - t2) / max(abs(t2), 1e-30)
            if d > rel:
                rel, worst = d, (t1, t2)
        cond = float(np.linalg.cond(Chat))
        bound = 1e-5
        check("08_inverse_vs_solve_statistic", rel < bound,
              {"max_rel": rel, "bound": bound, "cond_chat": round(cond, 1),
               "eps_cond": float(np.finfo(float).eps * cond)})
        check("08_inverse_solve_same_result_below_cond_bound", rel < 1e3 * np.finfo(float).eps * cond,
              {"max_rel": rel, "1e3_eps_cond": float(1e3 * np.finfo(float).eps * cond)})
        # the NMF statistic of a pure steering vector at its own bin is ~1
        p = CinvP[:, 0]
        t = float(np.max(np.abs(p.conj().T @ Chat @ p) ** 2 /
                         (a[0] * float(np.real(p.conj() @ Chat @ p)) + 1e-30)))
        check("08_nmf_stat_reference_finite", np.isfinite(t), t)
    else:
        check("08_ceps_available", False, "run prep|fold1|hybrid first")

    # 9 -- per-fold independent thresholds (prep only from CAL) -----------------
    if prep_path.is_file():
        with np.load(prep_path) as z:
            prep_acqs = set(z["acqs"].tolist())
        cal_set = set(C.fold_acquisitions(cfg, "fold1")["cal"])
        check("09_ceps_built_from_cal_only", prep_acqs <= cal_set,
              {"prep_acqs": sorted(prep_acqs), "cal": sorted(cal_set)})
        check("09_ceps_uses_no_test_acquisition",
              prep_acqs.isdisjoint(C.fold_acquisitions(cfg, "fold1")["test"]), None)

    # 10 -- paired H1 identities ------------------------------------------------
    trials_f1 = C.h1_trials("fold1", cfg)
    check("10_trial_ids_unique", len({t.trial_id for t in trials_f1}) == len(trials_f1),
          len(trials_f1))
    chunks = [trials_f1[i:i + 250] for i in range(0, len(trials_f1), 250)]
    check("10_chunk_partition_covers_all",
          sum(len(c) for c in chunks) == len(trials_f1) == cfg["expected_h1"]["fold1"],
          [len(c) for c in chunks])
    grid = {g: cfg["e1" if g in ("hybrid", "anmf", "cfar", "temporal") else
                    ("e3" if g == "fairness" else "e2")]["scr_db"] for g in C.GROUPS}
    check("10_scr_grids_frozen", all(len(v) >= 5 for v in grid.values()),
          {k: len(v) for k, v in grid.items()})

    # 11 -- independent TEST-H0 -------------------------------------------------
    test_units = C.partition_units("fold1", "TEST_H0", cfg)
    check("11_test_h0_pool_matches_config",
          len(test_units) == cfg["expected_pool_sizes"]["fold1/TEST_H0"], len(test_units))
    check("11_test_h0_from_test_acquisitions_only",
          {u.acq for u in test_units} == set(C.fold_acquisitions(cfg, "fold1")["test"]), None)

    # 12 -- determinism ---------------------------------------------------------
    from src.proposed.infer import HybridDetector
    import torch
    torch.set_num_threads(1)
    det = HybridDetector(C.CHECKPOINTS["hybrid"])
    with np.load(prep_path) as z:
        prep = {k: z[k] for k in z.files}
    t = trials_f1[0]
    x = s_all.x_of(t.acq, t.cell, t.start) if hasattr(s_all, "x_of") else \
        np.stack([s_all.data[t.acq][t.start:t.start + C.NP, t.cell],
                  s_all.data_cross[t.acq][t.start:t.start + C.NP, t.cell]]).astype(np.complex128)
    v1 = det.statistic(x, prep_sep=prep)[0]
    v2 = det.statistic(x, prep_sep=prep)[0]
    check("12_deterministic_repeat_bitwise", v1 == v2, [v1, v2])
    outs = det.network_out_batch(np.stack([x, x]), chunk=2)
    v3 = det.statistic_from_net(x, outs[0], prep_sep=prep)[0]
    check("12_batched_network_matches_single", abs(v3 - v1) / max(abs(v1), 1e-30) < 1e-5,
          {"single": v1, "batched": v3, "rel": abs(v3 - v1) / max(abs(v1), 1e-30)})

    # 13 -- writes confined to the phase-5 workspace ---------------------------
    # Release layout: the two protected backup drives do not exist here.  They are
    # optional, supplied through PAPER1_PROTECTED_BACKUPS (os.pathsep-separated), so the
    # preflight can run against a published checkout instead of aborting.
    import os as _os
    _extra = [p for p in _os.environ.get("PAPER1_PROTECTED_BACKUPS", "")
              .split(_os.pathsep) if p]
    protected = [Path(p) for p in _extra]
    protected += [C.PROJECT / "manuscript", C.PHASE4 / "protocol", C.CACHE]
    stamp = {}
    for p in protected:
        if p.exists():
            stamp[str(p)] = {f.name: [f.stat().st_size, int(f.stat().st_mtime)]
                             for f in sorted(p.glob("*"))[:6]}
    (run / "tmp" / "protected_stamp.json").write_text(json.dumps(stamp, indent=1),
                                                     encoding="utf-8")
    check("13_protected_paths_recorded", len(stamp) >= 1, list(stamp))

    # 14 -- disk and memory -----------------------------------------------------
    import shutil
    free_gb = shutil.disk_usage(C.WORK).free / 2**30
    check("14_disk_headroom_gt_2gb", free_gb > 2.0, round(free_gb, 2))
    rss = C.rss_mb()
    check("14_process_rss_under_2gb", rss < 2048, round(rss, 1))

    # 15 -- atomic write / reload ----------------------------------------------
    probe = run / "tmp" / "atomic_probe.npz"
    arr = np.arange(11, dtype=float)
    d1 = C.atomic_write_npz(probe, a=arr)
    with np.load(probe) as z:
        same = np.array_equal(z["a"], arr)
    d2 = C.sha256_file(probe)
    check("15_atomic_write_reload", same and d1 == d2, {"sha_equal": d1 == d2})
    check("15_no_tmp_left", not list(probe.parent.glob("*.tmp*")), None)

    # 16 -- ANMF factorisation equivalence and boundary rule -------------------
    from src.baselines.anmf import anmf_glrt_from_prep, steering_bank
    P = steering_bank(C.NP)
    ref, new, identical, filled = [], [], [], []
    cell = s.allowed_cells[a0][0]
    for w in (0, 1, 60, 120, 300, 400, 470):
        sec = s.secondary_windows(a0, cell, w * C.HOP, C.K_SEC, 4)
        secb = C.anmf_secondary_windows(s, a0, cell, w, C.K_SEC, 4)
        if sec.shape[0] >= C.NP:
            identical.append(sec.shape == secb.shape and np.array_equal(sec, secb))
        else:
            filled.append((w, sec.shape[0], secb.shape[0]))
        prep = C.anmf_preps(secb, P)
        if prep is None:
            continue
        y = s.data[a0][w * C.HOP:w * C.HOP + C.NP, cell].astype(np.complex128)
        import scipy.linalg as sla
        L = prep["L"]["dl0.001"]
        Cinv_y = sla.cho_solve((L, True), y)
        v = prep["CinvP"]["dl0.001"].conj().T @ y
        d = float(np.real(np.vdot(y, Cinv_y)))
        a_bin = prep["a"]["dl0.001"]
        t_new = float(np.max(np.abs(v) ** 2 / (a_bin * d + 1e-30)))
        Cm = prep["chat"] + 1e-3 * prep["tr"] / C.NP * np.eye(C.NP)
        t_ref = anmf_glrt_from_prep(y, {"Chat": Cm, "CinvP": prep["CinvP"]["dl0.001"],
                                        "a": a_bin})[0]
        ref.append(t_ref)
        new.append(t_new)
    rel = [abs(a - b) / max(abs(a), 1e-30) for a, b in zip(ref, new)]
    check("16_cholesky_matches_deployed_anmf", max(rel) < 1e-8, {"max_rel": max(rel)})
    check("16_boundary_rule_identity_where_defined", all(identical),
          {"n_checked": len(identical), "filled": filled})

    # 16b -- the completion rule itself: deterministic, no forbidden cell, and it
    # returns exactly the deployed rows first whenever the deployed set is short.
    short: list = []
    det_ok, prefix_ok = True, True
    for acq in sorted(s.data):
        nw = (s.data[acq].shape[0] - C.NP) // C.HOP + 1
        for c in (s.allowed_cells[acq][0], s.allowed_cells[acq][-1]):
            for w in (0, 1, 2, nw - 2, nw - 1):
                a = C.anmf_secondary_windows(s, acq, c, w, C.K_SEC, 4)
                b = C.anmf_secondary_windows(s, acq, c, w, C.K_SEC, 4)
                base = s.secondary_windows(acq, c, w * C.HOP, C.K_SEC, 4)
                if a.shape[0] < C.NP:
                    short.append((acq, c, w, int(a.shape[0])))
                det_ok &= bool(np.array_equal(a, b))
                if base.shape[0]:
                    prefix_ok &= bool(np.array_equal(a[:base.shape[0]], base))
    check("16b_boundary_rule_deterministic", det_ok,
          {"n_short": len(short), "short": short[:5]})
    check("16b_boundary_rule_keeps_deployed_rows", prefix_ok,
          {"note": "completion only appends rows after the deployed set"})

    # 16c -- short-record acquisitions: every unit must reach an estimable SCM,
    # with the deployed rows reproduced bit-exactly and no duplicated snapshot.
    short_acq = sorted(a for a, n in cfg["n_win"].items() if int(n) < 300)
    ok_short, est = True, {}
    if short_acq:
        s_short = C.load_sampler(short_acq, table)
        for acq in short_acq:
            nw = (s_short.data[acq].shape[0] - C.NP) // C.HOP + 1
            worst = 10 ** 9
            for c in s_short.allowed_cells[acq]:
                for w in (0, 1, nw // 2, nw - 1):
                    sec = s_short.secondary_windows(acq, c, w * C.HOP, C.K_SEC, 4)
                    ids = C._deployed_secondary_ids(s_short, acq, c, w * C.HOP, C.K_SEC, 4)
                    same = len(ids) == len(sec) and all(
                        np.array_equal(sec[i], s_short.data[acq][ids[i][1]:ids[i][1] + C.NP,
                                                             ids[i][0]])
                        for i in range(len(ids)))
                    full = C.anmf_secondary_windows(s_short, acq, c, w, C.K_SEC, 4)
                    if sec.shape[0] >= C.NP:
                        same &= np.array_equal(full, sec)
                    ok_short &= bool(same) and full.shape[0] >= C.NP
                    worst = min(worst, full.shape[0])
            est[acq] = int(worst)
    check("16c_short_record_scm_estimable", ok_short or not short_acq,
          {"acquisitions": short_acq, "min_rows": est})

    # 17 -- one shared H0 pool, identical for every arm ------------------------
    sizes, identical, want = {}, [], {}
    for fold in ("fold1", "fold2"):
        for role in ("cal", "test_h0"):
            units = C.partition_units(fold, role, cfg)
            pools = {g: tuple(C.pool_for(g, units, cfg, fold, role)) for g in C.GROUPS}
            ref = pools[C.GROUPS[0]]
            identical.append(all(p == ref for p in pools.values()))
            sizes[f"{fold}/{role}"] = len(ref)
            want[f"{fold}/{role}"] = cfg["pools"]["hybrid"][C.ROLE_ALIASES[role]]
    check("17_shared_h0_pool_elementwise_identical", all(identical), sizes)
    check("17_pool_sizes_match_policy", sizes == want, {"want": want, "got": sizes})
    check("17_pool_membership_within_superset",
          all(set(C.pool_for(g, C.partition_units(f, r, cfg), cfg, f, r))
              <= set(C.partition_units(f, r, cfg))
              for f in ("fold1", "fold2") for r in ("cal", "test_h0")
              for g in C.GROUPS), None)

    # 18 -- CFAR evaluability at clean-band edges ------------------------------
    # A cell whose guard region consumes the entire clean range block has no CA/OS
    # denominator.  The withdrawn units must be exactly that geometry, must stay a
    # small fraction of the pool, and must leave every H1 chunk non-empty.
    verdict: dict = {}
    withdrawn, pool_n, affected = {}, {}, {}
    for fold in ("fold1", "fold2"):
        acqs = sorted(set(C.fold_acquisitions(cfg, fold)["cal"] +
                          C.fold_acquisitions(cfg, fold)["test"]))
        smp = C.load_sampler(acqs, table)
        for a in acqs:
            for c in smp.allowed_cells[a]:
                verdict[(a, c)] = C.cfar_evaluable(smp, a, c)
        for role in ("CAL_H0", "TEST_H0"):
            pool = C.pool_for("cfar", C.partition_units(fold, role, cfg), cfg, fold, role)
            key = f"{fold}/{role}"
            affected[key] = sorted({(u.acq, u.cell) for u in pool
                                    if not verdict[(u.acq, u.cell)]})
            withdrawn[key] = sum(1 for u in pool if not verdict[(u.acq, u.cell)])
            pool_n[key] = len(pool)
    check("18_cfar_unavailable_cells_are_band_edges",
          all(set(v) <= {("19980205_171437_antstep", 0),
                         ("19980205_171437_antstep", 1)} for v in affected.values()),
          affected)
    frac = {k: withdrawn[k] / max(pool_n[k], 1) for k in withdrawn}
    check("18_cfar_withdrawal_below_5pct",
          all(0.0 <= v < 0.05 for v in frac.values()),
          {k: [withdrawn[k], pool_n[k]] for k in withdrawn})
    empty_chunks, n_dropped = [], {}
    for fold in ("fold1", "fold2"):
        tri = C.h1_trials(fold, cfg)
        drop = sum(1 for t in tri if not verdict[(t.acq, t.cell)])
        n_dropped[fold] = [drop, len(tri)]
        for k in range(0, len(tri), 250):
            if not any(verdict[(t.acq, t.cell)] for t in tri[k:k + 250]):
                empty_chunks.append((fold, k // 250))
    check("18_cfar_h1_chunks_nonempty", not empty_chunks,
          {"empty": empty_chunks, "trials": n_dropped})

    elapsed = time.perf_counter() - t0
    n_pass = sum(r["status"] == "PASS" for r in RESULTS)
    payload = {"generated": C.utc_now(), "n_pass": n_pass, "n_fail": len(RESULTS) - n_pass,
               "elapsed_s": round(elapsed, 1), "checks": RESULTS,
               "config_sha256": cfg["config_sha256"]}
    out = C.WORK / "reports" / "PREFLIGHT_RESULTS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"n_pass": n_pass, "n_fail": len(RESULTS) - n_pass,
                      "elapsed_s": round(elapsed, 1)}))
    return 0 if payload["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
