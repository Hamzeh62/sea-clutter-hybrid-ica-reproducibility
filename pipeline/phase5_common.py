"""Shared campaign machinery for the authorized Phase 5 minimal campaign.

Everything here is deterministic given the frozen manifests, the frozen seeds and the
campaign config.  Nothing in this module writes outside ``phase5_minimal/``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Release portability: the workstation layout (one drive letter per directory) is
# replaced by the repository layout, resolved in ``pipeline/paths.py``.  No numerical
# code below this header is modified.
sys.path.insert(0, str(Path(__file__).resolve().parent))       # pipeline/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root

import paths as P                                             # noqa: E402

WORK = P.OUTPUTS                        # reports, figures, statistics
PROJECT = P.REPO_ROOT                   # ``src`` package lives here
MDL = P.REPO_ROOT
PHASE4 = P.REPO_ROOT                    # frozen manifests are read as PHASE4/"protocol/..."
CACHE = P.CACHE

sys.path.insert(0, str(P.PROTOCOL))     # acquisition_ids, units_fixed, split_protocol
sys.path.insert(0, str(MDL))

NP, HOP, PRF = 512, 256, 1000.0
K_SEC = 1024
DESIGN_PFA = 1.0e-3
CHECKPOINTS = {
    "hybrid": P.CHECKPOINTS / "best_s0.pt",
    "nophys": P.CHECKPOINTS / "best_s0_nophys.pt",
}
CF = dict(train_r=8, guard_r=2, train_f=8, guard_f=2)

GROUPS = ("hybrid", "anmf", "cfar", "temporal", "fairness", "nophys", "cold", "ica")
ARMS = {
    "hybrid": ("hybrid",),
    "anmf": ("anmf_dl", "anmf_unloaded", "glrt"),
    "cfar": ("ca", "os"),
    "temporal": ("ar8", "notch"),
    "fairness": ("ar8_vvfit", "notch_clean"),
    "nophys": ("nophys",),
    "cold": ("cold",),
    "ica": ("ica_only",),
}
ALL_ARMS = tuple(a for g in GROUPS for a in ARMS[g])
PREP_VARIANTS = ("hybrid", "nophys", "cold", "ica", "ar8", "ar8vv", "notch")


# --------------------------------------------------------------------------- io
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomic replace with Windows sharing-violation retries.

    On Windows ``os.replace`` fails with a sharing violation while another process
    (an operator reading the state file, an antivirus scanner) holds the destination
    open.  That is transient, so the replace is retried instead of aborting a job or
    a supervisor loop.
    """
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))


def atomic_write_npz(path: Path, **arrays) -> str:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    digest = sha256_file(tmp)
    os.replace(tmp, path)
    return digest


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------- campaign i/o
def config() -> dict:
    cfg = read_json(PROJECT / "configs/campaign.json")
    declared = cfg["config_sha256"]
    body = dict(cfg)
    body.pop("config_sha256")
    actual = sha256_bytes(json.dumps(body, indent=2, sort_keys=True).encode("utf-8"))
    if declared != actual:
        raise RuntimeError(f"campaign config hash mismatch: declared {declared[:12]} "
                           f"actual {actual[:12]}; refusing to run")
    return cfg


def run_dir(cfg: dict) -> Path:
    d = WORK / "runs" / cfg["campaign"]
    for sub in ("scores", "scores/h0", "scores/h1", "preps", "logs", "tmp"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ manifests
@dataclass(frozen=True)
class Unit:
    acq: str
    cell: int
    window: int          # window index (start_sample = window * HOP)

    @property
    def start(self) -> int:
        return self.window * HOP


@dataclass(frozen=True)
class Trial:
    trial_id: str
    acq: str
    cell: int
    window: int
    block: int
    scr_db: float

    @property
    def start(self) -> int:
        return self.window * HOP


def _read_csv(path: Path) -> list[dict]:
    import csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


ROLE_ALIASES = {"cal": "CAL_H0", "test_h0": "TEST_H0",
                "CAL_H0": "CAL_H0", "TEST_H0": "TEST_H0"}


def h0_windows(acq: str, cfg: dict) -> set[int]:
    """H0 window indices of one acquisition under the frozen block layout.

    The Phase 4 manifest enumerates every window of every block, which includes the
    blocks the protocol reserves for H1 trials (``i % 4 == 3``).  Enumerating those
    as H0 would put the injected-target windows into the null pool.  The frozen
    ``split_protocol.block_layout`` is the authority the Phase 4 audit validated
    (27,156 H0 windows per fold); the manifest is cross-checked against it.
    """
    import split_protocol as sp
    return set(sp.block_layout(int(cfg["n_win"][acq])).h0_windows)


def partition_units(fold: str, role: str, cfg: dict | None = None) -> list[Unit]:
    """Enumerate the frozen H0 windows of one fold partition."""
    rows = _read_csv(PHASE4 / "protocol/DATA_PARTITION_MANIFEST.csv")
    role = ROLE_ALIASES.get(role, role)
    if cfg is None:
        cfg = config()
    keep = {a: h0_windows(a, cfg) for a in cfg["n_win"]}
    out: list[Unit] = []
    for r in rows:
        if r["fold"] != fold or r["role"] != role:
            continue
        lo, hi = int(r["start_window"]), int(r["stop_window"])
        n = int(r["n_windows"])
        if hi - lo + 1 != n:
            raise ValueError(f"manifest row inconsistent: {r}")
        acq = r["acq_id"]
        out.extend(Unit(acq, int(r["cell"]), w) for w in range(lo, hi + 1)
                   if w in keep[acq])
    if not out:
        raise RuntimeError(f"no units for {fold}/{role}")
    expected = (cfg or {}).get("expected_pool_sizes", {}).get(f"{fold}/{role}")
    if expected is not None and len(out) != expected:
        raise RuntimeError(f"{fold}/{role}: {len(out)} units but campaign config "
                           f"declares {expected}")
    return out


def h1_trials(fold: str, cfg: dict | None = None) -> list[Trial]:
    rows = _read_csv(PHASE4 / "protocol/H1_TRIAL_DESIGN.csv")
    out = [Trial(r["trial_id"], r["acq_id"], int(r["cell"]), int(r["start_window"]),
                 int(r["block"]), float(r["scr_db"]))
           for r in rows if r["fold"] == fold]
    if not out:
        raise RuntimeError(f"no H1 trials for {fold}")
    expected = (cfg or {}).get("expected_h1", {}).get(fold)
    if expected is not None and len(out) != expected:
        raise RuntimeError(f"{fold}: {len(out)} H1 trials but config declares {expected}")
    return out


def fold_acquisitions(cfg: dict, fold: str) -> dict[str, list[str]]:
    summary = read_json(PHASE4 / "protocol/PARTITION_SUMMARY.json")
    s = summary[fold]
    return {"cal": s["cal_acquisitions"], "test": s["test_acquisitions"]}


# --------------------------------------------------------------------- sampler
def canonical_table(cfg: dict):
    from acquisition_ids import build_table
    return build_table(PROJECT, CACHE, {})


def load_sampler(acqs: list[str], table) -> object:
    from units_fixed import FixedUnitSampler
    files = [str(CACHE / table[a].cache_name) for a in acqs]
    return FixedUnitSampler(files, NP, HOP, table, background_policy="clean")


# ------------------------------------------------------------- acquisition band
def clear_band(sampler, acq: str, n_cells: int = 8) -> tuple[float, float]:
    """Clutter-free Doppler band from the median PSD of eligible cells only.

    Eligible cells come from the exclusion table, so the band cannot be shaped by
    target power.  Deterministic: the n_cells lowest eligible cell indices are used.
    """
    from src.data.synth import clear_doppler_band, estimate_psd
    cells = list(sampler.allowed_cells[acq])[:n_cells]
    if not cells:
        raise RuntimeError(f"{acq}: no eligible cell for band estimation")
    psds = []
    for c in cells:
        f, p = estimate_psd(sampler.data[acq][:, c].astype(np.complex128), PRF, 512)
        psds.append(p)
    med = np.median(np.stack(psds), axis=0)
    return clear_doppler_band(med, f, guard_bins=8)


def trial_rng(trial_id: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(trial_id.encode()).digest()[:8], "little")
    return np.random.default_rng(seed % (2**63))


def make_target(trial: Trial, sampler, band: tuple[float, float], scr_db: float):
    """Deterministic Swerling-0 target for a trial at a requested empirical SCR."""
    from src.data.synth import swerling_target
    rng = trial_rng(trial.trial_id)
    phi0 = rng.uniform(0, 2 * np.pi)
    u = rng.uniform()
    fd = band[0] + u * (band[1] - band[0])
    vv = sampler.data[trial.acq][trial.start:trial.start + NP, trial.cell].astype(np.complex128)
    p_clutter = float(np.mean(np.abs(vv) ** 2))
    amp = np.sqrt(p_clutter * 10.0 ** (scr_db / 10.0))
    # phase is fixed by trial, not by SCR: reproduce the same tone at every SCR
    n = np.arange(NP)
    s = amp * np.exp(1j * (2 * np.pi * fd * n / PRF + phi0))
    return s, fd, phi0, p_clutter


# ------------------------------------------------------------------ detectors
def ar_fit(x: np.ndarray, p: int = 8) -> np.ndarray:
    n = len(x)
    r = np.array([np.vdot(x[: n - k], x[k:]) / (n - k) for k in range(p + 1)])
    R = np.empty((p, p), complex)
    for i in range(p):
        for j in range(p):
            R[i, j] = r[abs(i - j)]
    R += 1e-6 * r[0] * np.eye(p)
    return np.linalg.solve(R, r[1:])


def ar_residual(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    p = len(a)
    e = x[p:].copy()
    for k in range(p):
        e -= a[k] * x[p - 1 - k: len(x) - 1 - k]
    return e


def notch_vector(vv: np.ndarray) -> np.ndarray:
    rho = np.vdot(vv[:-1], vv[1:]) / max(np.sum(np.abs(vv) ** 2), 1e-30)
    w = np.array([1.0, -rho], complex)
    return w / np.linalg.norm(w)


def _deployed_secondary_ids(s, acq: str, cell: int, start: int, K: int,
                            cells_each_side: int) -> list[tuple[int, int]]:
    """Identities ``(cell, start_sample)`` in the order the deployed sampler uses.

    Mirrors ``FixedUnitSampler.secondary_windows`` exactly so that the completion rule
    can skip rows the deployed set already contains instead of duplicating snapshots
    (duplicates would silently deflate the SCM).
    """
    cube = s.data[acq]
    ncell = cube.shape[1]
    lo, hi = max(0, cell - cells_each_side), min(ncell, cell + cells_each_side + 1)
    cand = [c for c in s._background_cells(acq, lo, hi) if c != cell]
    if not cand:
        return []
    per_cell = int(np.ceil(K / len(cand)))
    offs = np.arange(per_cell) - per_cell // 2
    out: list[tuple[int, int]] = []
    for c in cand:
        for o in offs:
            st = start + int(o) * s.hop
            if st < 0 or st + s.Np > cube.shape[0]:
                continue
            out.append((int(c), int(st)))
            if len(out) >= K:
                return out
    return out


def anmf_secondary_windows(s, acq: str, cell: int, w0: int, K: int = K_SEC,
                           cells_each_side: int = 4) -> np.ndarray:
    """Secondary windows for ANMF: deployed set, then a deterministic completion.

    ``w0`` is a *window index* (multiply by ``HOP`` for the sample offset the sampler
    expects); the sampler's own convention is samples, and passing an index there
    silently estimates every covariance near the start of the cube.

    A unit whose deployed secondary set already has ``Np`` or more snapshots is
    returned untouched, so interior units are bit-identical to the deployed path.
    Otherwise the set is completed with unused windows of the same allowed cells,
    nearest window first; if the immediate neighbourhood cannot supply ``Np``
    snapshots (short ``antstep`` records, 233 windows, one allowed neighbour), the
    cell window is widened in steps of two until the SCM is estimable.  The rule is
    uniform, deterministic, never uses a target or guard cell, and never duplicates a
    snapshot.
    """
    start = w0 * HOP
    sec = s.secondary_windows(acq, cell, start, K, cells_each_side)
    if sec.shape[0] >= NP:
        return sec                       # deployed set is defined: leave it untouched
    cube = s.data[acq]
    n_win = (cube.shape[0] - NP) // HOP + 1
    used = set(_deployed_secondary_ids(s, acq, cell, start, K, cells_each_side))
    rows = list(sec)
    side = cells_each_side
    while len(rows) < NP and side <= cube.shape[1]:
        lo, hi = max(0, cell - side), min(cube.shape[1], cell + side + 1)
        cand = sorted((c for c in s._background_cells(acq, lo, hi) if c != cell),
                      key=lambda c: (abs(c - cell), c))
        order = sorted(range(n_win), key=lambda w: (abs(w - w0), 0 if w < w0 else 1, w))
        for w in order:
            for c in cand:
                if len(rows) >= K:
                    return np.stack(rows)
                key = (c, w * HOP)
                if key in used:
                    continue
                rows.append(cube[w * HOP:w * HOP + NP, c].astype(np.complex128))
                used.add(key)
        side += 2
    return np.stack(rows) if rows else np.empty((0, NP), np.complex128)


def anmf_preps(secondary: np.ndarray, P_BANK: np.ndarray) -> dict | None:
    """SCM + one Cholesky factorisation, then both loadings and the two statistics.

    The deployment path in ``src/baselines/anmf.py`` factorises the same matrix twice
    with two LU solves and re-solves it once per statistics call.  Replacing those
    solves by one Cholesky factorisation changes the statistics by < 1e-11 relative
    (measured, see PREFLIGHT report) and costs about a third of the time.
    """
    if secondary.shape[0] < NP:
        return None
    import scipy.linalg as sla
    Yn = secondary / np.sqrt(np.sum(np.abs(secondary) ** 2, axis=1, keepdims=True) + 1e-30)
    chat = (Yn.conj().T @ Yn) / Yn.shape[0]
    tr = float(np.trace(chat).real)
    out = {"chat": chat, "tr": tr, "a": {}, "CinvP": {}, "L": {}}
    for name, jitter in (("unloaded", 1e-10), ("dl0.001", 1e-3)):
        Cm = chat + jitter * tr / NP * np.eye(NP)
        L = sla.cholesky(Cm, lower=True, check_finite=False)
        # The bank solve reuses the factor: a second factorisation (np.linalg.solve
        # refactorises with LU) would double the per-window cost for no numerical gain.
        CinvP = sla.cho_solve((L, True), P_BANK, check_finite=False)
        out["a"][name] = np.real(np.einsum("ij,ij->j", P_BANK.conj(), CinvP))
        out["CinvP"][name] = CinvP
        out["L"][name] = L
    return out


def anmf_scores(y: np.ndarray, prep: dict) -> tuple[float, float, float]:
    import scipy.linalg as sla
    scores = {}
    for name in ("dl0.001", "unloaded"):
        L = prep["L"][name]
        Cinv_y = sla.cho_solve((L, True), y)
        v = prep["CinvP"][name].conj().T @ y
        d = float(np.real(np.vdot(y, Cinv_y)))
        v2 = np.abs(v) ** 2
        a = prep["a"][name]
        scores[name] = (float(np.max(v2 / (a * d + 1e-30))),
                        float(np.max(v2 / (a + 1e-30))))
    dl, glrt = scores["dl0.001"]
    return dl, scores["unloaded"][0], glrt


# --------------------------------------------------------------- thresholding
def cluster_labels(units: list[Unit]) -> list[tuple[str, int]]:
    return [(u.acq, u.cell) for u in units]


def stratified_subsample(units: list[Unit], n: int, seed: int) -> list[Unit]:
    """Deterministic acquisition-proportional subsample without replacement."""
    if n >= len(units):
        return list(units)
    by_acq: dict[str, list[int]] = {}
    for i, u in enumerate(units):
        by_acq.setdefault(u.acq, []).append(i)
    rng = np.random.default_rng(seed)
    acqs = sorted(by_acq)
    alloc = {a: 0 for a in acqs}
    frac = n / len(units)
    for a in acqs:
        alloc[a] = int(np.floor(frac * len(by_acq[a])))
    short = n - sum(alloc.values())
    order = sorted(acqs, key=lambda a: (-(frac * len(by_acq[a]) - alloc[a]), a))
    for a in order[:short]:
        alloc[a] += 1
    picked: list[Unit] = []
    for a in acqs:
        idx = np.asarray(by_acq[a])
        take = rng.choice(idx, size=min(alloc[a], len(idx)), replace=False)
        picked.extend(units[i] for i in sorted(take))
    return picked


def pool_for(group: str, units: list[Unit], cfg: dict, fold: str,
             role: str = "CAL_H0") -> list[Unit]:
    """The frozen H0 pool of one (group, fold, partition).

    Every group draws from the identical superset with the identical seed, so arms share
    units element-by-element: the pairing the primary contrast relies on.  A per-role cap
    therefore cannot make one arm faster than another in the comparison itself.
    """
    rule = cfg["pools"][group]
    if isinstance(rule, dict):
        rule = rule[ROLE_ALIASES.get(role, role)]
    if rule == "full":
        return list(units)
    seed = cfg["seeds"]["pool"] + int(fold[-1])
    return stratified_subsample(units, int(rule), seed)


# ------------------------------------------------------------- CFAR evaluability
def cfar_reference_columns(sampler, acq: str, cell: int) -> int:
    """Non-guard range columns inside the 2-D CFAR training window of one cell.

    The deployed CFAR removes the cell under test plus ``guard_r`` neighbours on each
    side before estimating its denominator.  Where the clean-policy block is that
    narrow the training window is empty, the CA/OS denominator is NaN, and the
    statistic is undefined rather than merely noisy.  That happens at the edge of a
    clean band that is separated from the cube boundary by a wide forbidden stretch
    (one record has a two-cell clean stub), so evaluability is decided from the
    *geometry before scoring* instead of from a NaN afterwards.
    """
    cells, ci = sampler.range_block(acq, cell, 0, CF["guard_r"] + CF["train_r"],
                                    inject_cell=cell)
    n = len(cells)
    g_lo, g_hi = max(0, ci - CF["guard_r"]), min(n, ci + CF["guard_r"] + 1)
    return int(n - (g_hi - g_lo))


def cfar_evaluable(sampler, acq: str, cell: int) -> bool:
    """True iff the CA/OS denominator exists at this cell."""
    return cfar_reference_columns(sampler, acq, cell) > 0


# --------------------------------------------------------------------- metrics
def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(_PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / 2**20


def peak_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    except Exception:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(_PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def unit_manifest(units: list[Unit]) -> dict:
    return {
        "n": len(units),
        "acqs": sorted({u.acq for u in units}),
        "cells": sorted({(u.acq, u.cell) for u in units}),
        "first": asdict(units[0]) if units else None,
        "last": asdict(units[-1]) if units else None,
    }
