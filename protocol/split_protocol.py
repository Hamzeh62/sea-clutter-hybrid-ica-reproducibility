"""Phase 4B - leakage-free acquisition-level split protocol.

Two facts change the split design relative to Phase 3:

1. The headline checkpoint ``best_s0.pt`` is the **synthetic-only S0** model
   (``src/proposed/train.py`` trains it with ``synth_batch``); the S1 fine-tune
   on real clutter exists but did not beat it and is not deployed.  The model
   has therefore never seen any real acquisition.
2. The only real-data component of the deployed pipeline is the C_sep
   covariance (``csep_prep.npz``, 250 old-TRAIN windows), which can be rebuilt
   from CAL-only windows for every fold (cheap; no network training).

So the split can use all 16 acquisitions instead of the 3 old test files.  A
grouped two-fold design is used: acquisitions are stratified by site and
partitioned 8/8; fold 1 tests A and calibrates on B, fold 2 tests B and
calibrates on A.  Every acquisition is tested exactly once, every TEST
acquisition is absent from its CAL set, and the C_sep source is the CAL set.

A residual, unavoidable dependence is disclosed separately: the S0 synthetic
prior uses the median K-distribution (nu, b) over ``clutter_stats.json``, which
was fitted on all files.  This is a two-scalar global prior; its fold
sensitivity is computed in ``phase4b_split_audit`` and a retrain with a
CAL-only prior is a full-campaign item.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLOCK = 31
GUARD = 2                 # unused window indices between adjacent blocks
STRIDE = BLOCK + GUARD    # 33 window starts = 8448 samples of separation
H1_BLOCK_MOD = 4          # block indices 3, 7, 11 are H1; the rest are H0
H1_TRIALS_PER_ACQ = 250
MIN_H1_SEPARATION = GUARD  # distinct H1 windows in one cell keep |dw| >= 2
SPLIT_VERSION = "P4-2FOLD-1.0"


@dataclass(frozen=True)
class BlockLayout:
    n_win: int
    blocks: tuple[tuple[int, int], ...]      # (start, stop) window offsets
    h0_windows: tuple[int, ...]
    h1_windows: tuple[int, ...]
    guard_windows: tuple[int, ...]

    @property
    def n_h0(self) -> int:
        return len(self.h0_windows)

    @property
    def n_h1(self) -> int:
        return len(self.h1_windows)


def block_layout(n_win: int) -> BlockLayout:
    blocks, guards = [], []
    for b in range(0, n_win, STRIDE):
        stop = min(b + BLOCK, n_win)
        if stop - b < BLOCK // 2:
            break
        blocks.append((b, stop))
        if stop < n_win:
            guards.append(stop)
    h0, h1 = [], []
    for i, (a, b) in enumerate(blocks):
        (h1 if i % H1_BLOCK_MOD == H1_BLOCK_MOD - 1 else h0).extend(range(a, b))
    return BlockLayout(n_win, tuple(blocks), tuple(h0), tuple(h1), tuple(guards))


def site_of(acq_id: str) -> str:
    return acq_id.split("_", 2)[2]


SPLIT_SEED = 20261029


def grouped_two_fold(table) -> dict[str, set[str]]:
    """Balanced 8/8 fold assignment that is stratified by site.

    The sites are very unequal (14 / 5 / 2 files), so a simple alternation
    cannot give an exact balance *and* keep every site on both sides.  We draw
    a seeded permutation and take the first half that satisfies both
    constraints; the first seed starting at ``SPLIT_SEED`` that qualifies is
    frozen and recorded.  No acquisition is ever re-drawn after outcomes are
    seen because the search only looks at acquisition IDs.
    """
    ids = sorted(table)
    sites = {aid: site_of(aid) for aid in ids}
    n_site: dict[str, int] = {}
    for aid in ids:
        n_site[sites[aid]] = n_site.get(sites[aid], 0) + 1
    # best achievable: per-site imbalance <= 1, total imbalance <= 1
    target_a = {s: n // 2 for s, n in n_site.items()}
    if sum(target_a.values()) * 2 + 1 < len(ids):
        target_a[max(n_site, key=lambda s: n_site[s] % 2)] += 1
    for offset in range(20_000):
        rng = np.random.default_rng(SPLIT_SEED + offset)
        perm = [ids[i] for i in rng.permutation(len(ids))]
        A = set(perm)
        # greedy site-quota assignment in permutation order, cap at target+1
        cnt: dict[str, int] = {s: 0 for s in n_site}
        keep: set[str] = set()
        for aid in perm:
            s = sites[aid]
            if cnt[s] < target_a[s] + 1 and len(keep) < sum(target_a.values()):
                keep.add(aid)
                cnt[s] += 1
        B = A - keep
        if len(keep) != sum(target_a.values()):
            continue
        if all(abs(cnt[s] - (n_site[s] - cnt[s])) <= 1 for s in n_site):
            return {"A": keep, "B": B, "seed_used": SPLIT_SEED + offset}
    raise AssertionError("no site-stratified split found")


def folds(table) -> dict[str, dict]:
    """fold_id -> {test: acqs, cal: acqs}; complementary folds."""
    halves = grouped_two_fold(table)
    if halves["A"] & halves["B"]:
        raise AssertionError("fold halves overlap")
    if halves["A"] | halves["B"] != set(table):
        raise AssertionError("fold halves do not cover the table")
    return {
        "fold1": {"test": set(halves["A"]), "cal": set(halves["B"]),
                  "seed": halves["seed_used"]},
        "fold2": {"test": set(halves["B"]), "cal": set(halves["A"]),
                  "seed": halves["seed_used"]},
    }


def partition(table, test_acqs: set[str], cal_acqs: set[str],
              n_win_by_acq: dict[str, int]) -> dict:
    """Cells and frozen window blocks for one fold (per-acquisition layouts)."""
    if test_acqs & cal_acqs:
        raise AssertionError("TEST and CAL acquisitions overlap")
    cal_cells = {a: list(table[a].eligible) for a in sorted(cal_acqs)}
    test_cells = {a: list(table[a].eligible) for a in sorted(test_acqs)}
    layouts = {a: block_layout(n_win_by_acq[a]) for a in set(cal_acqs) | set(test_acqs)}
    cal_windows = {a: list(layouts[a].h0_windows) for a in cal_cells}
    test_h0 = {a: list(layouts[a].h0_windows) for a in test_cells}
    test_h1_pool = {a: list(layouts[a].h1_windows) for a in test_cells}
    return {
        "layouts": layouts, "cal_cells": cal_cells, "test_cells": test_cells,
        "cal_windows": cal_windows, "test_h0_windows": test_h0,
        "test_h1_pool": test_h1_pool,
        "n_cal_windows": sum(len(cal_cells[a]) * len(cal_windows[a]) for a in cal_cells),
        "n_test_h0_windows": sum(len(test_cells[a]) * len(test_h0[a]) for a in test_cells),
        "n_test_h1_candidates": sum(len(test_cells[a]) * len(test_h1_pool[a])
                                    for a in test_cells),
    }


def h1_capacity(n_cells: int, pool_len: int, min_sep: int = MIN_H1_SEPARATION) -> int:
    """Guaranteed trials when every cell keeps starts ``min_sep`` windows apart.

    Uses ``pool_len // min_sep`` (not the ceiling) because each cell draws one
    random phase; a cell whose phase leaves an odd window at the tail yields one
    fewer trial.  The guarantee has to hold for every phase.
    """
    return n_cells * (pool_len // min_sep)


def sample_h1_windows(cells: list[int], pool: list[int], n_trials: int,
                      rng, min_sep: int = MIN_H1_SEPARATION) -> list[tuple[int, int]]:
    """Stratified systematic H1 selection with random phase and cell order.

    Each cell yields ``pool[phase::min_sep]`` (phase random per cell), the cells
    are visited in a random round-robin, and the draw stops at ``n_trials`` or
    when every cell is exhausted.  Two consequences matter:

    * no two trials in one range cell share a sample (|dw| >= 2 = 512 samples),
    * trials are spread evenly over cells instead of following a multinomial
      draw, so the cluster size is near-constant and the design effect is known
      before any run.

    ``n_trials`` above capacity is a hard error; callers must use
    :func:`h1_capacity` to set a feasible quota.
    """
    cap = h1_capacity(len(cells), len(pool), min_sep)
    if n_trials > cap:
        raise ValueError(f"asked {n_trials} H1 trials, capacity is {cap} "
                         f"({len(cells)} cells x {len(pool)} windows, sep {min_sep})")
    order = [int(c) for c in rng.permutation(np.asarray(sorted(cells)))]
    streams: dict[int, list[int]] = {}
    for c in order:
        phase = int(rng.integers(min_sep))
        streams[c] = [int(w) for w in pool[phase::min_sep]]
    chosen: list[tuple[int, int]] = []
    depth = 0
    while len(chosen) < n_trials:
        progressed = False
        for c in order:
            if depth < len(streams[c]):
                chosen.append((c, streams[c][depth]))
                progressed = True
                if len(chosen) == n_trials:
                    break
        if not progressed:
            break
        depth += 1
    if len(chosen) < n_trials:
        raise AssertionError(f"placed {len(chosen)}/{n_trials} H1 trials")
    return chosen
