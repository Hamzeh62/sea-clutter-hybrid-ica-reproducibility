"""The protocol defects that Phase 4 found and fixed, tested as regressions.

1. ``src/eval/units.py`` used ``Path(f).stem`` for cache keys and ``.cdf`` stripping for the
   metadata keys, so every target-cell lookup returned ``None`` and the exclusion set was
   silently empty (38 % of the intended-forbidden test cells were sampled).
2. A missing or empty exclusion set was a silent default rather than an error.
3. The ANMF secondary-window sampler was once called with a window index where it expects
   a sample offset, which estimated every covariance near the start of each cube.

Each test below fails if the corresponding defect returns.
"""
from __future__ import annotations

import numpy as np
import pytest


# ------------------------------------------------------------------ acquisition ids
def test_canonical_id_is_stable_across_transport_suffixes():
    from acquisition_ids import canonical_acq_id
    base = "19931107_141630_starea"
    for name in (f"{base}.cdf", f"{base}_cubes.npz", f"{base}.CDF",
                 f"{base}_cubes", f"{base}.npz"):
        assert canonical_acq_id(name) == base
    assert canonical_acq_id("19931118_162155_stareC0000_cubes.npz") == \
        "19931118_162155_starec0000"


def test_canonical_id_rejects_anything_it_cannot_parse():
    from acquisition_ids import canonical_acq_id
    for bad in ("cubes.npz", "19931107_141630_starea_extra", "19931107_141630_", ""):
        with pytest.raises(ValueError):
            canonical_acq_id(bad)


def test_the_cache_name_and_the_metadata_key_agree():
    """This is the exact mismatch that emptied the exclusion set in Phase 3."""
    from pathlib import Path
    from acquisition_ids import canonical_acq_id
    cache = Path("19931107_141630_starea_cubes.npz")
    meta = "19931107_141630_starea.cdf"
    assert canonical_acq_id(cache) == canonical_acq_id(meta)


def test_target_table_covers_every_cache_file(cache_dir):
    import json
    import paths as P
    from acquisition_ids import build_table
    table = build_table(P.REPO_ROOT, cache_dir, {})
    meta = json.loads((P.METADATA / "target_cells.json").read_text(encoding="utf-8"))
    assert len(table) == len(meta) == 16
    for aid, rec in table.items():
        assert rec.forbidden, f"{aid}: empty exclusion set"
        assert rec.targets and rec.n_eligible > 0
        # guard rule: every target cell plus its two neighbours, clipped
        expect = {c for t in rec.targets for c in (t - 1, t, t + 1) if 0 <= c < rec.n_range}
        assert set(rec.forbidden) == expect


def test_literature_cell_is_one_based_and_inside_the_range(cache_dir):
    import paths as P
    from acquisition_ids import build_table
    table = build_table(P.REPO_ROOT, cache_dir, {})
    found = 0
    for aid, rec in table.items():
        if rec.literature_1based is None:
            continue
        found += 1
        assert 1 <= rec.literature_1based <= rec.n_range
        assert rec.literature_0based == rec.literature_1based - 1
        assert rec.literature_0based in rec.targets
    assert found == 8, "the eight D43-annotated acquisitions must all be resolved"


def test_cell_selection_assertion_rejects_a_target_cell(cache_dir):
    import paths as P
    from acquisition_ids import assert_cell_selection, build_table
    table = build_table(P.REPO_ROOT, cache_dir, {})
    aid, rec = next((a, r) for a, r in table.items() if r.forbidden)
    assert_cell_selection(aid, rec.eligible, table)          # legal selection passes
    with pytest.raises(AssertionError):
        assert_cell_selection(aid, [rec.forbidden[0]], table)
    with pytest.raises(AssertionError):
        assert_cell_selection(aid, [rec.n_range + 5], table)


# ------------------------------------------------------------------ block layout
def test_block_layout_reserves_h1_blocks_and_guards():
    from split_protocol import BLOCK, GUARD, block_layout
    lay = block_layout(1000)
    h0, h1 = set(lay.h0_windows), set(lay.h1_windows)
    assert h0 & h1 == set()
    assert len(h0) + len(h1) <= 1000
    # blocks are separated: no window of one block starts inside another
    starts = sorted(h0)
    for a, b in zip(starts, starts[1:]):
        if b - a == 1:
            continue
        assert b - a >= GUARD, (a, b)
    assert lay.n_h0 == len(h0)


def test_every_fourth_block_is_h1():
    from split_protocol import H1_BLOCK_MOD, block_layout
    lay = block_layout(1000)
    for n, (lo, hi) in enumerate(lay.blocks):
        # blocks are stored as half-open [lo, hi)
        contiguous = set(range(lo, hi))
        if n % H1_BLOCK_MOD == 3:
            assert contiguous <= set(lay.h1_windows)
        else:
            assert contiguous <= set(lay.h0_windows)


# ------------------------------------------------------------------ thresholds
def test_threshold_uses_the_type7_quantile():
    from thresholds_frozen import estimate
    s = np.arange(1000, dtype=float)
    th = estimate(s, 0.01)                     # n*p = 10 -> empirical branch
    assert th.method == "empirical"
    assert th.value == pytest.approx(np.quantile(s, 0.99))


def test_threshold_falls_back_to_the_labelled_tail_when_events_are_rare():
    from thresholds_frozen import MIN_EVENTS_FOR_EMPIRICAL, estimate
    rng = np.random.default_rng(0)
    s = rng.exponential(size=500)
    th = estimate(s, 1e-3)                     # 500 * 1e-3 = 0.5 events
    assert 500 * 1e-3 < MIN_EVENTS_FOR_EMPIRICAL
    assert th.method == "extrapolated"
    assert th.value > np.quantile(s, 0.995)


def test_threshold_is_monotone_in_the_design_rate():
    from thresholds_frozen import estimate
    s = np.random.default_rng(1).exponential(size=20000)
    th = [estimate(s, p).value for p in (1e-2, 1e-3, 1e-4)]
    assert th[0] < th[1] < th[2]


def test_realised_pfa_uses_strict_exceedance():
    from thresholds_frozen import realised_pfa
    s = np.array([1.0, 2.0, 3.0, 4.0])
    assert realised_pfa(s, 3.0) == {"k": 1, "n": 4, "pfa": 0.25}
    assert realised_pfa(s, 4.0)["k"] == 0
    assert realised_pfa(s, -1.0)["pfa"] == 1.0


def test_threshold_rejects_empty_and_nonfinite_input():
    from thresholds_frozen import estimate
    with pytest.raises(ValueError):
        estimate([], 1e-3)
    with pytest.raises(ValueError):
        estimate([1.0, np.nan, 2.0], 1e-3)


# ------------------------------------------------------------------ ANMF sampler
def test_anmf_secondary_windows_treats_its_argument_as_a_window_index():
    """``anmf_secondary_windows(..., w0)`` must multiply by the hop, not pass w0 through."""
    import inspect
    import phase5_common as C
    src = inspect.getsource(C.anmf_secondary_windows)
    assert "start = w0 * HOP" in src
    assert "s.secondary_windows(acq, cell, start" in src
