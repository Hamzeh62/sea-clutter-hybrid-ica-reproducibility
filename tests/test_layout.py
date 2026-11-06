"""Repository layout, frozen configuration and partition protocol.

These tests need no IPIX data: they check that the released configuration and manifests
still enumerate exactly the populations the paper reports.
"""
from __future__ import annotations

import json

import pytest


def test_layout_is_resolvable(root):
    import paths as P
    d = P.describe()
    assert d["configs_present"] and d["protocol_present"]
    assert d["released_results_present"]
    assert d["repo_root"] == str(root)


def test_campaign_config_hash_verifies():
    """``phase5_common.config()`` refuses to run if the frozen config was edited."""
    import phase5_common as C
    cfg = C.config()
    assert cfg["campaign"] == "minimal_v1_20261029"
    assert cfg["design_pfa"] == pytest.approx(1e-3)
    assert cfg["amendments"][0]["id"] == "A1"


def test_config_pins_the_two_released_checkpoints():
    import hashlib
    import paths as P
    cfg_json = json.loads((P.CONFIGS / "campaign.json").read_text(encoding="utf-8"))
    ck = cfg_json["checkpoints"]
    assert set(ck) == {"hybrid", "nophys"}
    assert ck["hybrid"]["sha256"].startswith("a0a49db9")
    assert ck["nophys"]["sha256"].startswith("2bcf1041")
    assert len(ck["hybrid"]["sha256"]) == 64
    # the configured names are the ones the code loads
    import phase5_common as C
    assert C.CHECKPOINTS["hybrid"].name == "best_s0.pt"
    assert C.CHECKPOINTS["nophys"].name == "best_s0_nophys.pt"
    assert hashlib.sha256(b"x").hexdigest()  # sanity: hashlib import is live


@pytest.mark.parametrize("fold", ["fold1", "fold2"])
def test_h0_population_matches_the_frozen_count(fold):
    import phase5_common as C
    for role in ("cal", "test_h0"):
        units = C.partition_units(fold, role)
        assert len(units) == 27156
        assert len({(u.acq, u.cell, u.window) for u in units}) == len(units)


@pytest.mark.parametrize("fold,expected", [("fold1", 1990), ("fold2", 2000)])
def test_h1_trial_count(fold, expected):
    import phase5_common as C
    trials = C.h1_trials(fold)
    assert len(trials) == expected
    assert len({t.trial_id for t in trials}) == len(trials)


@pytest.mark.parametrize("fold", ["fold1", "fold2"])
def test_cal_and_test_acquisitions_are_disjoint(fold):
    import phase5_common as C
    acq = C.fold_acquisitions(C.config(), fold)
    assert len(acq["cal"]) == 8 and len(acq["test"]) == 8
    assert set(acq["cal"]) & set(acq["test"]) == set()


def test_every_acquisition_is_tested_exactly_once():
    import phase5_common as C
    cfg = C.config()
    tested = []
    for fold in ("fold1", "fold2"):
        tested += C.fold_acquisitions(cfg, fold)["test"]
    assert len(tested) == 16 and len(set(tested)) == 16


def test_h0_units_never_land_on_an_h1_window():
    """The H1 block layout reserves every fourth block; those windows must stay out of H0."""
    import phase5_common as C
    import split_protocol as sp
    cfg = C.config()
    for fold in ("fold1", "fold2"):
        h1 = {t.window for t in C.h1_trials(fold)}
        h0 = {u.window for u in C.partition_units(fold, "test_h0")}
        assert h1 & h0 == set()
        lay = sp.block_layout(int(cfg["n_win"]["19980205_171437_antstep"]))
        assert set(lay.h1_windows) & set(lay.h0_windows) == set()


def test_pool_sizes_are_identical_for_every_arm():
    """Amendment A1: every arm draws the same H0 units, element by element."""
    import phase5_common as C
    cfg = C.config()
    fold = "fold1"
    units = C.partition_units(fold, "test_h0", cfg)
    pools = {g: C.pool_for(g, units, cfg, fold, "test_h0") for g in C.GROUPS}
    ref = [(u.acq, u.cell, u.window) for u in pools["hybrid"]]
    for g, pool in pools.items():
        assert [(u.acq, u.cell, u.window) for u in pool] == ref, g
        assert len(pool) == 4000


def test_pool_subsample_is_deterministic():
    import phase5_common as C
    cfg = C.config()
    units = C.partition_units("fold2", "cal", cfg)
    a = C.pool_for("hybrid", units, cfg, "fold2", "cal")
    b = C.pool_for("hybrid", units, cfg, "fold2", "cal")
    assert [u.window for u in a] == [u.window for u in b]
    assert len(a) == 8000
