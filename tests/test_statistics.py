"""Statistical procedures used for every interval and test in the paper."""
from __future__ import annotations

import numpy as np
import pytest


def test_mcnemar_matches_the_exact_binomial():
    """``mcnemar`` returns ``(p, n01, n10)``; the p-value is the exact two-sided test."""
    from scipy import stats
    import p7_analyze as A
    h = np.array([1, 1, 0, 0, 1, 0])
    r = np.array([0, 1, 1, 0, 0, 0])
    p, n01, n10 = A.mcnemar(h, r)
    # h=1,r=0 at positions 0 and 4; h=0,r=1 at position 2
    assert (n01, n10) == (2, 1)
    assert p == pytest.approx(float(stats.binomtest(2, 3, 0.5).pvalue))
    p2, a, b = A.mcnemar(h, h)
    assert (a, b) == (0, 0) and p2 == 1.0


def test_mcnemar_is_symmetric_in_its_arguments():
    import p7_analyze as A
    h = np.array([1, 0, 1, 1, 0])
    r = np.array([0, 0, 1, 0, 1])
    assert A.mcnemar(h, r)[0] == pytest.approx(A.mcnemar(r, h)[0])


def test_clopper_pearson_brackets_the_point():
    import phase5_analyze_v2 as A
    lo, hi = A.clopper_pearson(50, 1000)
    assert lo < 0.05 < hi
    # 95 % interval on 50/1000: [0.0371, 0.0658]; a 68 % interval would be far narrower
    assert 0.036 < lo < 0.039 and 0.064 < hi < 0.068
    lo0, hi0 = A.clopper_pearson(0, 100)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.05


def test_cluster_bootstrap_is_reproducible_and_centred():
    from thresholds_frozen import cluster_bootstrap
    rng = np.random.default_rng(0)
    cells, scores = [], []
    for acq in ("a", "b", "c"):
        for c in range(12):
            v = rng.exponential(size=40) + 0.1 * c
            cells += [(acq, c)] * len(v)
            scores += list(v)
    r1 = cluster_bootstrap(scores, cells, 1e-2, seed=7, n_boot=200)
    r2 = cluster_bootstrap(scores, cells, 1e-2, seed=7, n_boot=200)
    assert r1 == r2
    assert r1["ci95"][0] <= r1["mean"] <= r1["ci95"][1]
    assert r1["rel_sd"] > 0
    r3 = cluster_bootstrap(scores, cells, 1e-2, seed=8, n_boot=200)
    assert r3["ci95"] != r1["ci95"]


def test_stratified_bootstrap_needs_acquisition_labels():
    from thresholds_frozen import cluster_bootstrap
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0, 2.0, 3.0], ["x", "y", "z"], 1e-2, seed=0, n_boot=5)


def test_paired_indices_keep_the_two_arms_aligned():
    """Both arms must be resampled with the same cell indices, or the contrast is lost."""
    import phase5_analyze_v2 as A
    acqs = ["a"] * 4 + ["b"] * 4
    cells = [1, 1, 2, 2, 1, 1, 2, 2]
    table = A.cell_table(acqs, cells)
    i1 = A.draw_indices(np.random.default_rng(3), table)
    i2 = A.draw_indices(np.random.default_rng(3), table)
    assert np.array_equal(i1, i2)
    # a two-stage resample of the same size: every acquisition contributes its own cell count
    assert len(i1) == len(np.unique(table["cells"]))


def test_bootstrap_uses_the_same_draws_for_every_arm():
    """``boot_binary`` draws once per replicate and applies that draw to every arm column.

    The check is direct: ``draw_indices`` must be entered once per replicate, not once per
    arm.  A per-arm draw would destroy the pairing and make every contrast invalid.
    """
    import phase5_analyze_v2 as A
    acqs = ["a"] * 4 + ["b"] * 4
    cells = [1, 1, 2, 2, 1, 1, 2, 2]
    y = np.array([[1, 1], [1, 1], [0, 0], [1, 1],
                  [1, 1], [0, 0], [1, 1], [1, 1]], dtype=bool)
    table = A.cell_table(acqs, cells, y)

    calls = {"n": 0}
    real = A.draw_indices

    def counted(rng, tbl):
        calls["n"] += 1
        return real(rng, tbl)

    A.draw_indices = counted
    try:
        a = A.boot_binary(table, np.random.default_rng(11), 50)
    finally:
        A.draw_indices = real

    assert calls["n"] == 50, "one resample per replicate must serve both arms"
    assert a.shape == (50, 2)
    lo, hi = A.ci(a[:, 0])
    assert lo <= hi
    # identical arms must give identical bootstrap columns from the same draw
    assert np.array_equal(a[:, 0], a[:, 1])
    b = A.boot_binary(table, np.random.default_rng(11), 50)
    assert np.array_equal(a, b)
