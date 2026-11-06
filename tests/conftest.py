"""Shared fixtures for the release test suite.

Two paths are added once, at collection time: the repository root (for ``src``) and
``pipeline/`` (whose modules are imported by filename, exactly as in the original
research workspace).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "pipeline", ROOT / "protocol"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    import paths as P
    if not P.CACHE.is_dir() or not list(P.CACHE.glob("*_cubes.npz")):
        pytest.skip("IPIX cube cache not present (set PAPER1_CACHE; see data/README.md)")
    return P.CACHE


@pytest.fixture(scope="session")
def checkpoints() -> Path:
    import paths as P
    if not (P.CHECKPOINTS / "best_s0.pt").is_file():
        pytest.skip("released checkpoints not present (see checkpoints/README.md)")
    return P.CHECKPOINTS
