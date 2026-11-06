"""Single source of truth for every filesystem location used by the released pipeline.

The original research workspace addressed each part of the project by an absolute path on
one Windows workstation (``F:\\ClaudeCode\\Topic 1\\...``).  This module replaces those
paths with the repository layout, and exposes three environment variables for the large
inputs that are not distributed with the repository:

``PAPER1_CACHE``
    Directory of IPIX complex-cube ``.npz`` files (default ``data/ipix_cache``).
``PAPER1_CHECKPOINTS``
    Directory holding the released ``.pt`` checkpoints (default ``checkpoints``).
``PAPER1_OUTPUTS``
    Directory for anything the pipeline writes (default ``outputs``).

Nothing here imports the ``src`` package, so it is safe to import from tests and from
scripts before the numerical stack is available.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PIPELINE = REPO_ROOT / "pipeline"
PROTOCOL = REPO_ROOT / "protocol"
CONFIGS = REPO_ROOT / "configs"
METADATA = REPO_ROOT / "metadata"
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"
DOCS = REPO_ROOT / "docs"

OUTPUTS = Path(os.environ.get("PAPER1_OUTPUTS", REPO_ROOT / "outputs"))
CACHE = Path(os.environ.get("PAPER1_CACHE", REPO_ROOT / "data" / "ipix_cache"))
CHECKPOINTS = Path(os.environ.get("PAPER1_CHECKPOINTS", REPO_ROOT / "checkpoints"))

PHASE7_DATA = RESULTS / "phase7" / "data"
PHASE7_REPORTS = RESULTS / "phase7"

#: Two-fold acquisition design used by every experiment in the paper.
FOLDS = ("fold1", "fold2")


def ensure_outputs() -> Path:
    """Create the output tree and return its root."""
    for sub in ("reports", "statistics", "figures", "runs", "phase7"):
        (OUTPUTS / sub).mkdir(parents=True, exist_ok=True)
    return OUTPUTS


def describe() -> dict:
    """Machine-readable summary of the resolved layout (used by ``scripts/doctor.py``)."""
    return {
        "repo_root": str(REPO_ROOT),
        "outputs": str(OUTPUTS),
        "cache": str(CACHE),
        "cache_present": CACHE.is_dir(),
        "checkpoints": str(CHECKPOINTS),
        "checkpoints_present": CHECKPOINTS.is_dir(),
        "configs_present": (CONFIGS / "campaign.json").is_file(),
        "protocol_present": (PROTOCOL / "DATA_PARTITION_MANIFEST.csv").is_file(),
        "released_results_present": (RESULTS / "phase5" / "MINIMAL_CAMPAIGN_RESULTS.csv").is_file(),
    }
