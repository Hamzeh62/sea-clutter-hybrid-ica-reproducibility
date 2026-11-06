#!/usr/bin/env python
"""Report whether this checkout can run each reproduction level.

Level 1 needs only the released tables (always available).
Level 2 needs the IPIX cube cache and the released checkpoints.
Level 3 needs the original NetCDF recordings and a GPU-free PyTorch install.

Nothing is written and nothing is downloaded.

Usage
-----
    python scripts/doctor.py
    python scripts/doctor.py --json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

MIN_PY = (3, 10)


def version(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "installed")
    except Exception as exc:                      # noqa: BLE001 - report, do not raise
        return f"NOT AVAILABLE ({type(exc).__name__})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    import paths as P
    d = P.describe()
    py = sys.version_info
    deps = {m: version(m) for m in ("numpy", "scipy", "torch", "yaml", "matplotlib",
                                    "pandas", "pytest", "psutil")}
    ck = sorted(p.name for p in P.CHECKPOINTS.glob("*.pt")) if P.CHECKPOINTS.is_dir() else []
    cubes = len(list(P.CACHE.glob("*_cubes.npz"))) if P.CACHE.is_dir() else 0

    level1 = d["released_results_present"]
    level2 = level1 and cubes >= 16 and len(ck) >= 2
    level3 = level2 and not str(deps["torch"]).startswith("NOT")

    report = {
        "python": platform.python_version(),
        "python_ok": py >= MIN_PY,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "paths": d,
        "dependencies": deps,
        "checkpoints_found": ck,
        "ipix_cubes_found": cubes,
        "levels": {"level1_tables_figures": level1,
                   "level2_detector_evaluation": level2,
                   "level3_training": level3},
    }
    if args.json:
        print(json.dumps(report, indent=1))
        return 0

    print(f"python      {report['python']}  ({'ok' if py >= MIN_PY else 'too old, need >= '
           + '.'.join(map(str, MIN_PY))})")
    print(f"platform    {report['platform']}  {report['machine']}")
    print("\ndependencies")
    for k, v in deps.items():
        print(f"   {k:12s} {v}")
    print("\nlayout")
    for k in ("repo_root", "outputs", "cache", "checkpoints"):
        print(f"   {k:12s} {d[k]}")
    print(f"   {'cubes':12s} {cubes}/16 found")
    print(f"   {'weights':12s} {len(ck)} checkpoint(s) found")
    print("\nreproduction levels")
    print(f"   Level 1  tables and figures from released data   "
          f"{'READY' if level1 else 'BLOCKED'}")
    print(f"   Level 2  detector evaluation on IPIX cubes       "
          f"{'READY' if level2 else 'NEEDS DATA/WEIGHTS'}")
    print(f"   Level 3  network training                        "
          f"{'READY' if level3 else 'NEEDS DATA/WEIGHTS'}")
    if not (py >= MIN_PY):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
