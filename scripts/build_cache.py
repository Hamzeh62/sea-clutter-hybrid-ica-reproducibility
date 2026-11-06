#!/usr/bin/env python
"""Build the complex-cube cache from IPIX NetCDF recordings.

The cache is what the detector code reads: one ``.npz`` per acquisition, holding the four
complex polarisation cubes plus the pulse repetition frequency and the range axis.  It is
*not* distributed with this repository, because it contains the recordings themselves; see
``docs/DATA_RIGHTS_AND_PROVENANCE.md``.

Usage
-----
    python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache
    python scripts/build_cache.py --cdf-dir ... --verify-only      # check what is missing

The output name follows the convention the sampler expects: ``<stem>_cubes.npz`` for a
recording named ``<stem>.cdf``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_one(src: Path, out_dir: Path, force: bool = False) -> Path:
    from src.data.ipix_loader import complex_cubes, read_ipix_file
    dst = out_dir / (src.stem + "_cubes.npz")
    if dst.is_file() and not force:
        return dst
    rec = read_ipix_file(src)
    cubes = complex_cubes(rec)
    if "VV" not in cubes:
        raise SystemExit(f"{src.name}: no VV channel in this recording "
                         f"(TX_polarization={rec['attrs'].get('TX_polarization')!r})")
    payload = {k: v.astype(np.complex64) for k, v in cubes.items()}
    payload["prf"] = np.float64(float(rec["attrs"]["PRF"]))
    rng = rec["data"].get("range")
    payload["ranges"] = (np.asarray(rng, dtype=np.float32) if rng is not None
                         else np.arange(payload["VV"].shape[1], dtype=np.float32))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".npz.tmp")
    np.savez_compressed(tmp, **payload)
    tmp.replace(dst)
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdf-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "ipix_cache"))
    ap.add_argument("--force", action="store_true", help="rebuild existing cubes")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    src_dir, out = Path(args.cdf_dir), Path(args.out)
    files = sorted([*src_dir.glob("*.cdf"), *src_dir.glob("*.CDF")])
    if not files:
        print(f"no .cdf recordings under {src_dir}")
        return 2

    if args.verify_only:
        # The study needs exactly the 16 acquisitions named in metadata/target_cells.json;
        # anything else is an extra, anything absent is a gap that blocks Level 2.
        import json
        meta = json.loads((ROOT / "metadata" / "target_cells.json").read_text(encoding="utf-8"))
        have = {p.name for p in out.glob("*_cubes.npz")}
        missing = [k for k in meta if k.replace(".cdf", "") + "_cubes.npz" not in have]
        print(f"{len(meta) - len(missing)}/{len(meta)} cubes present in {out}")
        for m in sorted(missing):
            print("   missing:", m)
        return 0 if not missing else 1

    for f in files:
        print(f"[cache] {f.name}")
        build_one(f, out, args.force)
    print(f"\ncache ready at {out} ({len(files)} recordings processed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
