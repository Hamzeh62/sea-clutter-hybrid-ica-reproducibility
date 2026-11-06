#!/usr/bin/env python
"""Verify that the released files are the files the experiments used.

Three independent chains are checked.

1. ``manifests/FROZEN_INPUTS.json`` -- the SHA-256 of every frozen protocol input, as
   declared in ``configs/campaign.json``.  Six of the seven verify byte for byte against
   ``protocol/``; ``acquisition_ids.py`` is a documented derivative, so its pristine copy in
   ``protocol/original/`` is what must match.
2. ``configs/campaign.json``'s own ``config_sha256``, recomputed exactly as
   ``phase5_common.config()`` does at run time.
3. ``manifests/SHA256SUMS.txt`` -- every other released file, in ``sha256sum`` format.

Usage
-----
    python scripts/verify_artifacts.py            # verify
    python scripts/verify_artifacts.py --write    # (re)write SHA256SUMS.txt first
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMS = ROOT / "manifests" / "SHA256SUMS.txt"
SKIP_DIRS = {"outputs", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_FILES = {"SHA256SUMS.txt"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def walk(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.name in SKIP_FILES:
            continue
        yield rel


def write_sums() -> int:
    lines = [f"{sha256(ROOT / rel)}  {rel.as_posix()}" for rel in walk(ROOT)]
    SUMS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {SUMS} ({len(lines)} files)")
    return 0


def check_frozen() -> list[str]:
    bad = []
    spec = json.loads((ROOT / "manifests/FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    for name, rec in spec["inputs"].items():
        p = ROOT / rec["release_path"]
        if not p.is_file():
            bad.append(f"missing frozen input {name} -> {rec['release_path']}")
        elif sha256(p) != rec["sha256"]:
            bad.append(f"frozen input {name} does not match its declared hash")
    for rel, digest in spec.get("auxiliary", {}).items():
        p = ROOT / rel
        if not p.is_file() or sha256(p) != digest:
            bad.append(f"auxiliary file {rel} does not match")
    return bad


def check_config() -> list[str]:
    cfg = json.loads((ROOT / "configs/campaign.json").read_text(encoding="utf-8"))
    declared = cfg["config_sha256"]
    body = dict(cfg)
    body.pop("config_sha256")
    actual = hashlib.sha256(json.dumps(body, indent=2, sort_keys=True).encode()).hexdigest()
    return [] if actual == declared else [
        f"config re-hash differs: declared {declared[:12]} actual {actual[:12]}"]


def check_sums() -> list[str]:
    if not SUMS.is_file():
        return ["manifests/SHA256SUMS.txt is absent; run --write"]
    bad = []
    listed = 0
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        p = ROOT / rel
        listed += 1
        if not p.is_file():
            bad.append(f"listed but absent: {rel}")
        elif sha256(p) != digest:
            bad.append(f"hash mismatch: {rel}")
    present = {rel.as_posix() for rel in walk(ROOT)}
    listed_paths = {ln.split("  ", 1)[1] for ln in
                    SUMS.read_text(encoding="utf-8").splitlines() if ln.strip()}
    for extra in sorted(present - listed_paths):
        bad.append(f"present but not listed: {extra}")
    print(f"   {listed} files listed")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    if args.write:
        write_sums()

    problems = []
    print("[1] frozen protocol inputs")
    problems += check_frozen()
    print("[2] campaign config hash")
    problems += check_config()
    print("[3] file manifest")
    problems += check_sums()

    if problems:
        print("\nFAILED:")
        for p in problems:
            print("  ", p)
        return 1
    print("\nARTIFACT INTEGRITY VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
