"""Write PHASE7_SHA256_MANIFEST.txt: every Phase 7 artefact plus the frozen inputs it reads.

Format matches the Phase 6 manifest: ``<sha256>  <path relative to Paper1_Revision_Working>  <bytes>``
so the two can be verified by the same reader.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORK7 = PROJECT / "results" / "phase7"
RUN = PROJECT / "phase5_minimal/runs/minimal_v1_20261029"
OUT = PROJECT / "outputs" / "phase7" / "reports" / "PHASE7_SHA256_MANIFEST.txt"


def sha(p: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def main() -> int:
    files: list[Path] = []
    for sub in ("code", "reports", "data", "figures", "configs"):
        files += [p for p in sorted((WORK7 / sub).rglob("*"))
                  if p.is_file() and p != OUT and "__pycache__" not in p.parts]
    # frozen inputs: campaign config, thresholds, manifests, checkpoints
    files += [p for p in sorted(RUN.glob("configs/*")) if p.is_file()]
    files += [p for p in (PROJECT / "phase6/reports/PHASE6_SHA256_MANIFEST.txt",
                          PROJECT / "ipix_hybrid_detector/results/checkpoints/best_s0.pt",
                          PROJECT / "ipix_hybrid_detector/results/checkpoints/best_s0_nophys.pt",
                          PROJECT / "phase6/configs/PHASE6_SIGNOFF.json")
              if p.is_file()]
    lines = [f"# PHASE7 SHA-256 manifest - Paper 1 confirmatory experiments",
             f"# campaign          minimal_v1_20261029",
             f"# generated_utc     {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             f"# interpreter       {platform.python_version()}  {sys.executable}",
             f"# platform          {platform.platform()}",
             f"# design_pfa        {json.loads((WORK7 / 'configs/PHASE7_SIGNOFF.json')
                                          .read_text(encoding='utf-8'))['design_pfa']}",
             f"# files             {len(files)}",
             "# format            <sha256>  <path relative to Paper1_Revision_Working>  <bytes>",
             ""]
    total = 0
    for p in files:
        h, n = sha(p)
        total += n
        lines.append(f"{h}  {p.relative_to(OUT.parents[2]).as_posix()}  {n}")
    lines.insert(6, f"# bytes             {total}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[manifest] {len(files)} files, {total/1e6:.2f} MB -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
