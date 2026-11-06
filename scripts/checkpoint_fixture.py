#!/usr/bin/env python
"""Freeze and verify a deterministic forward-pass fixture for the released checkpoints.

The fixture answers two separate questions.

1. *Is this the same file?*  ``checkpoint_sha256`` is compared with the SHA-256 published in
   ``manifests/CHECKPOINT_INVENTORY.csv``.
2. *Does this file still produce the published behaviour?*  The network is evaluated on a
   fixed synthetic input and the resulting statistic and parameter digest are compared with
   the frozen fixture.

The parameter digest is over the ``state_dict`` tensors in sorted-key order as little-endian
float32, so it is independent of the ``.pt`` container format.  The statistic is compared
with a tolerance because a different PyTorch build may reassociate floating-point operations;
the digest is compared exactly because it reads the stored weights.

Usage
-----
    python scripts/checkpoint_fixture.py --make    --checkpoints checkpoints
    python scripts/checkpoint_fixture.py --verify  --checkpoints checkpoints
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

FIXTURE = ROOT / "manifests" / "CHECKPOINT_FIXTURE.json"
RELEASED = {"hybrid": "best_s0.pt", "nophys": "best_s0_nophys.pt"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fixed_input(seed: int = 20261101) -> np.ndarray:
    """Deterministic two-channel CPI: correlated K-clutter plus a +6 dB tone."""
    from src.data.synth import synth_compound_gaussian, swerling_target
    rng = np.random.default_rng(seed)
    psd = np.ones(512, dtype=float)
    vv = synth_compound_gaussian(512, 1.5, 2.0, psd, 0.9, rng)
    vh = 0.6 * vv + synth_compound_gaussian(512, 1.5, 2.0, psd, 0.9, rng)
    t = swerling_target(512, 137.0, 1000.0,
                        np.sqrt(np.mean(np.abs(vv) ** 2) * 10 ** (6 / 10)), 0, rng)
    return np.stack([vv + t, vh + 0.5 * t])


def param_digest(model) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for key in sorted(model.state_dict()):
        arr = model.state_dict()[key].detach().cpu().numpy()
        h.update(key.encode())
        h.update(np.ascontiguousarray(arr.astype("<f4")).tobytes())
        n += int(arr.size)
    return h.hexdigest(), n


def measure(ckpt: Path) -> dict:
    import torch
    torch.set_num_threads(1)
    from src.proposed.infer import HybridDetector
    det = HybridDetector(ckpt)
    x = fixed_input()
    out = det.network_out(x)
    stat, _ = det.statistic(x, prep_sep=_prep())
    return {
        "checkpoint_sha256": file_sha256(ckpt),
        "parameter_sha256": param_digest(det.model)[0],
        "n_parameters": param_digest(det.model)[1],
        "a_hat_abs_sum": float(np.abs(out["A"]).sum()),
        "r_abs_sum": float(np.abs(out["r"]).sum()),
        "statistic": float(stat),
    }


_PREP_CACHE: dict = {}


def _prep():
    """A C_sep of the right dimension, built once from a fixed seed."""
    if not _PREP_CACHE:
        from src.proposed.m4_detector import prep_sep_from_windows
        rng = np.random.default_rng(7)
        win = rng.standard_normal((250, 512)) + 1j * rng.standard_normal((250, 512))
        _PREP_CACHE["prep"] = prep_sep_from_windows(win)
    return _PREP_CACHE["prep"]


TOL = {"statistic": 1e-6, "a_hat_abs_sum": 1e-4, "r_abs_sum": 1e-4}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    args = ap.parse_args(argv)
    ckdir = Path(args.checkpoints)

    missing = [n for n in RELEASED.values() if not (ckdir / n).is_file()]
    if missing:
        print(f"checkpoints not found in {ckdir}: {missing}")
        print("download them from the archive record (see checkpoints/README.md) or pass "
              "--checkpoints")
        return 2

    got = {name: measure(ckdir / fn) for name, fn in RELEASED.items()}

    if args.make:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps({"input_seed": 20261101, "tolerance": TOL,
                                       "checkpoints": got}, indent=2), encoding="utf-8")
        print(f"wrote {FIXTURE}")
        for name, rec in got.items():
            print(f"  {name}: stat {rec['statistic']:.9f}  params {rec['n_parameters']}")
        return 0

    if not FIXTURE.is_file():
        print(f"no fixture at {FIXTURE}; run --make first")
        return 2
    ref = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ok = True
    for name, rec in got.items():
        exp = ref["checkpoints"][name]
        same_file = rec["checkpoint_sha256"] == exp["checkpoint_sha256"]
        same_params = rec["parameter_sha256"] == exp["parameter_sha256"]
        d = abs(rec["statistic"] - exp["statistic"])
        print(f"{name}: file {'OK' if same_file else 'MISMATCH'}  "
              f"params {'OK' if same_params else 'MISMATCH'}  "
              f"statistic {rec['statistic']:.9f} vs {exp['statistic']:.9f} "
              f"(|d|={d:.2e})")
        ok &= same_file and same_params and d <= TOL["statistic"]
    print("CHECKPOINT FIXTURE", "VERIFIED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
