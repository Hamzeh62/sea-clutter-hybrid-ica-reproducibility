#!/usr/bin/env python
"""Minimal end-to-end example: the hybrid statistic on one synthetic CPI.

This is the shortest path that exercises the deployed detection chain -- network, mixing head,
anchored complex ICA and the separated NMF statistic -- without any IPIX data.

    python examples/quickstart.py --checkpoint checkpoints/best_s0.pt

It prints the statistic and the recovered mixing matrix.  It does not reproduce a published
number: the published numbers require the IPIX recordings and the campaign runner
(see docs/REPRODUCE_THE_PAPER.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NP = 512          # CPI length used throughout the paper
PRF = 1000.0


def synthetic_cpi(seed: int = 20261101) -> np.ndarray:
    """Correlated two-channel K-clutter with a +6 dB coherent target in VV and VH."""
    from src.data.synth import synth_compound_gaussian, swerling_target
    rng = np.random.default_rng(seed)
    psd = np.ones(NP, dtype=float)
    vv = synth_compound_gaussian(NP, 1.5, 2.0, psd, 0.9, rng)
    vh = 0.6 * vv + synth_compound_gaussian(NP, 1.5, 2.0, psd, 0.9, rng)
    amp = np.sqrt(np.mean(np.abs(vv) ** 2) * 10 ** (6 / 10))
    t = swerling_target(NP, 137.0, PRF, amp, 0, rng)
    return np.stack([vv + t, vh + 0.5 * t])


def separation_matrix(seed: int = 7):
    """A C_sep of the deployed dimension, estimated from fixed synthetic windows."""
    from src.proposed.m4_detector import prep_sep_from_windows
    rng = np.random.default_rng(seed)
    win = rng.standard_normal((250, NP)) + 1j * rng.standard_normal((250, NP))
    return prep_sep_from_windows(win)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best_s0.pt"))
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"checkpoint not found: {ckpt}")
        print("See checkpoints/README.md for how to obtain it.")
        return 2

    from src.proposed.infer import HybridDetector
    det = HybridDetector(ckpt)
    x = synthetic_cpi()
    stat, info = det.statistic(x, prep_sep=separation_matrix())
    net = det.network_out(x)

    print(f"input          2 x {NP} complex samples, PRF {PRF:g} Hz")
    print(f"statistic      {stat:.6f}")
    print(f"ICA sweeps     {info['iters']} ({'warm' if info['warm'] else 'cold'} start)")
    print(f"mixing matrix  {net['A'].shape}; |A| =")
    print(np.array2string(np.abs(net["A"]), precision=4, suppress_small=True))
    print(f"separated target source: mean |s_T| = {np.mean(np.abs(info['s_T'])):.4f}")
    print("\nNote: this is a synthetic CPI. No published result is reproduced by this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
