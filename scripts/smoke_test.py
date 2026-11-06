#!/usr/bin/env python
"""End-to-end smoke test on synthetic compound-Gaussian clutter.

What this checks
----------------
* every detector path used by the campaign executes and returns a finite statistic;
* the ANMF detector responds to an injected coherent target;
* the AR(8) reference mechanism reproduces on synthetic data: a predictor fitted on a
  reference that contains the target cancels it, a predictor fitted on a clean reference
  does not;
* the statistic is deterministic for a fixed seed.

What this does **not** check
----------------------------
It does not reproduce any published number.  The published results require the IPIX
recordings (see ``docs/DATA_DOWNLOAD_INSTRUCTIONS.md``) and the released checkpoints.
Passing this test means the code runs and behaves qualitatively as described; it says
nothing about the IPIX evaluation, and it is reported that way in
``docs/REPRODUCIBILITY_VALIDATION_REPORT.md``.

Usage
-----
    python scripts/smoke_test.py
    python scripts/smoke_test.py --checkpoint checkpoints/best_s0.pt   # adds the hybrid path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

NP = 512
PRF = 1000.0
NU, B = 1.5, 2.0          # K-distribution shape and scale of the synthetic clutter


def clutter_pair(rng: np.random.Generator, n: int = NP) -> np.ndarray:
    """Two-correlated-channel compound-Gaussian CPI, shape (2, n)."""
    from src.data.synth import synth_compound_gaussian
    psd = np.ones(512, dtype=float)
    vv = synth_compound_gaussian(n, NU, B, psd, 0.9, rng)
    vh = 0.6 * vv + synth_compound_gaussian(n, NU, B, psd, 0.9, rng)
    return np.stack([vv, vh])


def tone(n: int = NP, fd: float = 137.0, amp: float = 1.0) -> np.ndarray:
    from src.data.synth import swerling_target
    return swerling_target(n, fd, PRF, amp, 0, np.random.default_rng(0))


def sep_prep(n: int = NP, seed: int = 0):
    """A C_sep estimated from arbitrary complex windows (code path only, not physics).

    The dimension follows the domain: ``prep_sep_from_windows`` is called on the rows it
    will later score, so an AR(8) residual prep is ``N_p - 8`` long exactly as in
    ``phase5_runner.run_prep``.
    """
    from src.proposed.m4_detector import prep_sep_from_windows
    rng = np.random.default_rng(seed)
    win = (rng.standard_normal((250, n)) + 1j * rng.standard_normal((250, n)))
    return prep_sep_from_windows(win)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args(argv)

    from src.baselines.anmf import anmf_stat, glrt_stat, steering_bank
    from src.data.synth import synth_compound_gaussian
    from src.proposed.m4_detector import nmf_statistic
    import phase5_common as C

    rng = np.random.default_rng(20261101)
    P = steering_bank(NP)
    psd = np.ones(512, dtype=float)
    secondary = np.stack([synth_compound_gaussian(NP, NU, B, psd, 0.9, rng)
                          for _ in range(600)])

    print("== 1. ANMF / GLRT on synthetic clutter ==")
    y_h0 = synth_compound_gaussian(NP, NU, B, psd, 0.9, rng)
    a_h0 = anmf_stat(y_h0, secondary)
    g_h0 = glrt_stat(y_h0, secondary)
    s = tone(amp=np.sqrt(np.mean(np.abs(y_h0) ** 2) * 10 ** (20 / 10)))
    a_h1 = anmf_stat(y_h0 + s, secondary)
    print(f"   ANMF-DL  H0 {a_h0:10.4f}   H1(+20 dB) {a_h1:10.4f}")
    print(f"   GLRT     H0 {g_h0:10.4f}")
    ok = True
    ok &= np.isfinite([a_h0, g_h0, a_h1]).all()
    ok &= a_h1 > a_h0
    print(f"   [{'PASS' if a_h1 > a_h0 else 'FAIL'}] a +20 dB target raises the ANMF statistic")

    print("== 2. AR(8) / notch against the reference ==")
    prep_ar = sep_prep(NP - 8, seed=1)
    prep_notch = sep_prep(NP, seed=2)
    x = clutter_pair(rng)
    sig = tone(amp=np.sqrt(np.mean(np.abs(x[0]) ** 2) * 10 ** (10 / 10)))
    x_cont = x.copy()
    x_cont[0] += sig          # target in the detection channel
    x_cont_ref = x_cont.copy()
    x_cont_ref[1] += sig      # and in the reference (the modelled cross-polarised leak)

    # clean reference: predictor knows nothing about the target
    ar_clean = nmf_statistic(C.ar_residual(x_cont[0], C.ar_fit(x[1])), prep_ar)
    # contaminated reference: the predictor is fitted on data containing the target
    ar_cont = nmf_statistic(C.ar_residual(x_cont_ref[0], C.ar_fit(x_cont_ref[1])), prep_ar)
    notch_clean = nmf_statistic(C.notch_vector(x_cont[0]).conj() @ x_cont, prep_notch)
    notch_cont = nmf_statistic(C.notch_vector(x_cont_ref[0]).conj() @ x_cont_ref, prep_notch)
    print(f"   AR(8)+NMF   clean reference {ar_clean:10.4f}   contaminated {ar_cont:10.4f}")
    print(f"   Notch       clean reference {notch_clean:10.4f}   contaminated {notch_cont:10.4f}")
    ok &= np.isfinite([ar_clean, ar_cont, notch_clean, notch_cont]).all()
    cancelled = ar_cont < ar_clean
    print(f"   [{'PASS' if cancelled else 'FAIL'}] the contaminated reference lowers the "
          f"AR(8) statistic ({ar_cont / max(ar_clean, 1e-30):.3f}x)")
    ok &= cancelled

    print("== 3. determinism ==")
    again = nmf_statistic(C.ar_residual(x_cont[0], C.ar_fit(x[1])), prep_ar)
    same = again == ar_clean
    print(f"   [{'PASS' if same else 'FAIL'}] identical inputs give an identical statistic")
    ok &= same

    if args.checkpoint:
        print("== 4. hybrid path with a released checkpoint ==")
        from src.proposed.infer import HybridDetector
        det = HybridDetector(Path(args.checkpoint))
        stat, _ = det.statistic(x, prep_sep=sep_prep(NP, seed=3))
        print(f"   hybrid statistic {stat:.6f}")
        ok &= bool(np.isfinite(stat))
        print(f"   [{'PASS' if np.isfinite(stat) else 'FAIL'}] hybrid statistic is finite")
    else:
        print("== 4. hybrid path skipped (no --checkpoint) ==")

    print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
