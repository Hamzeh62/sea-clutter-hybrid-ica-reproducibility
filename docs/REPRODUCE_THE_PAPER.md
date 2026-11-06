> **Version scope:** The validation below pertains to the earlier Phase 8 two-figure manuscript. The restored SIVP article additionally retains original-protocol Swerling/ROC/Ablation evidence and must not be declared fully re-generated from this lightweight archive. See `RESTORED_MANUSCRIPT_REPRODUCIBILITY.md`.

# Reproduce the paper

This guide is written for a reviewer who has the repository and wants to check the manuscript's
numbers in the order they matter. Every command below was executed in this release; the measured
result is quoted next to it. Nothing here requires a GPU.

**Time to complete Levels 1 and 2 (with data and weights already in place): about 15 minutes.**
Level 1 alone is about 30 seconds.

---

## 1. Obtain the code

```bash
git clone <repository-url>            # URL assigned when the repository is published
cd sea-clutter-hybrid-ica-reproducibility
```

If you received a ZIP instead, extract it and `cd` into the extracted directory. The directory must
contain `README.md`, `scripts/`, `results/` and `manifests/`.

## 2. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
python -m pip install -r requirements.txt
```

The pinned set is CPython 3.12.10, NumPy 2.5.2, SciPy 1.18.1, PyTorch 2.13.0 (CPU wheel),
Matplotlib 3.11.1, PyYAML 6.0.3, pytest 9.1.1. A conda equivalent is in `environment.yml`. POSIX
commands are expected to work but were not exercised in this release.

## 3. Confirm the environment and the available levels

```bash
python scripts/doctor.py
```

Measured output on the release machine: Python 3.12.10, `platform Windows-11`, Level 1 `READY`,
Level 2 `NEEDS DATA/WEIGHTS` until steps 4–5 are done. The exit status is 0 if Python is new enough.

## 4. Obtain the data

The 16 IPIX recordings are third-party data and are not distributed here. Follow
`DATA_DOWNLOAD_INSTRUCTIONS.md`. Once you have the recordings:

```bash
python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache
python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache --verify-only
```

Expected: `16/16 cubes present in data/ipix_cache`. If you already have a cube cache, set
`PAPER1_CACHE` to it instead of rebuilding. **Skip this step entirely if you only need Level 1.**

## 5. Obtain the checkpoints

Download `PAPER1_CHECKPOINTS_v1.0.0.zip` from the archive record (`RELEASE_NOTES.md`), extract into
`checkpoints/`, then:

```bash
python scripts/checkpoint_fixture.py --verify --checkpoints checkpoints
```

Expected: `CHECKPOINT FIXTURE VERIFIED` for both `best_s0.pt` and `best_s0_nophys.pt`. The check is
file SHA-256 + `state_dict` parameter digest + a deterministic statistic on a fixed synthetic CPI.
**Skip if you only need Level 1.**

## 6. Verify the artifact hashes

```bash
python scripts/verify_artifacts.py
```

This checks three independent chains: the seven frozen protocol inputs against
`manifests/FROZEN_INPUTS.json` (six byte-identical, `acquisition_ids.py` matched via its pristine
copy in `protocol/original/`), the campaign configuration's own `config_sha256` recomputed exactly as
the runner does, and every released file against `manifests/SHA256SUMS.txt`.

Expected: `ARTIFACT INTEGRITY VERIFIED`, exit 0.

`scripts/verify_artifacts.py --write` regenerates `SHA256SUMS.txt` if you have modified your working
copy for your own experiments; do not use it to "fix" a mismatch you did not cause.

## 7. Reproduce the main figures and tables (Level 1)

```bash
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py --check
```

Measured:

```
T1 25/25   T2 24/24   T3 24/24   T4 24/24   T5 24/24   T6 12/12
133/133 published values reproduced (base tolerance 0.0006); 0 failing

fig1_mechanism: mean |delta pixel| = 0.0000 / 255  [PASS]
fig2_calibration: mean |delta pixel| = 0.0000 / 255  [PASS]
```

`manifests/PUBLISHED_VALUES.json` was transcribed from the manuscript, so this is a test of the
repository against the paper, not of the paper against itself. `--json` gives machine-readable
output for CI. `--show-unlisted` prints values the tables allow you to compute but the paper does not
print.

## 8. Recompute the principal statistics

The statistics live in the tables the previous step recomputed. To inspect one directly:

```bash
python -m pytest -q tests/test_statistics.py
```

Expected: `13 passed`. These tests assert the properties the paper's intervals rely on: exact
two-sided McNemar against `scipy.stats.binomtest`, Clopper–Pearson coverage, one shared bootstrap
draw per replicate for paired arms, reproducibility of the cluster bootstrap for a fixed seed, and
rejection of unstratified resampling when acquisition labels are missing.

The full suite:

```bash
python -m pytest -q
```

Expected: `34 passed` with the cube cache present; `31 passed, 3 skipped` without it (the three
acquisition-table tests skip because they need the recordings).

## 9. Re-run a selected corrected IPIX experiment (Level 2)

This is the strongest check available: it rebuilds the calibration preparation from the cubes and
re-runs one released scoring job, then compares every array with the released artifact.

```bash
python scripts/validate_against_released_scores.py ^
  --released /path/to/results/runs/minimal_v1_20261029 ^
  --job "h1|fold1|hybrid|c0"
```

You need the score archive (`PAPER1_RESULTS_v1.0.0.zip`), which contains
`runs/minimal_v1_20261029/`. Measured on the release machine, one thread, 4-core CPU:

```
[1/3] prep|fold1|hybrid        47 s   prep Chat/Cinv/CinvP/a  PASS  max|d| = 0.0
[2/3] h1|fold1|hybrid|c0       96 s
[3/3] all arrays               EXACT  arm_hybrid PASS  max|d| = 0.0  rel = 0.000e+00
LEVEL 2 REPRODUCTION EXACT
```

Two cores and ~2.5 minutes are sufficient. Elapsed times on a busy machine may be 2x longer; the
comparison is unaffected. The script pins BLAS and torch to one thread because the released numbers
were produced that way and float32 reductions are order-sensitive.

Other jobs can be substituted (`h1|fold2|anmf_dl|c0`, etc.), but each group needs its own
preparation; the script says so when a preparation is not applicable.

## 10. Where the limits are

Read these before drawing a conclusion about the paper from a successful run.

* **No real-target detection probability exists in this package.** The acquisitions carry no labelled
  real target. Every `P_d` is measured against an injected synthetic target.
* **The polarimetric contrast is a proxy** with an ~18 dB interval (median −1.8 dB). It is not a
  calibrated measurement and the paper does not present it as one.
* **10⁻³ is not certified and 10⁻⁴ is not claimed.** No clutter-only rule realises the nominal rate
  across acquisitions; that failure is the calibration contribution, not a defect of the release.
* **Sixteen acquisitions, two folds.** Cluster-aware intervals are reported over acquisitions, but no
  population guarantee follows.
* **Level 3 (training) was not executed.** `src/proposed/train.py` is provided unmodified; no claim of
  exact retraining reproducibility is made.
* **The score archive is needed only for Level 2.** If you do not have it, you can still verify every
  published number and both figures from `results/`.

## Numerical tolerance policy

| quantity | tolerance | rationale |
|---|---|---|
| released score arrays | `max|Δ| ≤ 1e-9` (measured 0.0) | one BLAS/torch thread reproduces the archived arithmetic |
| published values | `6e-4` absolute | half a unit in the third decimal, the precision printed in the paper |
| figure pixels | mean absolute difference ≤ 1.0/255 (measured 0.0) | identical data and identical plotting code; only font substitution could move it |
| checkpoint statistic | `1e-6` absolute | different torch builds may reassociate reductions |
| checkpoint identity | exact SHA-256 and exact parameter digest | identity, not behaviour |

## Reporting a discrepancy

Open an issue with: the command, the full output, your OS/Python/NumPy/SciPy/PyTorch versions from
`python scripts/doctor.py --json`, and the file hash from `manifests/SHA256SUMS.txt` for any artifact
that differs. If a published value fails to reproduce, that is a defect in the release, not in your
environment, unless the tolerance table above explains it.
