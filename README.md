# Hybrid Complex-Valued Deep Learning and Reference-Anchored ICA for Small-Target Detection in Sea Clutter: Detection and Calibration Analysis

**Author:** Hamzeh Ghahramani  
**Affiliation:** Department of Electrical Engineering, Bandar Khamir Center, Bandar Lengeh Branch, Islamic Azad University, Hormozgan, Iran  
**Correspondence:** hamze.ghahramani@gmail.com · hamze.ghahramani@iau.ac.ir

This repository is the software and lightweight numerical-results companion to a manuscript **being prepared for submission** to *Signal, Image and Video Processing* (Springer Nature). The manuscript has **not yet been submitted, accepted, or published**. GitHub repository: https://github.com/Hamzeh62/sea-clutter-hybrid-ica-reproducibility (public). The archived software release **v1.0.0** is available at https://doi.org/10.5281/zenodo.24005344 (concept DOI for all software versions: https://doi.org/10.5281/zenodo.24005343). This DOI identifies the **software archive**, not the separately prepared model checkpoints and experimental-results archive.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24005344.svg)](https://doi.org/10.5281/zenodo.24005344)

**Citation:** Ghahramani, H. (2026). *Hybrid Complex-Valued Deep Learning and Reference-Anchored ICA for Small-Target Detection in Sea Clutter: Detection and Calibration Analysis — reproducibility software (v1.0.0)*. Zenodo. https://doi.org/10.5281/zenodo.24005344

**Experimental artifacts DOI:** https://doi.org/10.5281/zenodo.24005920 — trained checkpoints and the compact experimental-results archive (CC BY 4.0). The original complete ~175 MB results archive is not included.

## Scientific scope and historical versus corrected results

The final SIVP version retains the **original Swerling detection-power curves, low-PFA ROC, and architectural ablations**, and separately reports a **corrected 16-acquisition IPIX evaluation** of polarimetric-reference contamination, retrospective common-false-alarm comparisons, warm-start effects, and calibration transfer.

- **Historical original-protocol figures** are preserved as publication-quality vector PDFs in `figures/publication/main/` and `figures/publication/supplementary/`. Their source curves have **not** been recomputed from the later corrected protocol. Historical numerical score archives and some legacy plotting scripts were not supplied in the present public-release inputs; reproducing those curves *from raw trials* is **not yet validated by this staging package**.
- **Corrected evaluation outputs** and their code are in `results/phase5`–`phase7`, `pipeline/`, and `scripts/`. The existing Phase 9 validation (133 values, 34 tests, two figures, one bit-exact IPIX replay when data are present) refers to the former **two-figure Phase 8 article**. It must **not** be represented as end-to-end numerical validation of every table and figure in the restored manuscript.
- **Figure provenance:** see `manifests/FINAL_SIVP_FIGURE_PROVENANCE.csv` and `docs/RESTORED_MANUSCRIPT_REPRODUCIBILITY.md`.
- **Checkpoints and compact score archive:** verified original checkpoint bytes and the available compact results archive are published separately in Zenodo at https://doi.org/10.5281/zenodo.24005920. The complete ~175 MB campaign archive remains separate and is not included in this handoff; see `checkpoints/README.md`, `results/README.md` and `docs/DEPOSIT_STATUS.md`.
- **IPIX raw CDF / derived cubes:** excluded pending third-party redistribution rights. Users must obtain source data through an authorized channel.

The manuscript deliberately distinguishes nominal PFA from independently measured false-alarm rates and does not claim universal superiority over classical detectors or cross-acquisition constant-false-alarm guarantees.

## Current folder contents

`src/` contains detector implementations; `pipeline/` and `protocol/` contain corrected evaluation workflow and split/calibration logic; `results/` contains *lightweight* numerical outputs; `scripts/` provides checked command-line utilities; `tests/` contains release regression tests; `publication/` contains the final manuscript PDFs; `figures/publication/` carries exact vector figures from the restored manuscript.

## Licensing status

**Licenses approved and applied:** MIT for author-owned source code (`LICENSE`) and CC BY 4.0 for author-owned numerical results and checkpoints (`DATA_LICENSE.md`). Third-party IPIX recordings remain excluded.

## Funding, competing interests and AI assistance

No external funding. The author declares no competing interests. GLM 5.3 Flash assisted with design, code, experiments and drafting; DeepSeek V4.1 Flash assisted with code and experiment review; ChatGPT assisted with planning and final manuscript review. The author takes full responsibility for the research. The full disclosure is in the manuscript.

---

## Installation

Tested on Windows 11 with CPython 3.12.10, NumPy 2.5.2, SciPy 1.18.1, PyTorch 2.13.0+cpu,
Matplotlib 3.11.1. No GPU is used or required.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

`environment.yml` provides the same pinning for conda. A POSIX equivalent
(`python3 -m venv .venv && . .venv/bin/activate`) should work but was **not** exercised in this
release; only the Windows commands above were run.

Check what this checkout can do:

```bash
python scripts/doctor.py
```

It prints the resolved layout, the installed versions, and which of the three reproduction levels
are available.

## Data preparation

Level 1 needs nothing. Level 2 needs the 16 complex cubes:

```bash
python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache
python scripts/build_cache.py --cdf-dir /path/to/ipix --out data/ipix_cache --verify-only
```

`--verify-only` checks the cache against `metadata/target_cells.json` and lists any missing
acquisition. Alternatively point `PAPER1_CACHE` at a cache you already have.

## Checkpoints

Two weights are required, and only for Level 2 and the hybrid smoke test:

| file | role | SHA-256 |
|---|---|---|
| `best_s0.pt` | hybrid (M1→M2→M3→M4) | `a0a49db9c24814706fa89b38c8fc1ee8b306fc15cf6a611f259148d18a06ce53` |
| `best_s0_nophys.pt` | physics-free ablation | `2bcf10417a97f4d234853c97d43106331b33c0c0ab99f3e5b30ebdcd18411da0` |

Place them in `checkpoints/` (or set `PAPER1_CHECKPOINTS`). Both are deposited in the separate experimental-artifacts Zenodo dataset at https://doi.org/10.5281/zenodo.24005920. Verify them with:

```bash
python scripts/checkpoint_fixture.py --verify --checkpoints checkpoints
```

This checks the file hash, the parameter digest and a deterministic forward pass against the frozen
fixture. See `checkpoints/README.md`.

## Quick start (Level 1, no data, no weights)

```bash
python scripts/verify_artifacts.py          # frozen inputs, config hash, file manifest
python scripts/reproduce_tables.py          # every published number, from results/
python scripts/reproduce_figures.py --check # regenerate both figures, compare pixel-wise
```

Expected output, as measured in this release:

```
133/133 published values reproduced (base tolerance 0.0006); 0 failing
fig1_mechanism: mean |delta pixel| = 0.0000 / 255  [PASS]
fig2_calibration: mean |delta pixel| = 0.0000 / 255  [PASS]
ARTIFACT INTEGRITY VERIFIED
```

`reproduce_tables.py` compares released tables against `manifests/PUBLISHED_VALUES.json`, which was
transcribed **from the manuscript**, not from the tables. A failure therefore means the repository
does not support a published number.

## Code tests

```bash
python -m pytest -q
```

34 tests: acquisition-ID canonicalisation, target/guard exclusion, block layout, threshold
estimator, bootstrap pairing, McNemar exactness, Clopper–Pearson, and the ANMF sampler-index
regression. The three acquisition-table tests skip unless `PAPER1_CACHE` (or `data/ipix_cache`)
holds the 16 cubes; with the cache present all 34 run.

A synthetic end-to-end smoke test (no IPIX data required):

```bash
python scripts/smoke_test.py --checkpoint checkpoints/best_s0.pt
```

It checks that every detector path executes, that ANMF responds to a target, and that an AR(8)
predictor fitted on a target-bearing reference cancels the target. It reproduces **no published
number** and is reported that way in the validation report.

## Level 2 — re-run a released campaign job

This rebuilds the calibration preparation from the cubes and re-runs one scoring job, then compares
every array with the released artifact:

```bash
python scripts/validate_against_released_scores.py ^
  --released /path/to/results/runs/minimal_v1_20261029 ^
  --job "h1|fold1|hybrid|c0"
```

Measured in this release: the preparation matches exactly (max |Δ| = 0.0 on the complex
`Chat`/`Cinv`/`CinvP`), and all score arrays match with max |Δ| = 0.0 at one BLAS/torch thread. Two
cores and about 2 minutes are enough for this job. Thread count matters: pin
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `PHASE5_TORCH_THREADS=1`, which
the script does for you.

## Level 3 — training

`src/proposed/train.py` and `configs/default.yaml` are the training implementation and its
configuration, unmodified. Training-data generation, architecture, losses, seeds and schedule are
documented in the manuscript's Methods and in the config. **Training was not re-executed in this
release**, so Level 3 is "code provided, not validated here": no claim of exact retraining
reproducibility is made. The measured cost of the evaluation campaign (18.6 CPU-hours over 398
jobs, peak RSS 630 MB, ANMF accounting for 15.0 h) is a lower bound on nothing but evaluation.

## Reproduction levels and their status

| level | needs | status |
|---|---|---|
| L1 tables and figures | nothing | **validated**, 133/133 values, 2/2 figures pixel-identical |
| L2 detector evaluation | cubes + weights | **validated** for one job (`h1|fold1|hybrid|c0`), bit-exact |
| L3 training | cubes + weights + recordings | **code only**, not executed in this release |

The full per-artifact status is in `docs/REPRODUCIBILITY_VALIDATION_REPORT.md`; the per-claim
classification is in `manifests/REPRODUCIBILITY_COVERAGE_MATRIX.csv`.

## Determinism and tolerances

* Released arrays are compared with `max|Δ| ≤ 1e-9`; the measured difference was 0.0.
* Published values are compared with `6e-4` absolute, i.e. half a unit in the third decimal, which
  is the precision at which the manuscript prints them.
* The checkpoint fixture uses exact hashes for the bytes and the parameter digest, and `1e-6` on the
  statistic because a different PyTorch build may reassociate floating-point reductions.
* No random seed is drawn at evaluation time; the trial design and calibration subsamples are frozen
  in `protocol/` and `configs/campaign.json`.

## Citation

See `CITATION.cff`. Until the article is published, cite the archived software record with the DOI
https://doi.org/10.5281/zenodo.24005344 (software release v1.0.0); the separate checkpoints/compact-results dataset DOI is https://doi.org/10.5281/zenodo.24005920.

## License

**Status: draft, author approval required.** The code is the author's own work; no license has been
granted yet. `LICENSE` records the decision that is pending and must not be read as a grant of
rights. The IPIX data keeps its own terms and is not covered by any license here.

## Archival record

`docs/RELEASE_NOTES.md` lists what is archived, where, and with which version. The repository and
the archive share one version string, `v1.0.0`, and one set of hashes in `manifests/SHA256SUMS.txt`.

### Contact

Corresponding author: Hamzeh Ghahramani — hamze.ghahramani@gmail.com; confirmed university email: hamze.ghahramani@iau.ac.ir.


## Licensing and publication status (2026-11-06)

- **MIT** applies to original author-owned software (`LICENSE`).
- **CC BY 4.0** applies to original author-owned results and released checkpoints (`DATA_LICENSE.md`).
- Third-party raw IPIX recordings and near-lossless derivative cubes are **not distributed or licensed**.
- GitHub has been published and the v1.0.0 source release archived under https://doi.org/10.5281/zenodo.24005344. A separate research-artifacts dataset is available at https://doi.org/10.5281/zenodo.24005920 with the two checkpoint/compact-results ZIPs. Neither DOI contains the full ~175 MB score archive or raw IPIX recordings.
- This release contains archived historical-protocol figures as well as corrected-evaluation results. The two protocols are distinct, and the existing coverage limitations remain documented.
