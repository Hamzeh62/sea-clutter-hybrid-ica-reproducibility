> **Version scope:** The validation below pertains to the earlier Phase 8 two-figure manuscript. The restored SIVP article additionally retains original-protocol Swerling/ROC/Ablation evidence and must not be declared fully re-generated from this lightweight archive. See `RESTORED_MANUSCRIPT_REPRODUCIBILITY.md`.

# Reproducibility validation report

**Package:** Paper 1 reproducibility repository, `v1.0.0`
**Validated:** 2026-11-01
**Verdict:** Level 1 fully validated; Level 2 validated end to end for one released job; Level 3
provided but not executed.

Status vocabulary: **PASS** (executed, expected outcome), **PARTIAL** (executed with a stated
limitation), **NOT TESTED**, **BLOCKED** (could not be executed here, reason given).

---

## 1. Environments used

| environment | Python | NumPy | SciPy | PyTorch | purpose |
|---|---|---|---|---|---|
| research venv (origin of the results) | 3.12.10 | 2.5.2 | 1.18.1 | 2.13.0+cpu | in-place validation, Level 2, checkpoint fixture |
| clean venv (created for this report) | 3.12.10 | 2.5.2 | 1.18.1 | not installed | clean-room Level-1 validation |
| host | Windows 11 (10.0.22631), AMD64, 4 logical cores | | | | |

The clean venv was created with `python -m venv` and populated from `requirements.txt`
(`numpy`, `scipy`, `matplotlib`, `PyYAML`, `pytest`, `Pillow` installed successfully). Installing
`torch==2.13.0` in it failed twice with `ReadTimeoutError` on `files.pythonhosted.org` and was
abandoned after 10 minutes; torch-dependent checks were therefore run in a clean **copy** of the
repository using the research venv. This is a network limitation of the validation host, not a
package defect, and it is why the torch-dependent rows say PARTIAL for installation but PASS for
execution.

## 2. Results

| # | check | command | result | status |
|---|---|---|---|---|
| 1 | frozen protocol inputs, config hash, file manifest | `python scripts/verify_artifacts.py` | all seven frozen inputs verify (six byte-identical, `acquisition_ids.py` via its pristine copy); config re-hash matches; every released file listed and matched | **PASS** |
| 2 | published values recomputed from released tables | `python scripts/reproduce_tables.py` | 133/133 (T1 25, T2 24, T3 24, T4 24, T5 24, T6 12); 0 failing | **PASS** |
| 3 | figures regenerated from released tables | `python scripts/reproduce_figures.py --check` | fig1 and fig2, mean absolute pixel difference 0.0000/255 each | **PASS** |
| 4 | release test suite, cache present | `python -m pytest -q` | 34 passed | **PASS** |
| 5 | release test suite, no cache | `python -m pytest -q` | 31 passed, 3 skipped (acquisition-table tests, by design) | **PASS** |
| 6 | checkpoint byte identity, parameter digest, deterministic forward pass | `python scripts/checkpoint_fixture.py --verify` | both files: file OK, params OK, statistic difference 0.00e+00 | **PASS** |
| 7 | synthetic end-to-end smoke test | `python scripts/smoke_test.py --checkpoint ...` | all four detector paths finite; ANMF rises with a +20 dB target; contaminated-reference AR(8) statistic falls to 0.050x; determinism holds | **PASS** |
| 8 | Level-2 re-execution, `h1|fold1|hybrid|c0` | `python scripts/validate_against_released_scores.py ...` | prep rebuilt from cubes matches exactly (complex `Chat`/`Cinv`/`CinvP` max abs diff 0.0); all score arrays EXACT; `arm_hybrid` max abs diff 0.0, rel 0.0 | **PASS** |
| 9 | example | `python examples/quickstart.py --checkpoint ...` | statistic 0.461222, identical to the frozen fixture; warm start, 51 ICA sweeps | **PASS** |
| 10 | clean-room Level 1 (fresh venv, extracted copy, no shared state) | as rows 1–5 in `/tmp/p1clean.../repo` | artifacts verified; 133/133 values; both figures 0.0; 31 passed/3 skipped without cache, 34 passed with cache | **PASS** |
| 11 | clean-room torch paths (extracted copy, research venv) | checkpoint fixture; smoke test | `CHECKPOINT FIXTURE VERIFIED`; `SMOKE TEST PASSED` | **PARTIAL** (installation of torch in the fresh venv blocked by network; execution passed) |
| 12 | Level 3 training | — | `src/proposed/train.py` + `configs/default.yaml` provided; not executed | **NOT TESTED** |
| 13 | full campaign re-execution (all 398 jobs) | — | not attempted; one job validated end to end | **NOT TESTED** |
| 14 | macOS / Linux execution | — | not attempted; POSIX commands are labelled unverified in the docs | **NOT TESTED** |

## 3. What the numbers mean

* **133/133** is a test against the *paper*, not against itself: `manifests/PUBLISHED_VALUES.json`
  was transcribed from the compiled manuscript, and the script reads the released tables. A failure
  would mean the repository cannot support a published number.
* **0.0000 pixel difference** means the plotting code and the input tables are the ones that produced
  the published vector figures. It does not depend on the PDF being byte-identical, which would
  require an identical TeX installation.
* **Level-2 EXACT** means the released calibration preparation and scores are reproducible from the
  cube cache and the checkpoint, not merely archived. It was measured at one BLAS thread; with
  multiple threads the hybrid statistic moves by ~1e-6 relative because float32 reductions are
  reassociated, which is why the script pins threads and reports the setting.
* **34 tests** cover the specific defects the revision found: the silent-empty exclusion set, the
  ANMF window/sample-offset confusion, threshold order-statistic choice, CAL/TEST separation,
  acquisition-ID canonicalisation and paired bootstrap alignment.

## 4. Defects found and fixed while preparing this release

These were found by this validation, not inherited from the science:

1. **The release test suite had four wrong tests** (mismatched `mcnemar` return order, a
   Clopper–Pearson bound 8e-5 above the true lower limit, an off-by-one block range, and two
   bootstrap tests that bypassed `cell_table`). The tests were corrected; the released code was not
   changed, because the code was right.
2. **`validate_against_released_scores.py` normalised the chunk filename wrongly** (`c0` vs `c00`)
   and its comparison cast complex arrays to float, which would have hidden a wrong imaginary part
   in the ICA matrices. Both fixed before the Level-2 result above was recorded.
3. **`pipeline/phase5_preflight.py` hardcoded two backup drives** that exist only on the author's
   workstation and would abort in any published checkout. Rewritten to take an optional
   `PAPER1_PROTECTED_BACKUPS` list plus the paths that do exist here.
4. **`__pycache__` directories** carrying absolute build paths were being staged into the tree. The
   build now purges them, and they are excluded from the archives.

## 5. Known limitations of this validation

* Only one of 398 jobs was re-executed end to end. That job exercises the hybrid path, the
  calibration preparation and the paired trial set, but it is not a full campaign replay.
* The synthetic smoke test proves the code runs and behaves qualitatively; it reproduces no
  published number and is not presented as evidence that the IPIX evaluation reproduces.
* `torch` could not be installed in the fresh venv on this host (network timeouts), so the clean
  room is complete for Level 1 and partial for the torch-dependent paths.
* The three acquisition-table tests skip without the IPIX cache; that is a deliberate, reported skip,
  not a hidden pass.
* Training (Level 3) was not run, so no claim of retraining reproducibility is made anywhere.

## 6. Sign-off

The package supports every published number and both published figures from its own released
artifacts, and re-executes a released detector job bit-exactly given the third-party data and the
released weights. It does not support a claim of full end-to-end reproduction of the campaign, and
this report does not make one.
