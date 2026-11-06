# Released results

These files are the numerical evidence for the manuscript. They are released so that an independent
reader can check a claim without obtaining the IPIX recordings.

| directory | contents | used by |
|---|---|---|
| `phase5/` | campaign result tables: per-condition detection probabilities, contrasts, ablations, acquisition-level results, realised-PFA audit, heterogeneity, threshold sensitivity, matched-PFA analysis, runtime, reference-contamination E3 | Tables 2-6, Figure 2 |
| `phase6/` | polarimetric channel and target tables, calibration table, warm-start table | Table 1 |
| `phase7/` | V1 polarimetric results and contrasts, V2 calibration, V3 reference controls, corrected polarimetry | Tables 1-6, Figures 1-2 |
| `phase7/data/` | the Phase 7 per-fold JSON artifacts (V2 detail, V3 folds, warm-start contrasts, reconciliation) | Table 5, Table 6 |

Each table's mapping to a source file, a script and a tolerance is in
`../manifests/FIGURE_TABLE_REPRODUCTION_MATRIX.csv`. Every published value, and the exact key it is
read from, is in `../manifests/PUBLISHED_VALUES.json`.

`scripts/reproduce_tables.py` recomputes all 133 published values from these files; the measured
result is 133/133 with a 6e-4 tolerance.

The **score arrays** behind these tables (398 jobs, ~20 MB compressed) are not in this directory.
They are in the archival record and are needed only for Level 2 or for a different analysis of the
same trials. `MANIFEST` in each archive part lists them with hashes.

## Provenance

Every file here was copied byte-for-byte from the revision workspace and is hashed in
`../manifests/SHA256SUMS.txt`. `scripts/verify_artifacts.py` re-checks the whole tree.


**License:** Original author-owned numerical result artifacts are released under CC BY 4.0; see `../DATA_LICENSE.md`. The compact results archive is a separate direct-Zenodo deposit file.
