# Paper 1 reproducibility release — v1.0.0 (2026-11-06)

**Author:** Hamzeh Ghahramani  
**GitHub repository:** https://github.com/Hamzeh62/sea-clutter-hybrid-ica-reproducibility

## Related manuscript

*Hybrid Complex-Valued Deep Learning and Reference-Anchored ICA for Small-Target Detection in Sea Clutter: Detection and Calibration Analysis*, prepared for submission to *Signal, Image and Video Processing*. No manuscript acceptance, publication, or DOI is asserted.

## Included

- Hybrid complex-valued network and reference-anchored ICA implementation, detector baselines, preprocessing and calibrated-evaluation scripts.
- Frozen corrected 16-acquisition protocol and summary tables from Phases 5–7.
- Archived original-protocol Swerling power curves, ROC and ablation figures, explicitly separated from the corrected evaluation.
- Runnable unit tests and figure/table reproduction scripts.
- The two verified original model checkpoints as a **separate** direct-Zenodo file.
- A compact numerical results archive as a **separate** direct-Zenodo file.

## Excluded or not fully demonstrated

- Raw and near-lossless derived third-party IPIX radar data are not redistributed.
- The complete approximately 175 MB per-trial campaign results archive is **not included** in this particular release handoff; the smaller authentic compact-results archive does not replace full raw-score availability.
- Some historical figures are archived as the original vector images; their end-to-end reproduction from raw trials is not certified by the corrected-protocol test suite.
- No network retraining, full Monte Carlo rerun, or new scientific experiment was performed to create this repository release.

## Licenses

MIT for author-owned software (`LICENSE`). CC BY 4.0 for author-owned numerical results and released checkpoint bytes (`DATA_LICENSE.md`). Neither covers unlicensed IPIX recordings or unrelated third-party content. Licenses approved 2026-11-06.

## Integrity and publication status

The release staging passed the SHA-256 artifact verification, 31 unit tests with 3 raw-IPIX-dependent skips, and reproduction of 133/133 corrected-protocol published numerical values. This does not prove full reproduction of every historical plot or the availability of raw IPIX data. The remote repository was initially private and empty. No Zenodo record or DOI has been published as part of local packaging.
