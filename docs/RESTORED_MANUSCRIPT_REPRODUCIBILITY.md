# Reproducibility scope of the restored SIVP manuscript

Article: **Hybrid Complex-Valued Deep Learning and Reference-Anchored ICA for Small-Target Detection in Sea Clutter: Detection and Calibration Analysis**

## Which claims can be reproduced now?

1. The published Phase 7 corrected tables can be recomputed from included CSV/JSON files with the tested `scripts/reproduce_tables.py` command.
2. The previously published two Phase 8 figures can be regenerated using `scripts/reproduce_figures.py`; **these are not the complete figure set of the restored SIVP article**.
3. The restored 6-figure main text and supplementary figure set are preserved byte-identically as vector figures. The historical-protocol curves should not be described as independent repeats of Phase 7.
4. A full detector rerun needs the third-party IPIX data, and checkpoint archives, neither of which is included in this package.
5. New independent reproduction of the historical 17-point Swerling curves and legacy convergence traces is **not established** by the files currently available here; the original paper's vector curves are preserved unchanged.

These distinctions prevent old and new calibration thresholds from being represented as the same protocol.

Data must not be invented or numerically re-interpolated to fill gaps.
