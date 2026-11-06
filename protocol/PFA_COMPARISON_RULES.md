# PFA comparison rules (pre-registered)

These rules are fixed **before** any corrected H1 score exists. Changing them after
seeing E1 outcomes turns the analysis into exploratory work and must be reported as
such.

---

## 1. Estimands

For detector `d`, fold `f`, design PFA `p`:

| symbol | definition | reported with |
|---|---|---|
| `PFA_design` | `p` | — |
| `PFA_CAL,d` | fraction of CAL windows above `t_d` | threshold bootstrap CI |
| `PFA_TEST,d` | fraction of TEST-H0 windows above `t_d` | CP **and** cell-stratified bootstrap CI |
| `Pd_d` | fraction of paired TEST-H1 trials above `t_d` | cell-stratified bootstrap CI |
| `ΔPd_{d,e}` | `Pd_d − Pd_e` on identical trials | clustered CI, exact McNemar |
| `ΔPFA_{d,e}` | `PFA_TEST,d − PFA_TEST,e` | clustered CI |

`t_d` is estimated once per (fold, detector, p) by the frozen estimator
(`THRESHOLD_ESTIMATOR_SPECIFICATION.md`) and never re-estimated on TEST.

## 2. Primary analysis (confirmatory)

1. Detectors: Hybrid, ANMF-DL, AR(8), Notch. ANMF-unloaded, GLRT, CA-CFAR and OS-CFAR
   are reported as controls/diagnostics.
2. Operating point: `p = 1e-3`.
3. Each `Pd` is computed at that detector's frozen threshold; no matched-rate
   re-thresholding is applied to the primary table.
4. `ΔPFA` is reported next to every `ΔPd`. A detector with a higher realised false
   alarm rate is not credited with a "matched" comparison.
5. The primary comparison family is the 3 contrasts Hybrid−ANMF-DL, Hybrid−AR(8),
   Hybrid−Notch. Holm adjustment at family-wise α = 0.05 across those 3 exact
   McNemar tests; unadjusted p-values are also shown.
6. Direction: two-sided tests. A one-sided claim ("Hybrid better than X") may only
   be made after the two-sided test rejects and the sign agrees.

## 3. Secondary analyses (exploratory, labelled)

* `p = 1e-4`, with the extrapolated-threshold label and no confirmatory status.
* Common-FPR ROC: interpolate each detector's TEST-H1 score quantiles against its
  TEST-H0 score quantiles on a shared realised-FPR grid
  `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`. Interpolation is linear in
  `(log10 PFA, Pd)`. Report partial AUC over `[1e-4, 1e-2]` with bootstrap CI.
  Label: **"common-FPR ROC (evaluation only)"**.
* `SCR@Pd=0.5`: linear interpolation between the two bracketing grid points, with a
  bootstrap CI that carries the whole curve; report "not reached" if the curve does
  not cross 0.5 inside the grid.
* Per-acquisition Pd and per-site Pd, as descriptive tables.
* ANMF `clean` vs `operational` reference: the difference is an information-value
  estimate, not a detector comparison.

## 4. Multiplicity

| family | tests | correction |
|---|---|---|
| primary Pd contrasts | 3 | Holm, FWER 0.05 |
| realised-PFA contrasts | 3 | Holm, FWER 0.05 |
| ablation contrasts (E2) | 3 | Holm, FWER 0.05 |
| per-SCR grid | descriptive | no correction; CIs are pointwise and labelled as such |

## 5. Uncertainty

* Proportions: Clopper–Pearson (reported, labelled "iid-assumption").
* All inferential statements: cell-stratified bootstrap, B = 2,000, resampling cells
  within acquisition and acquisitions within fold; seed `20261029 + 1000·fold + detector`.
* Paired differences: same resampling of cells/trials, so pairing is preserved.
* Threshold uncertainty is propagated by recomputing `Pd` at each bootstrap
  threshold draw for the "threshold-uncertainty" sensitivity band; the primary CI
  conditions on the frozen threshold and is labelled accordingly.

## 6. Decision rules (fixed in advance)

| question | supported only if |
|---|---|
| **A. Hybrid > ANMF-DL** | `ΔPd > 0` with clustered 95 % CI excluding 0 at `p = 1e-3` in **both** folds, and the sign agrees in ≥ 6 of 8 TEST acquisitions |
| **B. physics loss helps** | E2 `ΔPd` CI excludes 0 under the same protocol; otherwise "inconclusive" |
| **C. warm start helps** | E2 cold-ICA contrast CI excludes 0; otherwise "inconclusive" |
| **D. robust across acquisitions** | per-acquisition signs agree in ≥ 12 of 16; with 16 clusters a stronger claim is not available |
| **E. practical value despite AR(8)** | a pre-specified non-Pd advantage (runtime, label-free operation, calibration stability) measured with CI, or an explicit statement that none survives |
| **F. theory survives** | Phase-2 corrections re-derived under the corrected protocol; otherwise the theoretical section is withdrawn |

Outcome vocabulary is fixed: **Supported**, **Inconclusive**, **Contradicted**,
**Not testable with available resources**. "Supported" never means "significant in
one fold only".

## 7. What is forbidden

* re-thresholding a detector on TEST-H1 or on TEST-H0 to improve its Pd;
* selecting the SCR grid, seeds or trial count after seeing Pd;
* dropping an acquisition because its Pd is low;
* reporting a pooled CI that ignores acquisition clustering;
* calling a 1e-4 result "verified".
