# Threshold estimator specification (frozen)

Reference implementation: `phase4/patched_code/thresholds_frozen.py`.
Every number in a Phase-5 table must be reproducible from a saved CAL score array
plus this specification. Nothing here depends on TEST data.

---

## 1. Inputs

| Input | Source | Constraint |
|---|---|---|
| CAL score vector `s` | one detector, one fold, all CAL windows | length exactly `n_cal` of the manifest; all finite |
| design PFA `p` | 1e-3 primary, 1e-4 exploratory | fixed before scoring |
| tail model | exponential tail above the 99.5 % quantile | the only permitted extrapolator |

The CAL score vector must be written to disk **before** the threshold is computed,
with the SHA-256 of the array and the manifest row range. A missing value is a
hard error; the vector is never filtered, clipped or winsorised.

## 2. Decision rule

```
n = len(s)
if n * p >= 10:
    t = quantile(s, 1 - p, method="linear")      # Hyndman-Fan type 7
    method = "empirical"
    k_lo, k_hi = floor((n-1)*(1-p)), ceil((n-1)*(1-p))
    bracket = (sorted_s[k_lo], sorted_s[k_hi])
else:
    t = extrapolate(s, p)
    method = "extrapolated"
```

`extrapolate` is the existing deployed rule, reproduced verbatim for continuity:

```
t0 = quantile(s, 0.995)
e  = s[s > t0]
lam = len(e) / (sum(e) - len(e)*t0)          # exponential rate of the tail
c  = len(e) / len(s)
t  = t0 + log(c / p) / lam
```

The extrapolator is a **model**, not a measurement. Any row using it must carry
the `extrapolated` label in the table and the text.

## 3. Determinism

* NumPy version and BLAS thread count are recorded in the run manifest.
* Sorting is stable; ties are resolved by the type-7 interpolation, never by index
  order.
* The estimator consumes no random numbers. The only randomness in calibration is
  the cluster bootstrap used for uncertainty, which uses
  `default_rng(SEED_BOOT + 1000*fold_index + detector_index)` with `SEED_BOOT = 20261029`.
* Re-running the estimator on the same array must return a bit-identical float;
  the pilot asserts this.

## 4. Score conventions

| Detector | Raw statistic | Orientation | Notes |
|---|---|---|---|
| Hybrid | separated-domain NMF/ICA statistic | larger = target | network + anchored CICA, `C_sep` from CAL |
| ANMF loaded | `max_f |p^H C^-1 y|^2 / (p^H C^-1 p · y^H C^-1 y)` | larger | diagonal loading `1e-10·tr(C)/Np` (deployed) plus the `dl0.001` variant |
| ANMF unloaded | same, `load_jitter = 1e-10` | larger | diagnostic |
| GLRT | `max_f |p^H C^-1 y|^2 / (p^H C^-1 p)` | larger | diagnostic |
| AR(8) + NMF | NMF statistic of the VV prediction-error residual | larger | AR fit on VH of the same CPI |
| Two-tap notch | NMF statistic of `w^H x`, `w ∝ [1, −ρ̂]` | larger | `ρ̂` from the CUT's primary channel |
| CA-CFAR | cell-average threshold ratio | larger | same-CPI training set, guard cells |
| OS-CFAR | ordered-statistic threshold ratio | larger | `k = ceil(0.75·N_train)` |

Any detector whose natural statistic is a *p*-value or a distance must be negated
before calibration; the sign is recorded in the manifest.

## 5. Thresholds are per (detector, fold, PFA)

No threshold is shared across detectors, folds or operating points. A pooled
threshold across folds is **not** a deployed quantity and may appear only in the
E4 transfer analysis, labelled as such.

## 6. Required artifacts per calibration

```
runs/<run_id>/cal_scores/<fold>__<detector>.npy
runs/<run_id>/cal_scores/<fold>__<detector>.sha256
runs/<run_id>/thresholds/<fold>__<detector>__p<pfa>.json
runs/<run_id>/thresholds_manifest.sha256
```

The threshold JSON contains: value, method, `n`, bracketing order statistics,
design PFA, calibration exceedance, cluster-bootstrap CI, seed, code hash, and the
manifest hash of the CAL windows used.
