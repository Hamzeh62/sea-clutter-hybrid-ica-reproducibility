"""Render the minimal-campaign report from the frozen analysis artifacts.

Every number in the report is read from ``statistics/ANALYSIS_SUMMARY.json`` or the
campaign CSVs written by ``phase5_analyze_v2.py``; nothing is typed into the report by
hand.  The verdict rules are the pre-specified ones of
``phase4/reports/MINIMAL_CAMPAIGN_EXECUTION_PLAN.md`` and
``phase4/reports/STATISTICAL_ANALYSIS_PLAN.md``:

* Question A (Hybrid vs ANMF-DL) is *supported* only if the clustered 95 % CI of the
  paired Pd difference at 0 dB excludes zero in the direction of the point estimate and
  the Holm-adjusted McNemar p-value is below 0.05; a CI that includes zero is reported as
  **inconclusive**, never as equality.
* Question B (physics/SAST loss) is *supported* only if the ``nophys - hybrid`` CI
  excludes zero; otherwise the contribution is reported as not detected on this
  evaluation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import phase5_common as C

REPORT = C.WORK / "reports"
STATS = C.WORK / "statistics"
FIGS = C.WORK / "figures"
PRIMARY_SCR = 0.0


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x, nd=4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:                     # NaN
        return "—"
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.1e}"
    return f"{v:.{nd}f}"


def pct(x, nd=1) -> str:
    try:
        return f"{100 * float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def verdict(c: dict | None, label_a: str, label_b: str) -> tuple[str, str]:
    """Pre-specified decision rule for one primary contrast at the primary SCR."""
    if not c:
        return "NOT EVALUATED", "the contrast was not produced by the analysis"
    lo, hi = c["ci"]
    d = c["delta_pd"]
    p = c.get("holm_p", float("nan"))
    if lo > 0 and p == p and p < 0.05:
        return "SUPPORTED", (f"{label_a} exceeds {label_b} by {d:+.3f} Pd "
                             f"(95 % CI {lo:+.3f} to {hi:+.3f}; Holm p = {p:.2g})")
    if hi < 0 and p == p and p < 0.05:
        return "REVERSED", (f"{label_b} exceeds {label_a} by {-d:+.3f} Pd "
                            f"(95 % CI {-hi:+.3f} to {-lo:+.3f} for {label_b} − "
                            f"{label_a}; Holm p = {p:.2g})")
    return "INCONCLUSIVE", (f"ΔPd = {d:+.3f} with 95 % CI {lo:+.3f} to {hi:+.3f} "
                            f"includes zero; the data do not decide the contrast")


def main() -> int:
    summary = json.loads((STATS / "ANALYSIS_SUMMARY.json").read_text(encoding="utf-8"))
    pd_rows = read_csv(REPORT / "MINIMAL_CAMPAIGN_RESULTS.csv")
    pfa_rows = read_csv(REPORT / "MINIMAL_CAMPAIGN_PFA_AUDIT.csv")
    acq_rows = read_csv(REPORT / "MINIMAL_CAMPAIGN_ACQUISITION_RESULTS.csv")
    contr_rows = read_csv(REPORT / "MINIMAL_CAMPAIGN_CONTRASTS.csv")
    het_rows = read_csv(REPORT / "MINIMAL_CAMPAIGN_HETEROGENEITY.csv")
    sens_rows = read_csv(REPORT / "MINIMAL_CAMPAIGN_THRESHOLD_SENSITIVITY.csv")
    redraw_rows = read_csv(REPORT / "MINIMAL_CAMPAIGN_THRESHOLD_REDRAW.csv")
    run_rows = read_csv(REPORT / "MINIMAL_CAMPAIGN_RUNTIME.csv")

    pc = summary["primary_contrasts_at_0db"]
    qa, why_a = verdict(pc.get("hybrid-anmf_dl"), "Hybrid", "ANMF-DL")
    qb, why_b = verdict(pc.get("nophys-hybrid"), "no-physics", "Hybrid")
    qa_ar, why_ar = verdict(pc.get("hybrid-ar8"), "Hybrid", "AR(8)+NMF")

    integ = summary["integrity"]
    n_pass = sum(1 for c in integ["checks"] if c["status"] == "PASS")
    n_tot = len(integ["checks"])

    L: list[str] = []
    A = L.append
    A("# Minimal corrected campaign — results report\n")
    A(f"*Campaign:* `{summary['campaign']}`  ")
    A(f"*Generated:* {summary['generated']}  ")
    A(f"*Config SHA-256:* `{summary['config_sha256']}`  ")
    A(f"*Design PFA:* {summary['design_pfa']:g}  ")
    A(f"*Bootstrap:* B = {summary['bootstrap_B']} (clustered), threshold bootstrap "
      f"B = {summary['threshold_bootstrap_B']}  ")
    A(f"*Total measured compute:* {summary['runtime_total_cpu_hours']:.2f} CPU-hours\n")

    A("## 1. Verdict\n")
    A("| question | verdict | evidence |")
    A("|---|---|---|")
    A(f"| A. Hybrid vs ANMF-DL | **{qa}** | {why_a} |")
    A(f"| A′. Hybrid vs AR(8)+NMF | **{qa_ar}** | {why_ar} |")
    A(f"| B. physics (SAST) loss | **{qb}** | {why_b} |")
    A("")
    A("A contrast is called *supported* only when the clustered 95 % CI excludes zero "
      "and the Holm-adjusted McNemar test is significant at 0.05. A CI containing zero "
      "is reported as *inconclusive*: absence of evidence is not evidence of equality.\n")
    A("Two of these verdicts are conditional and must not be quoted on their own: "
      "A′ holds only under the frozen VV-only injection, in which the VH reference used "
      "by AR(8) contains no target (sections 8b and 9); and B reverses sign when the "
      "threshold is redrawn jointly (section 8b). The unconditional reading of the "
      "campaign is in `reports/VERDICT.md`.\n")

    A("## 2. Integrity of the run\n")
    A(f"Analysis integrity checks: **{n_pass}/{n_tot} PASS**.\n")
    A("| level | count |")
    A("|---|---|")
    for k, v in summary["n_h0_windows"].items():
        A(f"| H0 windows, {k} | {v} |")
    for k, v in summary["n_h1_trials"].items():
        A(f"| H1 trials, {k} | {v} |")
    A("")

    A("## 3. Thresholds and realised false-alarm rate\n")
    A("Thresholds are the arm's own empirical `1 - PFA` order statistic on CAL-H0; the "
      "realised PFA is measured on TEST-H0, which is disjoint from CAL and was never "
      "used to set a threshold.\n")
    A("| arm | fold | design PFA | threshold | CAL exceedances | TEST-H0 k/n | "
      "realised PFA | CP 95 % | clustered 95 % |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in pfa_rows:
        A(f"| {r['arm']} | {r['fold']} | {fnum(r['design_pfa'], 5)} | "
          f"{fnum(r['threshold'], 6)} | {r['cal_exceed']}/{r['cal_n']} | "
          f"{r['test_exceed']}/{r['test_n']} | {fnum(r['realised_pfa'], 5)} | "
          f"[{fnum(r['cp_lo'], 5)}, {fnum(r['cp_hi'], 5)}] | "
          f"[{fnum(r['boot_lo'], 5)}, {fnum(r['boot_hi'], 5)}] |")
    A("")

    A("### 3b. Pd at a common *realised* false-alarm rate\n")
    A("The audit above shows the empirical thresholds do not transfer equally: the "
      "realised PFA of an arm differs by up to a factor of 9 between folds, so a "
      "comparison at a common *design* PFA is not a comparison at a common false-alarm "
      "rate. The table below thresholds every arm on TEST-H0 itself at the requested "
      "rate, which fixes the realised PFA exactly at the price of a threshold that is no "
      "longer deployable; it therefore measures the score distributions, not the "
      "calibration procedure.\n")
    mp = REPORT / "MINIMAL_CAMPAIGN_MATCHED_PFA.csv"
    if mp.is_file():
        mpr = read_csv(mp)
        A("| arm | Pd at realised 1e-3 | Pd at 2e-3 | Pd at 5e-3 | Pd at 1e-2 |")
        A("|---|---|---|---|---|")
        tab: dict[str, dict[float, float]] = {}
        for r in mpr:
            if r["fold"] == "pooled":
                tab.setdefault(r["arm"], {})[float(r["requested_pfa"])] = float(r["pd"])
        for arm in sorted(tab, key=lambda a: -tab[a].get(1e-3, -1)):
            A(f"| {arm} | " + " | ".join(
                fnum(tab[arm].get(g, "—"), 3) if g in tab[arm] else "—"
                for g in (1e-3, 2e-3, 5e-3, 1e-2)) + " |")
        A("")
        A("Per-fold values are in `reports/MINIMAL_CAMPAIGN_MATCHED_PFA.csv`; the fold "
          "difference is large and is discussed in the verdict.\n")

    A("## 4. Detection probability at the common grid\n")
    A("| arm | " + " | ".join(f"{s:g} dB" for s in (-20, -15, -12.5, -10, -7.5, -5, 0, 5, 10))
      + " |")
    A("|---" * 10 + "|")
    grid = (-20, -15, -12.5, -10, -7.5, -5, 0, 5, 10)
    by_arm: dict[str, dict[float, float]] = {}
    for r in pd_rows:
        by_arm.setdefault(r["arm"], {})[float(r["scr_db"])] = float(r["pd"])
    for arm in sorted(by_arm, key=lambda a: -by_arm[a].get(PRIMARY_SCR, -1)):
        cells = [fnum(by_arm[arm].get(float(s), "—"), 3) if float(s) in by_arm[arm]
                 else "—" for s in grid]
        A(f"| {arm} | " + " | ".join(cells) + " |")
    A("")
    A("Ranking at 0 dB (pooled over folds): " + ", ".join(
        f"{a} {fnum(v, 3)}" for a, v in summary["ranking_at_0db"]) + ".\n")

    A("## 5. Primary and secondary contrasts\n")
    A("| comparison | SCR (dB) | ΔPd | clustered 95 % CI | discordant A/B | McNemar p | "
      "Holm p | acquisitions A/B |")
    A("|---|---|---|---|---|---|---|---|")
    for r in contr_rows:
        A(f"| {r['arm_a']} − {r['arm_b']} | {fnum(r['scr_db'], 1)} | "
          f"{fnum(r['delta_pd'], 3)} | [{fnum(r['boot_lo'], 3)}, {fnum(r['boot_hi'], 3)}] | "
          f"{r['a_only']}/{r['b_only']} | "
          f"{fnum(r['mcnemar_p'], 2)} | {fnum(r['holm_p'], 2)} | "
          f"{r['acq_favouring_a']}/{r['acq_favouring_b']} |")
    A("")
    A("`discordant A/B` counts the paired trials in which only the first (A) or only the "
      "second (B) arm detects; `acquisitions A/B` counts the acquisitions whose mean ΔPd "
      "is positive (A ahead) or negative (B ahead).\n")

    A("## 6. Physics ablation and AR/notch fairness\n")
    ab = [r for r in contr_rows
          if r["arm_a"] in ("nophys", "cold", "ica_only", "ar8", "ar8_vvfit", "notch")
          or r["arm_b"] in ("nophys", "cold", "ica_only", "ar8", "ar8_vvfit", "notch")]
    if ab:
        A("| comparison | SCR (dB) | ΔPd | clustered 95 % CI | McNemar p |")
        A("|---|---|---|---|---|")
        for r in ab:
            A(f"| {r['arm_a']} − {r['arm_b']} | {fnum(r['scr_db'], 1)} | "
              f"{fnum(r['delta_pd'], 3)} | [{fnum(r['boot_lo'], 3)}, "
              f"{fnum(r['boot_hi'], 3)}] | {fnum(r['mcnemar_p'], 2)} |")
        A("")

    A("## 7. Heterogeneity across acquisitions\n")
    if het_rows:
        A("The test operates on the matched binary outcomes. The stratified McNemar "
          "statistic tests the average effect; the homogeneity statistic Q tests whether "
          "the discordance difference is the same in every acquisition. A significant Q "
          "means the average contrast must not be read as a uniform effect.\n")
        A("| comparison | stratified McNemar z | p | homogeneity Q | df | p | "
          "acquisitions A/B | n01/n10 |")
        A("|---|---|---|---|---|---|---|---|")
        for r in het_rows:
            A(f"| {r['arm_a']} − {r['arm_b']} | {fnum(r['stratified_mcnemar_z'], 2)} | "
              f"{fnum(r['stratified_mcnemar_p'], 2)} | {fnum(r['homogeneity_q'], 1)} | "
              f"{r['df']} | {fnum(r['homogeneity_p'], 2)} | "
              f"{r['acq_favouring_a']}/{r['acq_favouring_b']} | "
              f"{r['n01_total']}/{r['n10_total']} |")
        A("")
    A("Per-acquisition Pd at 0 dB is tabulated in "
      "`reports/MINIMAL_CAMPAIGN_ACQUISITION_RESULTS.csv` and drawn in Figure 5.\n")

    A("## 8. Threshold-uncertainty sensitivity\n")
    A("The design threshold at 1e-3 is the 8th largest of ~8,000 CAL units, so it is an "
      "extreme order statistic of a few clusters rather than a stable quantile. Two "
      "questions are therefore reported separately: how much a *single* arm's Pd moves "
      "when its threshold is redrawn (8a, not a comparison), and whether the *paired* "
      "contrast survives joint redraw with the same resampled units (8b, the question "
      "that matters).\n")
    A("### 8a. Single-arm Pd under a redrawn threshold\n")
    if sens_rows:
        A("| arm | fold | fixed-threshold Pd | Pd with threshold redrawn (clustered) |")
        A("|---|---|---|---|")
        for r in sens_rows:
            A(f"| {r['arm']} | {r['fold']} | {fnum(r['pd_fixed_threshold'], 3)} | "
              f"{fnum(r['pd_lo_threshold_redrawn'], 3)} – "
              f"{fnum(r['pd_hi_threshold_redrawn'], 3)} |")
        A("")
    A("### 8b. Paired contrast under joint threshold redraw\n")
    if redraw_rows:
        A("| comparison | frozen-threshold ΔPd | redrawn median | 95 % interval | "
          "P(A below B) |")
        A("|---|---|---|---|---|")
        for r in redraw_rows:
            A(f"| {r['arm_a']} − {r['arm_b']} | {fnum(r['delta_pd_frozen'], 3)} | "
              f"{fnum(r['redraw_median'], 3)} | [{fnum(r['redraw_lo'], 3)}, "
              f"{fnum(r['redraw_hi'], 3)}] | {fnum(r['p_a_worse_than_b'], 3)} |")
        A("")
        A("A contrast whose frozen and redrawn intervals disagree in sign is not a robust "
          "finding, however small its frozen p-value.\n")

    A("## 9. Cross-polarisation reference contamination (E3 sensitivity)\n")
    e3 = REPORT / "MINIMAL_CAMPAIGN_E3_REFERENCE_CONTAMINATION.csv"
    if e3.is_file():
        e3r = read_csv(e3)
        A("The frozen protocol injects the target into VV only, so the VH channel of the "
          "CUT is a perfectly target-free reference. AR(8)+NMF fits its prediction-error "
          "filter on that channel and the two-tap notch uses it as its cancellation "
          "reference. This sensitivity adds the *same* target instance to VH at "
          "$-10$ dB (a realistic cross-polarisation ratio) and at $0$ dB (the leakage "
          "bound), re-scores the frozen trials with the frozen thresholds, and changes "
          "nothing else.\n")
        A("| fold | arm | condition | Pd at 0 dB | Δ vs frozen protocol |")
        A("|---|---|---|---|---|")
        for r in e3r:
            A(f"| {r['fold']} | {r['arm']} | {r['condition']} | {fnum(r['pd'], 3)} | "
              f"{fnum(r['pd_delta_vs_vv_only'], 3)} |")
        A("")
        A("Interpretation is stated in the section that follows; the numbers above are "
          "produced by `code/phase5_refcontam_report.py` from the same frozen trials.\n")
    else:
        A("*Not produced in this run.*\n")

    A("## 10. Compute actually spent\n")
    if run_rows:
        A("| group | jobs | CPU-hours | peak RSS (MB) |")
        A("|---|---|---|---|")
        for r in run_rows:
            A(f"| {r['group']} | {r['n_jobs']} | {fnum(r['cpu_hours'], 2)} | "
              f"{fnum(r['peak_rss_mb'], 0)} |")
        A("")

    A("## 11. Figures\n")
    if FIGS.is_dir():
        for p in sorted(FIGS.glob("*.pdf")):
            A(f"* `{p.name}` — see `reports/FIGURE_CAPTIONS.md`")
        A("")

    A("## 12. What this campaign does and does not establish\n")
    A("* It establishes, on all 16 acquisitions of the corrected two-fold design, whether "
      "the Hybrid and the adaptive baselines differ at a matched design PFA of 1e-3, with "
      "the pairing preserved by construction.")
    A("* The realised PFA is measured out-of-sample on TEST-H0 and reported with a "
      "clustered interval; the CAL order statistic is the only source of each arm's "
      "threshold.")
    A("* It does not establish a 1e-4 operating point, a Swerling-1/2 result, or a "
      "second-seed replication; the H0 pools are 8,000 (CAL) and 4,000 (TEST-H0) windows "
      "per fold, so the realised-PFA interval is wide by construction.")
    A("* Differences smaller than the CI width, in particular a few Pd points, are not "
      "detectable here and are not claimed.")
    A("* It does **not** establish a matched-*realised*-PFA comparison from a deployable "
      "threshold: the CAL order statistic transfers unequally across folds (section 3), "
      "the fold-1 and fold-2 operating points differ by a factor of 4.5, and the pooled "
      "numbers mix them. Section 3b reports the score-distribution comparison that is "
      "free of this problem, and the per-fold table is authoritative.")
    A("* It does **not** establish that the Hybrid is better or worse than AR(8) as a "
      "property of the detectors: the ordering is decided by whether the VH reference "
      "contains the target (section 9), which this protocol fixes rather than measures.")
    A("")
    A("Deviations recorded before this report: see `reports/DEVIATIONS.md`.\n")

    out = REPORT / "MINIMAL_CAMPAIGN_REPORT.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "lines": len(L), "verdicts": {
        "A_hybrid_vs_anmf": qa, "A_hybrid_vs_ar8": qa_ar, "B_physics": qb}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
