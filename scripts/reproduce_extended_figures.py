"""Phase 7 figures.  Every panel reads a result CSV/JSON; nothing is typed in by hand.

fig1  Pd against the cross-polarised target contrast, deployment operating point
fig2  paired Hybrid contrasts at a matched operating point, with acquisition-clustered CIs
fig3  relative-phase dependence (AR(8) and Notch)
fig4  calibration transfer: realised / design Pfa for the four rules
fig5  reference-definition variants of the AR(8) statistic
fig6  warm start and ablations, deployment versus matched operating point
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORK7 = PROJECT / "results" / "phase7"      # released Phase 7 data (read only)
FIG = PROJECT / "figures" / "extended"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True,
    "grid.linewidth": 0.4, "grid.alpha": 0.35,
})
COL = {"hybrid": "#1f4e79", "ar8": "#b03a2e", "notch": "#1e8449",
       "anmf_dl": "#7d3c98"}
LBL = {"hybrid": "Hybrid", "ar8": "AR(8)+NMF", "notch": "Notch (2-tap)",
       "anmf_dl": "ANMF-DL"}
MK = {"hybrid": "o", "ar8": "s", "notch": "^", "anmf_dl": "D"}


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[fig] {name}")


XPOS = {("-inf",): 0, ("-20",): 1, ("-10",): 2, ("0",): 3}
XLAB = ["absent\n(−∞)", "−20", "−10", "0"]


def xof(ds: str) -> int:
    return XPOS[(ds.strip(),)]


# ------------------------------------------------------------------------ fig 1
def fig1() -> None:
    dep = rows(WORK7 / "PHASE7_V1_POLARIMETRIC_RESULTS.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, fold in zip(axes, ("fold1", "fold2")):
        for arm in ("hybrid", "ar8", "notch"):
            pts = {}
            for r in dep:
                if r["fold"] != fold or r["arm"] != arm:
                    continue
                pts[xof(r["dSCR_dB"])] = float(r["pd"])
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], marker=MK[arm], color=COL[arm],
                    label=LBL[arm], clip_on=False)
        # ANMF-DL never reads VH: its operating point is exact at every dSCR
        anmf = [float(r["pd"]) for r in dep
                if r["fold"] == fold and r["arm"] == "anmf_dl"]
        if anmf:
            ax.axhline(anmf[0], color=COL["anmf_dl"], linestyle=(0, (4, 2)),
                       linewidth=1.0,
                       label="ANMF-DL (VH-independent)")
        ax.set_xticks(range(4))
        ax.set_xticklabels(XLAB)
        ax.set_xlim(-0.35, 3.35)
        ax.set_ylim(0, 1)
        ax.set_xlabel("ΔSCR$_{VH}$  (dB)")
        ax.set_title(f"{'Fold 1' if fold == 'fold1' else 'Fold 2'}"
                     "  (frozen CAL threshold)", fontsize=8)
    axes[0].set_ylabel("$P_d$ at SCR$_{VV}$ = 0 dB")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, "fig1_pd_vs_dscr")


# ------------------------------------------------------------------------ fig 2
def fig2() -> None:
    con = [r for r in rows(WORK7 / "PHASE7_V1_CONTRASTS.csv")
           if r["mode"] == "matched_0.005"]
    order = (("hybrid-ar8", "Hybrid − AR(8)", "#b03a2e"),
             ("hybrid-anmf_dl", "Hybrid − ANMF-DL", "#7d3c98"),
             ("hybrid-notch", "Hybrid − Notch", "#1e8449"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    w = 0.26
    for ax, fold in zip(axes, ("fold1", "fold2")):
        for i, (key, lab, col) in enumerate(order):
            ds, dd, lo, hi = [], [], [], []
            for r in con:
                if r["fold"] != fold or r["contrast"] != key:
                    continue
                ds.append(xof(r["dSCR_dB"]))
                dd.append(float(r["delta_pd"]))
                lo.append(float(r["delta_pd"]) - float(r["ci_lo"]))
                hi.append(float(r["ci_hi"]) - float(r["delta_pd"]))
            off = (i - 1) * w
            ax.bar([d + off for d in ds], dd, w, color=col, label=lab,
                   yerr=[lo, hi], capsize=2, error_kw={"linewidth": 0.7})
        ax.axhline(0, color="k", linewidth=0.7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(XLAB)
        ax.set_xlabel("ΔSCR$_{VH}$  (dB)")
        ax.set_title("Fold 1" if fold == "fold1" else "Fold 2", fontsize=8)
    axes[0].set_ylabel("Δ$P_d$ (matched $P_{fa}$ = 5×10$^{-3}$)")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "fig2_contrast_vs_dscr")


# ------------------------------------------------------------------------ fig 3
def fig3() -> None:
    dep = rows(WORK7 / "PHASE7_V1_POLARIMETRIC_RESULTS.csv")
    cases = (("dSCR0__phi0", "0 dB, φ=0"), ("dSCR0__randphi", "0 dB, random φ"),
             ("dSCR-10__phi90", "−10 dB, φ=90°"), ("dSCR-10__phi180", "−10 dB, φ=180°"),
             ("dSCR-10__randphi", "−10 dB, random φ"))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    w = 0.2
    for i, arm in enumerate(("ar8", "notch")):
        for j, fold in enumerate(("fold1", "fold2")):
            vals = []
            for tag, _ in cases:
                v = [float(r["pd"]) for r in dep
                     if r["condition"] == tag and r["arm"] == arm
                     and r["fold"] == fold]
                vals.append(v[0] if v else np.nan)
            ax.bar([x + (2 * i + j - 1.5) * w for x in range(len(cases))], vals, w,
                   color=COL[arm], label=f"{LBL[arm].split(' ')[0]} F{fold[-1]}",
                   hatch=("" if j == 0 else "////"), edgecolor="white",
                   linewidth=0.3)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[1] for c in cases], rotation=20, ha="right")
    ax.set_ylabel("$P_d$ (deployment threshold)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "fig3_phase_dependence")


# ------------------------------------------------------------------------ fig 4
def fig4() -> None:
    v2 = rows(WORK7 / "PHASE7_V2_CALIBRATION_RESULTS.csv")
    rules = ("A", "B", "C", "D")
    RLAB = {"A": "A pooled", "B": "B acquisition-\nstratified",
            "C": "C robust\n(CAL-selected)", "D": "D hierarchical\n(CAL-selected)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, fold in zip(axes, ("fold1", "fold2")):
        w = 0.2
        for i, arm in enumerate(("hybrid", "ar8", "notch", "anmf_dl")):
            vals, worst = [], []
            for rule in rules:
                sel = [r for r in v2 if r["fold"] == fold and r["arm"] == arm
                       and r["rule"] == rule]
                vals.append(float(sel[0]["realised_over_design"]) if sel else np.nan)
                worst.append(float(sel[0]["worst_acq_over_design"]) if sel else np.nan)
            x = [j + (i - 1.5) * w for j in range(len(rules))]
            ax.bar(x, vals, w, color=COL[arm], label=LBL[arm])
            ax.plot(x, worst, linestyle="none", marker="v", color=COL[arm],
                    markersize=3, markeredgecolor="k", markeredgewidth=0.4)
        ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 120)
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels([RLAB[r] for r in rules], fontsize=6.5)
        ax.set_title("Fold 1" if fold == "fold1" else "Fold 2", fontsize=8)
    axes[0].set_ylabel("realised $P_{fa}$ / design $P_{fa}$")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.06, "bars: pooled realised rate;  ▼: worst single acquisition "
             "(8 acquisitions per fold, 4000 TEST-H0 units)",
             ha="center", fontsize=6.5)
    save(fig, "fig4_calibration_transfer")


# ------------------------------------------------------------------------ fig 5
def fig5() -> None:
    order = ("deployed_g0", "deployed_g1", "target_channel", "delay1",
             "temporal_oracle", "delay512")
    L = {"deployed_g0": "clean G0\n(frozen)", "deployed_g1": "VH\ncontaminated",
         "target_channel": "same-channel\nfit", "delay1": "VV shift\n1 sample",
         "temporal_oracle": "VH +128\nCPIs", "delay512": "VV previous\nCPI"}
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    w = 0.36
    for i, fold in enumerate(("fold1", "fold2")):
        d = json.loads((WORK7 / f"data/PHASE7_V3_{fold}.json").read_text())
        vals = [d["variants"][k] for k in order]
        ax.bar([j + (i - 0.5) * w for j in range(len(order))], vals, w,
               color=("#1f4e79" if fold == "fold1" else "#5b9bd5"),
               label="Fold 1" if fold == "fold1" else "Fold 2")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([L[k] for k in order], rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("$P_d$ at SCR = 0 dB")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    save(fig, "fig5_reference_variants")


# ------------------------------------------------------------------------ fig 6
def fig6() -> None:
    d = json.loads((WORK7 / "data/PHASE7_WARMSTART.json").read_text())
    pairs = (("hybrid-cold", "vs cold start"), ("hybrid-ica", "vs ICA-only"),
             ("hybrid-nophys", "vs no physics prior"))
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    styles = (("fold1", "deployment", "#1f4e79", "F1 deploy"),
              ("fold1", "matched_0.005", "#8faadc", "F1 matched"),
              ("fold2", "deployment", "#c00000", "F2 deploy"),
              ("fold2", "matched_0.005", "#f4b183", "F2 matched"))
    w = 0.2
    for k, (fold, mode, col, lab) in enumerate(styles):
        xs, ds, err = [], [], [[], []]
        for i, (key, _p) in enumerate(pairs):
            c = d["contrasts"][f"{key}|{fold}"][mode]
            xs.append(i * 1.6 + (k - 1.5) * w)
            ds.append(c["delta_pd"])
            err[0].append(c["delta_pd"] - c["ci_lo"])
            err[1].append(c["ci_hi"] - c["delta_pd"])
        ax.bar(xs, ds, w, color=col, label=lab, yerr=err, capsize=1.5,
               error_kw={"linewidth": 0.5})
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xticks([i * 1.6 for i in range(len(pairs))])
    ax.set_xticklabels([p[1] for p in pairs], fontsize=7)
    ax.set_ylabel("Δ$P_d$ of Hybrid")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "fig6_warmstart")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
