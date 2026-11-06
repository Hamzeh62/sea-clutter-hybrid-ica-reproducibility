"""Vector figures for the minimal campaign, from the frozen score arrays only.

Every panel is written as PDF (and a PNG preview) plus a JSON record of the data
plotted, so a reader can regenerate or check a figure without re-running a detector.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import phase5_common as C  # noqa: E402
from phase5_analyze_v2 import (ARM_GROUP, DESIGN_PFA, cell_table, clopper_pearson,  # noqa: E402
                               load_scores, threshold)

FIGS = C.WORK / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "figure.dpi": 160, "savefig.bbox": "tight",
})
STYLE = {
    "hybrid": dict(color="#1f4e79", marker="o"),
    "anmf_dl": dict(color="#c00000", marker="s"),
    "anmf_unloaded": dict(color="#e69138", marker="^"),
    "glrt": dict(color="#7f6000", marker="v"),
    "ar8": dict(color="#2e7d32", marker="D"),
    "notch": dict(color="#6a1b9a", marker="P"),
    "ca": dict(color="#555555", marker="x"),
    "os": dict(color="#999999", marker="+"),
    "nophys": dict(color="#0b7a75", marker=">"),
    "cold": dict(color="#b03a2e", marker="<"),
    "ica_only": dict(color="#a0522d", marker="*"),
}
LABEL = {
    "hybrid": "Hybrid (proposed)", "anmf_dl": "ANMF-DL", "anmf_unloaded": "ANMF unloaded",
    "glrt": "GLRT (un-normalised)", "ar8": "AR(8)", "notch": "Notch (2-tap)",
    "ca": "CA-CFAR", "os": "OS-CFAR", "nophys": "Hybrid, no physics loss",
    "cold": "Cold-start ICA", "ica_only": "ICA only (no network)",
}


def save(fig, name: str, record: dict) -> None:
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=220)
    (FIGS / f"{name}.json").write_text(json.dumps(record, indent=2, default=float),
                                       encoding="utf-8")
    plt.close(fig)
    print("wrote", FIGS / f"{name}.pdf")


def pd_curve(rows, arm, folds=("fold1", "fold2")):
    out = []
    for fold in folds:
        pts = sorted((r["scr_db"], r["pd"], r["boot_lo"], r["boot_hi"]) for r in rows
                     if r["arm"] == arm and r["fold"] == fold)
        if pts:
            out.append(pts)
    return out


def fig_power(rows, arms, name, title):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    record = {"figure": name, "panels": {}}
    for arm in arms:
        curves = pd_curve(rows, arm)
        if not curves:
            continue
        scr = np.array([p[0] for p in curves[0]])
        pd = np.array([[p[1] for p in c] for c in curves]).mean(0)
        lo = np.array([[p[2] for p in c] for c in curves]).mean(0)
        hi = np.array([[p[3] for p in c] for c in curves]).mean(0)
        ax.plot(scr, pd, label=LABEL[arm], **STYLE.get(arm, {}))
        ax.fill_between(scr, lo, hi, color=STYLE.get(arm, {}).get("color", "k"),
                        alpha=0.12, linewidth=0)
        record["panels"][arm] = {"scr_db": scr.tolist(), "pd": pd.tolist(),
                                 "boot_lo": lo.tolist(), "boot_hi": hi.tolist()}
    ax.set_xlabel("SCR (dB)")
    ax.set_ylabel(r"$P_d$ at design $P_{fa}=10^{-3}$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(title, fontsize=8)
    save(fig, name, record)


def fig_ablation(rows, name):
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    record = {"figure": name, "panels": {}}
    for arm in ("hybrid", "nophys", "cold", "ica_only"):
        curves = pd_curve(rows, arm)
        if not curves:
            continue
        scr = np.array([p[0] for p in curves[0]])
        pd = np.array([[p[1] for p in c] for c in curves]).mean(0)
        lo = np.array([[p[2] for p in c] for c in curves]).mean(0)
        hi = np.array([[p[3] for p in c] for c in curves]).mean(0)
        ax.plot(scr, pd, label=LABEL[arm], **STYLE.get(arm, {}))
        ax.fill_between(scr, lo, hi, color=STYLE.get(arm, {}).get("color", "k"),
                        alpha=0.12, linewidth=0)
        record["panels"][arm] = {"scr_db": scr.tolist(), "pd": pd.tolist()}
    ax.set_xlabel("SCR (dB)")
    ax.set_ylabel(r"$P_d$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Physics-informed loss and warm-start ablations", fontsize=8)
    save(fig, name, record)


def fig_per_acq(rows, acqs, name):
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    arms = ("hybrid", "anmf_dl")
    x = np.arange(len(acqs))
    w = 0.38
    rec = {}
    for i, arm in enumerate(arms):
        vals = []
        for acq in acqs:
            r = [q for q in rows if q["arm"] == arm and q["acq"] == acq]
            vals.append(r[0]["pd"] if r else np.nan)
        ax.bar(x + (i - 0.5) * w, vals, width=w, label=LABEL[arm],
               color=STYLE[arm]["color"], alpha=0.85, linewidth=0)
        rec[arm] = [float(v) for v in vals]
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_starea", "").replace("_starec0000", "")
                        .replace("_antstep", "") for a in acqs], rotation=90, fontsize=5.5)
    ax.set_ylabel(r"$P_d$ at 0 dB")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Per-acquisition detection at 0 dB (each acquisition tested out-of-fold)",
                 fontsize=8)
    save(fig, name, {"figure": name, "acquisitions": list(acqs), "pd": rec})


def fig_pfa_audit(pfa_rows, name):
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    arms = ("hybrid", "anmf_dl", "ar8", "ca", "os", "nophys")
    xs = np.linspace(0.6, 1.4, len(arms)) * DESIGN_PFA
    for i, arm in enumerate(arms):
        for fold in ("fold1", "fold2"):
            r = [q for q in pfa_rows if q["arm"] == arm and q["fold"] == fold]
            if not r:
                continue
            r = r[0]
            ax.errorbar(xs[i] * (1 - 0.06) if fold == "fold1" else xs[i] * (1 + 0.06),
                        r["realised_pfa"],
                        yerr=[[r["realised_pfa"] - r["cp_lo"]], [r["cp_hi"] - r["realised_pfa"]]],
                        fmt=STYLE.get(arm, {}).get("marker", "o"), color=STYLE.get(arm, {}).get("color", "k"),
                        markersize=4, capsize=2, linewidth=0.8)
    ax.axhline(DESIGN_PFA, color="k", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[a].split(" (")[0] for a in arms], rotation=25, fontsize=6)
    ax.set_ylabel("independently observed TEST-H0 $P_{fa}$")
    ax.set_title("Threshold transfer: design $10^{-3}$ vs measured", fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.4, axis="y")
    save(fig, name, {"figure": name, "design_pfa": DESIGN_PFA,
                     "rows": pfa_rows})


def fig_roc(h0, h1, thr, name):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    arms = ("hybrid", "anmf_dl", "ar8", "notch", "ca", "os")
    grid = np.logspace(-5, np.log10(0.3), 220)
    rec = {}
    for arm in arms:
        group = ARM_GROUP[arm]
        pds = []
        for fold in ("fold1", "fold2"):
            if (fold, "test_h0") not in h0 or (fold, group) not in h1:
                pds = []
                break
            h0s = np.sort(h0[(fold, "test_h0")][arm]["score"])[::-1]
            d = h1[(fold, group)]
            j = int(np.argmin(np.abs(d["scr_db"] - 0.0)))
            h1s = d["scores"][arm][:, j]
            n0, n1 = len(h0s), len(h1s)
            k0 = np.maximum(1, np.round(grid * n0)).astype(int)
            k0 = np.minimum(k0, n0)
            thrs = np.array([h0s[k - 1] for k in k0]) - 1e-15
            pds.append(np.array([np.mean(h1s > t) for t in thrs]))
        if not pds:
            continue
        pd = np.mean(np.stack(pds), axis=0)
        ax.semilogx(grid, pd, label=LABEL[arm], **STYLE.get(arm, {}))
        rec[arm] = {"pfa": grid.tolist(), "pd": pd.tolist()}
    ax.axvline(DESIGN_PFA, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$P_{fa}$ (TEST-H0, independent of calibration)")
    ax.set_ylabel(r"$P_d$ at 0 dB")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("ROC at 0 dB", fontsize=8)
    save(fig, name, {"figure": name, "curves": rec, "design_pfa": DESIGN_PFA})


def fig_refcontam(name: str = "fig7_reference_contamination") -> None:
    """Pd of the three VH-reading arms against target leakage into the reference.

    The frozen protocol injects into VV only, so VH is a perfectly target-free
    reference.  Bars are trial-weighted over the two folds; the folds are listed in the
    JSON record so the (large) fold difference stays visible.
    """
    import csv
    path = C.WORK / "reports/MINIMAL_CAMPAIGN_E3_REFERENCE_CONTAMINATION.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({**r, "pd": float(r["pd"]), "n": int(r["n_trials"])})
    conds = ["vv_only", "vh_-10dB", "vh_0dB"]
    cname = {"vv_only": "VV only\n(frozen protocol)",
             "vh_-10dB": "VV + VH at $-10$ dB",
             "vh_0dB": "VV + VH at $0$ dB\n(leakage bound)"}
    arms = ("hybrid", "ar8", "notch")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(conds))
    w = 0.26
    rec = {}
    for i, arm in enumerate(arms):
        vals = []
        for c in conds:
            rr = [r for r in rows if r["arm"] == arm and r["condition"] == c]
            vals.append(sum(r["pd"] * r["n"] for r in rr) / sum(r["n"] for r in rr))
            rec.setdefault(arm, {})[c] = {r["fold"]: r["pd"] for r in rr}
        ax.bar(x + (i - 1) * w, vals, width=w, label=LABEL[arm],
               color=STYLE[arm]["color"], alpha=0.85, linewidth=0)
        rec[arm]["pooled"] = dict(zip(conds, [float(v) for v in vals]))
    ax.set_xticks(x)
    ax.set_xticklabels([cname[c] for c in conds], fontsize=6.5)
    ax.set_ylabel(r"$P_d$ at 0 dB")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=6)
    ax.set_title("Target in the cross-polarisation reference", fontsize=8)
    save(fig, name, {"figure": name, "conditions": conds, "pd": rec,
                     "note": "frozen thresholds; trials identical across conditions"})


def main() -> int:
    cfg = C.config()
    run = C.run_dir(cfg)
    h0, h1 = load_scores(run)
    summary = json.loads((C.WORK / "statistics/ANALYSIS_SUMMARY.json").read_text())
    rows = []
    import csv
    with open(C.WORK / "reports/MINIMAL_CAMPAIGN_RESULTS.csv", newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({**r, "scr_db": float(r["scr_db"]), "pd": float(r["pd"]),
                         "boot_lo": float(r["boot_lo"]), "boot_hi": float(r["boot_hi"])})
    pfa_rows = []
    with open(C.WORK / "reports/MINIMAL_CAMPAIGN_PFA_AUDIT.csv", newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pfa_rows.append({k: (float(v) if k not in ("arm", "fold") else v)
                             for k, v in r.items()})
    acq_rows = []
    with open(C.WORK / "reports/MINIMAL_CAMPAIGN_ACQUISITION_RESULTS.csv", newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            acq_rows.append({**r, "pd": float(r["pd"])})
    thr = {(k.split("|")[0], k.split("|")[1]): v
           for k, v in summary["thresholds"].items()}

    fig_power(rows, ("hybrid", "anmf_dl"), "fig1_hybrid_vs_anmf_power",
              "Corrected comparison at matched design $P_{fa}=10^{-3}$")
    fig_power(rows, ("hybrid", "ar8", "notch", "ca", "os"), "fig2_hybrid_vs_temporal",
              "Hybrid vs temporal and CFAR references")
    fig_roc(h0, h1, thr, "fig3_roc_0db")
    fig_ablation(rows, "fig4_physics_ablation")
    acqs = sorted({r["acq"] for r in acq_rows if r["arm"] == "hybrid"})
    fig_per_acq([r for r in acq_rows if r["arm"] in ("hybrid", "anmf_dl")], acqs,
                "fig5_per_acquisition_pd")
    fig_pfa_audit(pfa_rows, "fig6_pfa_transfer")
    if (C.WORK / "reports/MINIMAL_CAMPAIGN_E3_REFERENCE_CONTAMINATION.csv").is_file():
        fig_refcontam()
    print(json.dumps({"figures": sorted(p.name for p in FIGS.glob("*.pdf"))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
