#!/usr/bin/env python
"""Regenerate the manuscript's two figures from the released result tables.

Figure 1 (``fig1_mechanism``) is the mechanism panel: detection probability at the deployed
threshold against the cross-polarised target contrast, both folds.
Figure 2 (``fig2_calibration``) is the calibration panel: realised-to-nominal false-alarm ratio
for the four clutter-only calibration rules.

The plotting code is the code that produced the published files (``phase8_sivp/code/
p8_figures.py``), with only the input and output directories changed.  Nothing is redrawn from
a screenshot and no value is typed in by hand.

Usage
-----
    python scripts/reproduce_figures.py                # -> outputs/figures
    python scripts/reproduce_figures.py --out FIGDIR   # elsewhere
    python scripts/reproduce_figures.py --check        # compare with figures/ and report MAE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P7 = ROOT / "results" / "phase7"
PUBLISHED = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 4, "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.grid": True, "grid.linewidth": 0.35, "grid.alpha": 0.35,
})
COL = {"hybrid": "#1f4e79", "ar8": "#b03a2e", "notch": "#1e8449", "anmf_dl": "#6c3483"}
LBL = {"hybrid": "Hybrid (M1–M3)", "ar8": "AR(8)+NMF", "notch": "Notch (2-tap)",
       "anmf_dl": "ANMF-DL"}
MK = {"hybrid": "o", "ar8": "s", "notch": "^", "anmf_dl": "D"}


def rows(path: Path) -> list[dict]:
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[figure] {out / name}.pdf")


def fig_mechanism(out: Path) -> None:
    dep = rows(P7 / "PHASE7_V1_POLARIMETRIC_RESULTS.csv")
    order = ("vh_absent", "dSCR-20__randphi", "dSCR-10__randphi", "dSCR0__randphi")
    xl = ["absent\n(−∞)", "−20", "−10", "0"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, fold in zip(axes, ("fold1", "fold2")):
        for arm in ("hybrid", "ar8", "notch"):
            vals = []
            for cond in order:
                v = [float(r["pd"]) for r in dep
                     if r["fold"] == fold and r["arm"] == arm and r["condition"] == cond
                     and r["mode"] == "deployment"]
                vals.append(v[0] if v else np.nan)
            ax.plot(range(4), vals, marker=MK[arm], color=COL[arm], label=LBL[arm],
                    clip_on=False)
        a = [float(r["pd"]) for r in dep if r["fold"] == fold and r["arm"] == "anmf_dl"
             and r["mode"] == "deployment"]
        if a:
            ax.axhline(a[0], color=COL["anmf_dl"], ls=(0, (4, 2)), lw=1.0,
                       label="ANMF-DL (VH-invariant)")
        ax.set_xticks(range(4)); ax.set_xticklabels(xl)
        ax.set_xlim(-0.3, 3.3); ax.set_ylim(0, 1)
        ax.set_xlabel("$\\Delta$SCR$_{VH}$  (dB)")
        ax.set_title("Fold 1" if fold == "fold1" else "Fold 2", fontsize=8)
    axes[0].set_ylabel("$P_d$ at SCR$_{VV}$ = 0 dB")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, out, "fig1_mechanism")


def fig_calibration(out: Path) -> None:
    v2 = rows(P7 / "PHASE7_V2_CALIBRATION_RESULTS.csv")
    rules = ("A", "B", "C", "D")
    RL = ["A\npooled", "B\nacq.-stratified", "C\nrobust", "D\nhierarchical"]
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    w = 0.2
    for i, arm in enumerate(("hybrid", "ar8", "notch", "anmf_dl")):
        vals, worst = [], []
        for rule in rules:
            sel = [r for r in v2 if r["arm"] == arm and r["rule"] == rule]
            vals.append(max(float(r["realised_over_design"]) for r in sel) if sel else np.nan)
            worst.append(max(float(r["worst_acq_over_design"]) for r in sel) if sel else np.nan)
        x = [j + (i - 1.5) * w for j in range(len(rules))]
        ax.bar(x, vals, w, color=COL[arm], label=LBL[arm])
        ax.plot(x, worst, ls="none", marker="v", color=COL[arm], markersize=3,
                markeredgecolor="k", markeredgewidth=0.35)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_ylim(0.4, 120)
    ax.set_xticks(range(len(rules))); ax.set_xticklabels(RL, fontsize=6.5)
    ax.set_ylabel("worst realised / nominal $P_{fa}$")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))
    save(fig, out, "fig2_calibration")


def compare(out: Path) -> int:
    """Mean absolute pixel difference against the released renderings."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; skipping the pixel comparison")
        return 0
    ok = True
    for name in ("fig1_mechanism", "fig2_calibration"):
        a = np.asarray(Image.open(PUBLISHED / f"{name}.png").convert("L"), dtype=float)
        b = np.asarray(Image.open(out / f"{name}.png").convert("L"), dtype=float)
        if a.shape != b.shape:
            print(f"{name}: size differs {a.shape} vs {b.shape}")
            ok = False
            continue
        mae = float(np.abs(a - b).mean())
        status = "PASS" if mae <= 1.0 else "DIFFERS"
        ok &= status == "PASS"
        # ASCII only: the default Windows console code page cannot encode a delta sign.
        print(f"{name}: mean |delta pixel| = {mae:.4f} / 255  [{status}]")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "figures"))
    ap.add_argument("--check", action="store_true",
                    help="compare the regenerated PNGs with figures/")
    args = ap.parse_args(argv)
    out = Path(args.out)
    fig_mechanism(out)
    fig_calibration(out)
    return compare(out) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
