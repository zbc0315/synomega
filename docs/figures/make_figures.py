"""Generate the documentation's quantitative charts as self-contained SVGs.

Every number here is transcribed from the SynOmega JCIM paper (main text + SI)
and the module evaluation reports; nothing is synthesised. Re-run after editing
the data tables below:

    python synomega/docs/figures/make_figures.py

Output: SVGs next to this script, embedded by the research-report pages.
Style: white background, thin marks, colourblind-safe categorical pair,
one y-axis per panel (never dual-axis).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent

# Two-colour categorical pair (original vs simplifying), CVD-safe.
C_ORIG = "#1f6fb2"   # blue  — original (unconstrained) model
C_SIMP = "#d1662a"   # orange — simplification-constrained model
GRID = "#d9d9d9"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- expansion-width sweep, 1000 ChEMBL targets (SI Table S5) ----------------
K = [3, 4, 5, 6, 7, 8, 9, 10]
ORIG = {
    "solved": [71.8, 78.4, 79.5, 81.3, 81.3, 81.8, 81.5, 81.8],
    "mean_t": [0.97, 1.28, 1.38, 1.39, 1.43, 1.44, 1.51, 1.62],
    "exp":    [34.9, 41.9, 42.5, 40.8, 40.2, 39.4, 38.9, 38.8],
}
SIMP = {
    "solved": [68.1, 74.9, 79.6, 81.5, 83.3, 84.8, 85.0, 85.1],
    "mean_t": [0.37, 0.48, 0.57, 0.66, 0.75, 0.86, 0.94, 1.04],
    "exp":    [21.9, 28.1, 30.1, 31.2, 31.9, 32.0, 32.2, 32.0],
}
REF_SOLVED = 83.9   # original @ k=50 reference
REF_TIME = 4.59


def _grid(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig_ksweep():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, key, ylab, title in zip(
        axes,
        ["solved", "mean_t", "exp"],
        ["Solved rate (%)", "Mean search time (s)", "Mean node expansions"],
        ["(a) Solved rate", "(b) Search time", "(c) Search effort"],
    ):
        ax.plot(K, ORIG[key], "-o", color=C_ORIG, lw=2, ms=5, label="original")
        ax.plot(K, SIMP[key], "-s", color=C_SIMP, lw=2, ms=5, label="simplifying")
        if key == "solved":
            ax.axhline(REF_SOLVED, ls="--", color="#888", lw=1.2,
                       label="original @ k=50")
        if key == "mean_t":
            ax.axhline(REF_TIME, ls="--", color="#888", lw=1.2,
                       label="original @ k=50")
        ax.set_xlabel("expansion width k")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, loc="left")
        _grid(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "ksweep.svg")
    plt.close(fig)


# --- head-to-head at operating point k=10 (main Fig 3d / Fig 4e) -------------
def fig_solve_compare():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labels = ["Simplifying\n(k=10)", "Original\n(k=10)", "AiZynthFinder\n(matched)"]
    solved = [85.1, 81.8, 46.7]
    colors = [C_SIMP, C_ORIG, "#7a7a7a"]
    bars = ax.bar(labels, solved, color=colors, width=0.6, zorder=3)
    for b, v in zip(bars, solved):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Solved rate on 1000 ChEMBL (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Multi-step solved rate, budget-matched", fontsize=11, loc="left")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "solve_compare.svg")
    plt.close(fig)


# --- paired expansions on 783 jointly-solved targets (main Fig 3d) -----------
def fig_expansions():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    groups = ["All 1000 targets", "783 jointly solved"]
    orig = [38.8, 22.0]
    simp = [32.0, 15.7]
    x = range(len(groups))
    w = 0.36
    ax.bar([i - w / 2 for i in x], orig, w, color=C_ORIG, label="original", zorder=3)
    ax.bar([i + w / 2 for i in x], simp, w, color=C_SIMP, label="simplifying", zorder=3)
    for i, (o, s) in enumerate(zip(orig, simp)):
        ax.text(i - w / 2, o + 0.5, f"{o:.1f}", ha="center", fontsize=9)
        ax.text(i + w / 2, s + 0.5, f"{s:.1f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Mean node expansions")
    ax.set_title("Search effort (lower is better)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "expansions.svg")
    plt.close(fig)


# --- forward: reactant-count distribution + demo hit rate --------------------
def fig_forward_data():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    # reactant-count distribution (DESIGN.md §2.3)
    ax = axes[0]
    cats = ["1", "2", "3", "≥4"]
    pct = [26.4, 61.7, 10.1, 1.5]
    bars = ax.bar(cats, pct, color=C_ORIG, width=0.6, zorder=3)
    for b, v in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}%", ha="center", fontsize=9)
    ax.set_xlabel("number of reactant molecules")
    ax.set_ylabel("share of reactions (%)")
    ax.set_title("(a) Reactant-count distribution", fontsize=11, loc="left")
    _grid(ax)
    # top-1 accuracy (forward_integration.md)
    ax = axes[1]
    labs = ["template\ntop-1", "product\ntop-1"]
    vals = [75.9, 63.6]
    bars = ax.bar(labs, vals, color=[C_ORIG, C_SIMP], width=0.5, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%",
                ha="center", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("(b) Forward top-1 accuracy", fontsize=11, loc="left")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "forward_data.svg")
    plt.close(fig)


# --- plausibility: discriminative power (net-negative filter) ----------------
def fig_plausibility():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    # fraction of candidates scored below a threshold (singlestep_filter_analysis)
    thr = ["<0.3", "<0.4", "<0.5"]
    correct = [2.02, 2.63, 3.60]
    wrong = [8.10, 10.42, 12.95]
    x = range(len(thr))
    w = 0.36
    ax.bar([i - w / 2 for i in x], correct, w, color=C_ORIG,
           label="correct candidates", zorder=3)
    ax.bar([i + w / 2 for i in x], wrong, w, color=C_SIMP,
           label="wrong candidates", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(thr)
    ax.set_xlabel("plausibility score below threshold")
    ax.set_ylabel("share of candidates (%)")
    ax.set_title("Discriminative power on the single-step axis", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "plausibility_power.svg")
    plt.close(fig)


# --- SynScore: U distribution over 1000 targets (SI Table S6) ----------------
def fig_udist():
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    cats = ["0", "1", "2", "3", "4", "≥5", "none"]
    orig = [818, 106, 61, 10, 3, 1, 1]
    simp = [851, 11, 21, 26, 17, 72, 2]
    x = range(len(cats))
    w = 0.38
    ax.bar([i - w / 2 for i in x], orig, w, color=C_ORIG, label="original", zorder=3)
    ax.bar([i + w / 2 for i in x], simp, w, color=C_SIMP, label="simplifying", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_xlabel("U = number of non-purchasable starting materials in best route")
    ax.set_ylabel("number of targets (of 1000)")
    ax.set_title("SynScore driver: distribution of U", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "udist.svg")
    plt.close(fig)


# --- structure-only baselines: per-molecule scoring time (SI Table S1) -------
def fig_baseline_time():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    labels = ["SAscore", "RAscore", "SCScore"]
    ms = [0.22, 63.0, 65.0]
    bars = ax.barh(labels, ms, color=C_ORIG, height=0.55, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("per-molecule scoring time (ms, log scale)")
    for b, v in zip(bars, ms):
        ax.text(v * 1.1, b.get_y() + b.get_height() / 2,
                f"{v:g} ms", va="center", fontsize=9)
    ax.set_title("Structure-only accessibility scores (20k ZINC)",
                 fontsize=11, loc="left")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_time.svg")
    plt.close(fig)


if __name__ == "__main__":
    fig_ksweep()
    fig_solve_compare()
    fig_expansions()
    fig_forward_data()
    fig_plausibility()
    fig_udist()
    fig_baseline_time()
    print("wrote:", *(p.name for p in sorted(OUT.glob("*.svg"))))
