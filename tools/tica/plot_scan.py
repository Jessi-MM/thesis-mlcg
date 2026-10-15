#!/usr/bin/env python3
"""Figures for the lambda TICA featurisation scan (3000 segments, cis-split).

Numbers are transcribed from /srv/data/itzeljem79/mlcg-tk/lambda/tica_scan.log
(scan of 2026-10-13, 10:33 -> 14:17).
"""
import sys
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/theme")

import numpy as np
from thesis_style import apply_style, PALETTE, save_figure

apply_style("talk")
import matplotlib.pyplot as plt

LAG_NS = np.array([2.0, 5.0, 10.0, 16.0, 25.0])
SEG_NS = 50.0  # one adaptive-sampling segment

TS1 = {
    "dihedrals": [191.05, 266.12, 326.81, 366.28, 402.04],
    "distances": [331.46, 509.96, 676.88, 812.26, 998.40],
    "both":      [360.87, 542.45, 712.04, 849.66, 1044.73],
}
TS2 = {
    "dihedrals": [86.84, 129.16, 170.95, 202.93, 234.81],
    "distances": [236.22, 353.17, 447.39, 518.71, 617.03],
    "both":      [281.83, 396.60, 487.03, 556.76, 656.28],
}
VAMP = {
    "dihedrals": [2.93916, 2.89819, 2.84173, 2.78423, 2.70412],
    "distances": [2.96932, 2.94729, 2.91239, 2.86888, 2.79961],
    "both":      [2.97360, 2.95242, 2.91781, 2.87409, 2.80465],
}
NFEAT = {"dihedrals": 154, "distances": 3160, "both": 3314}
COLOR = {"dihedrals": PALETTE["variant_2"],
         "distances": PALETTE["variant_1"],
         "both":      PALETTE["variant_3"]}
ORDER = ["dihedrals", "distances", "both"]

fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))

# ---- panel 1: implied timescales -------------------------------------------
ax = axs[0]
ax.axhspan(0, SEG_NS, color=PALETTE["muted"], alpha=0.18, zorder=0)
ax.axhline(SEG_NS, color=PALETTE["spine"], lw=1.2, ls=":", zorder=1)
ax.text(2.3, SEG_NS * 0.80, "one segment = 50 ns", fontsize=12,
        color=PALETTE["spine"], va="top")

for k in ORDER:
    ax.plot(LAG_NS, TS1[k], "-o", color=COLOR[k], label=f"{k}  ({NFEAT[k]})", zorder=3)
    ax.plot(LAG_NS, TS2[k], "--s", color=COLOR[k], alpha=0.55, ms=5, lw=1.6, zorder=2)

ax.set_xlabel("TICA lag time  (ns)")
ax.set_ylabel("implied timescale  (ns)")
ax.set_title("No plateau: estimates still rising at the lag ceiling", fontsize=14)
ax.set_xlim(0, 27)
ax.set_ylim(0, 1150)
leg = ax.legend(title="featurisation (features)", loc="upper left", fontsize=12,
                title_fontsize=12, frameon=False)
ax.annotate("solid = TIC 1\ndashed = TIC 2", xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=12, color=PALETTE["spine"])

# ---- panel 2: cross-validated VAMP-2 ---------------------------------------
ax = axs[1]
ax.axhline(3.0, color=PALETTE["spine"], lw=1.2, ls=":", zorder=1)
ax.text(24.6, 2.9955, "ceiling at score-dim 2", fontsize=12,
        color=PALETTE["spine"], ha="right", va="top")
for k in ORDER:
    ax.plot(LAG_NS, VAMP[k], "-o", color=COLOR[k], label=k, zorder=3)
ax.set_xlabel("TICA lag time  (ns)")
ax.set_ylabel("cross-validated VAMP-2")
ax.set_title("Same ranking, but compressed against the ceiling", fontsize=14)
ax.set_xlim(0, 27)
ax.set_ylim(2.66, 3.01)
ax.legend(loc="lower left", fontsize=12, frameon=False)

fig.suptitle("Lambda repressor: TICA featurisation comparison  "
             "(3000 of 15700 segments, held-out scoring)", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save_figure(fig, "lambda_tica_scan",
            output_dir="/srv/data/itzeljem79/mlcg-tk/lambda/tica/figures",
            formats=("png",))

# ---- a second, small figure: is "both" really better than "distances"? -----
fig2, ax = plt.subplots(figsize=(7.2, 4.6))
d1 = 100 * (np.array(TS1["both"]) / np.array(TS1["distances"]) - 1)
d2 = 100 * (np.array(TS2["both"]) / np.array(TS2["distances"]) - 1)
ax.axhline(0, color=PALETTE["baseline"], lw=1.5)
ax.plot(LAG_NS, d1, "-o", color=PALETTE["variant_3"], label="TIC 1")
ax.plot(LAG_NS, d2, "--s", color=PALETTE["variant_3"], alpha=0.6, label="TIC 2")
ax.set_xlabel("TICA lag time  (ns)")
ax.set_ylabel("gain of both over distances  (%)")
ax.set_title("Adding 154 dihedral features to 3160 distances", fontsize=14)
ax.set_xlim(0, 27)
ax.set_ylim(0, 22)
ax.legend(loc="upper right", fontsize=12, frameon=False)
ax.text(1.0, 1.2, "positive at every lag, on both components",
        fontsize=12, color=PALETTE["spine"])
fig2.tight_layout()
save_figure(fig2, "lambda_tica_both_vs_distances",
            output_dir="/srv/data/itzeljem79/mlcg-tk/lambda/tica/figures",
            formats=("png",))
