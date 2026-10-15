#!/usr/bin/env python3
"""The two implied-timescale plots side by side.

Left is the notebook's plot (cell 6): TICA timescales vs lag. For bba it flattens
and you read the lag off it. For lambda it does not, so it cannot choose the lag.

Right is the MSM implied-timescale test, which the notebook does not do, and which
is what actually settles the lag for lambda.

The grey wedge is the region where timescale < lag: anything there is faster than
the lag and cannot be resolved. A curve must sit well above it to mean anything.
"""
import sys
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/theme")
import numpy as np
from thesis_style import apply_style, PALETTE, save_figure

apply_style("talk")
import matplotlib.pyplot as plt

LAG_NS = np.array([2.0, 5.0, 10.0, 16.0, 25.0])
TICA_TS = {1: [360.87, 542.45, 712.04, 849.66, 1044.73],
           2: [281.83, 396.60, 487.03, 556.76, 656.28]}
MSM_TS = {1: [222.9, 264.1, 276.8, 280.7, 285.0],
          2: [97.8, 105.0, 111.3, 116.8, 123.2],
          3: [58.9, 85.6, 103.8, 113.6, 119.8]}
COL = {1: PALETTE["variant_1"], 2: PALETTE["variant_2"], 3: PALETTE["variant_3"]}

fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.4), sharex=True)

for ax, (data, title) in zip(axs, [
        (TICA_TS, "TICA timescales — the notebook's plot\n(lambda: never flattens)"),
        (MSM_TS, "MSM timescales — the extra test\n(lambda: flattens after 10 ns)")]):
    lo = np.linspace(0, 27, 50)
    ax.fill_between(lo, 0, lo, color=PALETTE["muted"], alpha=0.30, zorder=0)
    ax.plot(lo, lo, color=PALETTE["spine"], lw=1.0, ls=":", zorder=1)
    for k, v in data.items():
        ax.plot(LAG_NS, v, "-o", color=COL[k], label=f"timescale {k}", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("lag time  (ns)")
    ax.set_ylabel("implied timescale  (ns)")
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 27)
    ax.set_ylim(20, 2000)
    ax.legend(loc="lower right", fontsize=12, frameon=False)

axs[0].text(1.1, 120, "still rising at the ceiling\n-> cannot pick a lag here",
            fontsize=12, color=PALETTE["variant_5"], va="top")
axs[1].annotate("flat: <3% from 10 to 25 ns\n-> lag 10 ns chosen here",
                xy=(16, 280.7), xytext=(2.5, 850), fontsize=12,
                color=PALETTE["variant_5"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["variant_5"], lw=1.6))
axs[0].text(1.1, 23, "grey wedge: timescale < lag, nothing resolvable there",
            fontsize=11, color=PALETTE["spine"], va="bottom")

fig.suptitle("Lambda repressor: why the lag had to be chosen from the MSM, not from TICA",
             fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.91))
save_figure(fig, "lambda_its_comparison",
            output_dir="/srv/data/itzeljem79/mlcg-tk/lambda/tica/figures",
            formats=("png",))
