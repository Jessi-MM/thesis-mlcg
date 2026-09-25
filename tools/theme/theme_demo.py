#!/usr/bin/env python3
"""Preview the thesis theme using the real BBA priors: one fit panel per term."""
import os
import pickle as pkl
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/theme")
from thesis_style import (apply_style, AXIS_LABELS, TERM_LABELS, get_figure_dim,
                          fit_panel, format_legend, save_figure, show_palette, term_color)

from mlcg_tk.prior_tools import optimal_offset, prior_evaluator

OUT = "/srv/data/itzeljem79/mlcg-tk/bba/out"
TAG = "badn_poly_min_pair_4"
beta = 1.0 / (350.0 * 0.0019872041)
XLAB = {"bonds": "bond_length", "angles": "cos_angle",
        "pseudo_ca_dihedral": "dihedral", "non_bonded": "distance"}

apply_style("draft")
show_palette(os.path.join(OUT, "theme_palette.png"))

with open(f"{OUT}/{TAG}_prior_builders.pck", "rb") as f:
    builders = pkl.load(f)
model = torch.load(f"{OUT}/{TAG}_prior_model.pt", weights_only=False)

fig, axs = plt.subplots(2, 2, figsize=get_figure_dim(width_in=9.0, aspect_ratio=0.62))
for ax, bldr in zip(axs.ravel(), builders):
    name = bldr.name
    stats = bldr.histograms.data[name]
    centres = np.asarray(bldr.histograms.bin_centers)
    module = model.models[name].model

    key, hist = max(stats.items(), key=lambda kv: np.asarray(kv[1]).sum())
    hist = np.asarray(hist, dtype=float)
    mask = hist > 1e-1
    x = centres[mask]
    dG = -np.log(hist[mask]) / beta

    at_data = prior_evaluator(module, key, torch.tensor(x)).detach().numpy()
    off = optimal_offset(dG, at_data)
    grid = torch.linspace(float(x.min()), float(x.max()), 201)
    curve = prior_evaluator(module, key, grid).detach().numpy() + off

    fit_panel(ax, x, dG - dG.min(), grid.numpy(), curve - dG.min(), term=name)
    ax.set_title(f"{TERM_LABELS.get(name, name)}   ·   {key}", loc="left")
    ax.set_xlabel(AXIS_LABELS[XLAB[name]])
    ax.set_ylabel(AXIS_LABELS["free_energy"])
    ax.text(0.97, 0.94, f"{int(mask.sum())} bins", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=term_color(name))

format_legend(axs[0, 0], loc="upper center")
fig.suptitle("BBA priors — data vs fit (theme preview, draft mode)", y=1.0)
fig.tight_layout()
save_figure(fig, "theme_demo_priors", output_dir=OUT)
