#!/usr/bin/env python3
"""Free-energy vs fitted-prior plots, and a NUMBER for how good each fit is.

Same construction as mlcg-tk's examples/prior_analysis/prior_check.ipynb:
  circles     = -1/beta * log(histogram)      (Boltzmann inversion of the data)
  dashed line = the fitted prior, vertically offset to align

Adds a per-term RMS deviation so fit quality can go in a results table instead
of only being eyeballed.

Usage:
  python prior_fit_check.py --save_dir /srv/data/.../bba/out \
      --prior_tag badn_poly_min_pair_4 --temperature 350 --term pseudo_ca_dihedral
"""
import argparse
import os
import pickle as pkl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mlcg_tk.prior_tools import optimal_offset, prior_evaluator

p = argparse.ArgumentParser()
p.add_argument("--save_dir", required=True)
p.add_argument("--prior_tag", required=True)
p.add_argument("--temperature", type=float, default=350.0)
p.add_argument("--term", default=None, help="bonds | angles | non_bonded | pseudo_ca_dihedral; default = all")
p.add_argument("--top", type=int, default=9, help="how many type-tuples to plot (most sampled first)")
p.add_argument("--outdir", default=None)
a = p.parse_args()

beta = 1.0 / (a.temperature * 0.0019872041)
J = lambda *x: os.path.join(a.save_dir, *x)
outdir = a.outdir or J(f"{a.prior_tag}_fitcheck")
os.makedirs(outdir, exist_ok=True)

with open(J(f"{a.prior_tag}_prior_builders.pck"), "rb") as f:
    builders = pkl.load(f)
model = torch.load(J(f"{a.prior_tag}_prior_model.pt"), weights_only=False)

terms = [a.term] if a.term else [b.name for b in builders]
print(f"{'term':<22}{'key':<30}{'bins':>6}{'RMS dev':>10}  (kcal/mol)")
print("-" * 72)

for bldr in builders:
    if bldr.name not in terms or bldr.name not in model.models:
        continue
    name = bldr.name
    stats = bldr.histograms.data[name]
    centres = np.asarray(bldr.histograms.bin_centers)
    module = model.models[name].model

    # rank type-tuples by how much data they have
    ranked = sorted(stats.items(), key=lambda kv: -np.asarray(kv[1]).sum())[:a.top]

    n = len(ranked)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)

    devs = []
    for ax, (key, hist) in zip(axs.ravel(), ranked):
        hist = np.asarray(hist, dtype=float)
        mask = hist > 1e-1
        if mask.sum() < 4:
            ax.set_visible(False)
            continue
        x = centres[mask]
        dG = -np.log(hist[mask]) / beta

        at_data = prior_evaluator(module, key, torch.tensor(x)).detach().numpy()
        off = optimal_offset(dG, at_data)
        dev = float(np.sqrt(np.mean((dG - (at_data + off)) ** 2)))
        devs.append((key, int(mask.sum()), dev))

        grid = torch.linspace(float(x.min()), float(x.max()), 201)
        curve = prior_evaluator(module, key, grid).detach().numpy() + off

        ax.scatter(x, dG, s=14, facecolors="none", edgecolors="k", label="data")
        ax.plot(grid.numpy(), curve, "--", lw=2, color="tab:blue", label="prior fit")
        ax.set_title(", ".join(str(k) for k in key), fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    axs[0, 0].set_ylabel("free energy (kcal/mol)")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"{name}  —  {a.prior_tag}  (T = {a.temperature:.0f} K)")
    fig.tight_layout()
    fn = os.path.join(outdir, f"fit_{name}.png")
    fig.savefig(fn, dpi=130)
    plt.close(fig)

    for key, nb, dev in devs:
        flag = "   <== poor fit" if dev > 0.5 else ""
        print(f"{name:<22}{str(key):<30}{nb:>6}{dev:>10.3f}{flag}")
    if devs:
        print(f"{name:<22}{'MEAN':<30}{'':>6}{np.mean([d for _, _, d in devs]):>10.3f}")
    print(f"  -> {fn}")
