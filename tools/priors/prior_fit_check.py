#!/usr/bin/env python3
"""Free-energy vs fitted-prior plots, and a NUMBER for how good each fit is.

Same construction as mlcg-tk's examples/prior_analysis/prior_check.ipynb:
  circles     = -1/beta * log(histogram)      (Boltzmann inversion of the data)
  dashed line = the fitted prior, vertically offset to align

Adds a per-term RMS deviation so fit quality can go in a results table instead
of only being eyeballed.

Usage:
  python prior_fit_check.py --data_dir /srv/data/.../bba/out \
      --prior_tag badn_poly_min_pair_4 --temperature 350 --term pseudo_ca_dihedral
"""
import argparse
import importlib
import os
import pickle as pkl
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mlcg_tk.prior_tools import optimal_offset, prior_evaluator

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True,
               help="directory the pipeline wrote its output to; this script only READS it")
p.add_argument("--prior_tag", required=True)
p.add_argument("--temperature", type=float, default=350.0)
p.add_argument("--term", default=None, help="bonds | angles | non_bonded | pseudo_ca_dihedral; default = all")
p.add_argument("--top", type=int, default=9, help="how many type-tuples to plot (most sampled first)")
p.add_argument("--outdir", default=None)
p.add_argument("--mode", choices=("draft", "talk", "paper"), default=None,
               help="use the thesis theme; talk saves PDF and PNG")
p.add_argument("--base-font-size", type=int, default=None,
               help="override the theme font size for dense figures")
p.add_argument("--repulsion-focus", action="store_true",
               help="also create a short-distance non-bonded plot")
p.add_argument("--repulsion-max-distance", type=float, default=8.0,
               help="upper distance in Angstroms for the focused non-bonded plot")
p.add_argument("--embedding-map", default="mlcg_tk.input_generator.CGEmbeddingMapCA",
               help="dotted path to the CG embedding map, used to print bead type "
                    "codes as residue names; pass '' to keep the raw integers")
p.add_argument("--repulsion-min-distance", type=float, default=3.0,
               help="lower distance for the focused plot; the prior is drawn down to "
                    "here even though no frame ever samples it")
a = p.parse_args()

if a.mode:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "theme"))
    from thesis_style import (AXIS_LABELS, TERM_LABELS, apply_style, fit_panel,
                              get_figure_dim, save_figure, term_color)

    apply_style(a.mode, base_font_size=a.base_font_size)

def _type_names(dotted):
    """bead type code -> residue name, e.g. 9 -> LYS.

    The embedding map goes name -> code and is not injective (LEU and NLE both
    map to 10), so invert with setdefault: the first name declared wins.
    """
    if not dotted:
        return {}
    try:
        mod, attr = dotted.rsplit(".", 1)
        obj = getattr(importlib.import_module(mod), attr)
        mapping = obj() if isinstance(obj, type) else obj
        names = {}
        for resname, code in dict(mapping).items():
            names.setdefault(int(code), str(resname))
        return names
    except Exception as exc:
        print(f"warning: could not load embedding map {dotted!r} ({exc}); "
              f"falling back to raw type codes")
        return {}


TYPE_NAMES = _type_names(a.embedding_map)


def fmt_key(key):
    """(9, 15) -> 'LYS, ARG'; unknown codes stay as their integer."""
    return ", ".join(TYPE_NAMES.get(int(t), str(t)) for t in key)


beta = 1.0 / (a.temperature * 0.0019872041)
J = lambda *x: os.path.join(a.data_dir, *x)
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

    # rank type-tuples by how much data they have.
    #
    # A group and its reversal are the same physical interaction: (i,j)/(j,i) for
    # bonds and non-bonded, (i,j,k)/(k,j,i) for angles, (i,j,k,l)/(l,k,j,i) for
    # dihedrals. mlcg-tk stores BOTH, with byte-identical histograms, so an
    # unfiltered ranking always pairs them on adjacent ranks and --top 9 really
    # shows 5 distinct groups. Collapse onto a canonical orientation first.
    canonical = {}
    for key, hist in stats.items():
        ckey = min(tuple(key), tuple(reversed(key)))
        canonical.setdefault(ckey, (key, hist))
    n_mirrored = len(stats) - len(canonical)
    ranked = sorted(canonical.values(), key=lambda kv: -np.asarray(kv[1]).sum())[:a.top]
    if n_mirrored:
        print(f"{name:<22}({len(stats)} tuples -> {len(canonical)} distinct groups; "
              f"{n_mirrored} mirrored duplicates dropped)")

    n = len(ranked)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    if a.mode:
        fig, axs = plt.subplots(
            nrow, ncol,
            figsize=(get_figure_dim(aspect_ratio=0.30)[0], 3.2 * nrow),
            squeeze=False,
        )
    else:
        fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)

    focus_fig = focus_axs = None
    if a.repulsion_focus and name == "non_bonded":
        if a.mode:
            focus_fig, focus_axs = plt.subplots(
                nrow, ncol,
                figsize=(get_figure_dim(aspect_ratio=0.30)[0], 3.2 * nrow),
                squeeze=False,
            )
        else:
            focus_fig, focus_axs = plt.subplots(
                nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False
            )

    devs = []
    for slot, (ax, (key, hist)) in enumerate(zip(axs.ravel(), ranked)):
        hist = np.asarray(hist, dtype=float)
        mask = hist > 1e-1
        if mask.sum() < 4:
            ax.set_visible(False)
            if focus_axs is not None:
                focus_axs.ravel()[slot].set_visible(False)
            continue
        x = centres[mask]
        dG = -np.log(hist[mask]) / beta

        at_data = prior_evaluator(module, key, torch.tensor(x)).detach().numpy()
        off = optimal_offset(dG, at_data)
        dev = float(np.sqrt(np.mean((dG - (at_data + off)) ** 2)))
        devs.append((key, int(mask.sum()), dev))

        grid = torch.linspace(float(x.min()), float(x.max()), 201)
        curve = prior_evaluator(module, key, grid).detach().numpy() + off

        if a.mode:
            fit_panel(ax, x, dG, grid.numpy(), curve, term=name)
            ax.set_title(fmt_key(key))
        else:
            ax.scatter(x, dG, s=14, facecolors="none", edgecolors="k", label="data")
            ax.plot(grid.numpy(), curve, "--", lw=2, color="tab:blue", label="prior fit")
            ax.set_title(fmt_key(key), fontsize=8)
            ax.tick_params(labelsize=7)

        if focus_axs is not None:
            # `grid` above stops at the first occupied bin, because that is where the
            # DATA stops. The prior itself is defined everywhere, and for a repulsion
            # the whole point is what it does closer in than anything ever observed --
            # so evaluate it down to --repulsion-min-distance and mark that stretch
            # as extrapolation rather than leaving the panel blank.
            fax = focus_axs.ravel()[slot]
            g_ex = torch.linspace(a.repulsion_min_distance, float(x.min()), 201)
            c_ex = prior_evaluator(module, key, g_ex).detach().numpy() + off
            if a.mode:
                fit_panel(fax, x, dG, grid.numpy(), curve, term=name)
                fax.plot(g_ex.numpy(), c_ex, ":", color=term_color(name), zorder=3,
                         label="prior, never sampled")
                fax.set_title(fmt_key(key))
            else:
                fax.scatter(x, dG, s=14, facecolors="none", edgecolors="k", label="data")
                fax.plot(grid.numpy(), curve, "--", lw=2, color="tab:blue", label="prior fit")
                fax.plot(g_ex.numpy(), c_ex, ":", lw=2, color="tab:red",
                         label="prior, never sampled")
                fax.set_title(fmt_key(key), fontsize=8)
                fax.tick_params(labelsize=7)
            # mark where the data runs out, and keep the y-axis on the data -- the
            # r^-6 divergence would otherwise squash every other feature flat
            fax.axvline(float(x.min()), color="0.6", lw=0.8, zorder=1)
            pad = 0.1 * (dG.max() - dG.min())
            fax.set_ylim(dG.min() - pad, dG.max() + pad)

    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    if a.mode:
        x_label = {"bonds": "bond_length", "angles": "cos_angle",
                   "pseudo_ca_dihedral": "dihedral", "non_bonded": "distance"}[name]
        fig.supxlabel(AXIS_LABELS[x_label])
        fig.supylabel(AXIS_LABELS["free_energy"])
        axs[0, 0].legend(loc="best")
        fig.suptitle(f"{TERM_LABELS.get(name, name)} — {a.prior_tag} (T = {a.temperature:.0f} K)")
    else:
        axs[0, 0].set_ylabel("free energy (kcal/mol)")
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(f"{name}  —  {a.prior_tag}  (T = {a.temperature:.0f} K)")
    fig.tight_layout()
    if a.mode:
        paths = save_figure(fig, f"fit_{name}", output_dir=outdir)
        # save_figure(fig, "fit_bonds", formats=("pdf", "png")) if needed pdf
        fn = ", ".join(paths)
    else:
        fn = os.path.join(outdir, f"fit_{name}.png")
        fig.savefig(fn, dpi=130)
    plt.close(fig)

    focus_fn = None
    if focus_fig is not None:
        for focus_ax in focus_axs.ravel()[n:]:
            focus_ax.set_visible(False)
        for focus_ax in focus_axs.ravel()[:n]:
            focus_ax.set_xlim(left=a.repulsion_min_distance,
                              right=a.repulsion_max_distance)
        if a.mode:
            focus_fig.supxlabel(AXIS_LABELS["distance"])
            focus_fig.supylabel(AXIS_LABELS["free_energy"])
            focus_axs[0, 0].legend(loc="best")
            focus_fig.suptitle(
                f"{TERM_LABELS[name]} — short-distance repulsion "
                f"(r ≤ {a.repulsion_max_distance:g} Å)"
            )
        else:
            focus_axs[0, 0].set_ylabel("free energy (kcal/mol)")
            focus_axs[0, 0].legend(fontsize=7)
            focus_fig.suptitle(
                f"{name} — short-distance repulsion "
                f"(r <= {a.repulsion_max_distance:g} A)"
            )
        focus_fig.tight_layout()
        if a.mode:
            focus_paths = save_figure(
                focus_fig, f"fit_{name}_repulsion_focus", output_dir=outdir
            )
            focus_fn = ", ".join(focus_paths)
        else:
            focus_fn = os.path.join(outdir, f"fit_{name}_repulsion_focus.png")
            focus_fig.savefig(focus_fn, dpi=130)
        plt.close(focus_fig)

    # A *_from_values fit function never looks at the shape of the curve -- e.g.
    # fit_repulsion_from_values just sets sigma to a percentile of the distance
    # distribution. Nothing was fitted to these points, so an RMS deviation
    # against them measures nothing. Print n/a rather than a number that would
    # always trip the "poor fit" flag.
    # .func unwraps a functools.partial -- mlcg-tk wraps the non-bonded fit fn to
    # bake in percentile/cutoff, and a partial has no __name__.
    fit_fn = getattr(bldr.prior_fit_fn, "func", bldr.prior_fit_fn)
    fit_fn_name = getattr(fit_fn, "__name__", str(fit_fn))
    is_fitted = not fit_fn_name.endswith("_from_values")

    for key, nb, dev in devs:
        if is_fitted:
            flag = "   <== poor fit" if dev > 0.5 else ""
            print(f"{name:<22}{fmt_key(key):<30}{nb:>6}{dev:>10.3f}{flag}")
        else:
            print(f"{name:<22}{fmt_key(key):<30}{nb:>6}{'n/a':>10}")
    if devs and is_fitted:
        print(f"{name:<22}{'MEAN':<30}{'':>6}{np.mean([d for _, _, d in devs]):>10.3f}")
    elif devs:
        print(f"{name:<22}{'-- not a fitted term --':<30}{'':>6}{'':>10}"
              f"   {fit_fn_name}() sets parameters without fitting; RMS is meaningless")
    print(f"  -> {fn}")
    if focus_fn:
        print(f"  -> {focus_fn}")


# Example usage:
# Plot all available terms for a given prior, and save the plots to a directory:
# cd /storage/mi/itzeljem79/Projects/thesis-mlcg/tools/priors && python prior_fit_check.py --data_dir /srv/data/itzeljem79/mlcg-tk/bba/out --prior_tag badn_poly_min_pair_4 --temperature 350 --mode talk --base-font-size 12 --outdir /srv/data/itzeljem79/mlcg-tk/bba/out/fitcheck_badn_poly_min_pair_4_talk

# Plot only a specific term (e.g., pseudo_ca_dihedral) for a given prior:
# cd /storage/mi/itzeljem79/Projects/thesis-mlcg/tools/priors && python prior_fit_check.py --data_dir /srv/data/itzeljem79/mlcg-tk/bba/out --prior_tag badn_poly_min_pair_4 --temperature 350 --term pseudo_ca_dihedral --mode talk --base-font-size 12 --outdir /srv/data/itzeljem79/mlcg-tk/bba/out/fitcheck_badn_poly_min_pair_4_talk