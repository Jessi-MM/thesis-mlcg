#!/usr/bin/env python3
"""Free energy surfaces on the reference TICA axes, N panels side by side.

Each panel is  F = -kT ln P  over a 2D histogram of TIC 1 / TIC 2, via deeptime's
`energy2d` / `plot_energy2d` -- the same construction the group's reference
notebook uses.

TWO THINGS THIS ENFORCES, BOTH LEARNED THE HARD WAY
---------------------------------------------------
1. ONE COLOUR SCALE for every panel. deeptime auto-scales each panel
   independently, which makes two surfaces look far more different than they are.
   Panels drawn on different scales cannot be compared by eye.
2. ONE SET OF AXIS LIMITS, taken from the reference if given, so a model that
   explores less shows up as a smaller blob rather than being silently zoomed in.

REWEIGHTING
-----------
`--weights` applies per-frame MSM weights, and belongs ONLY to the reference: the
all-atom data came from adaptive sampling, so raw frame counts are not equilibrium
populations. CG simulations are Langevin runs at equilibrium and are histogrammed
raw -- reweighting them would be wrong.

USAGE
-----
  python plot_fes.py --out fes.png \\
      --reference /group/.../bba_cadist_diheds_lag_100_ref_tics.npy \\
      --weights   /group/.../bba_cadist_diheds_lag_100_msm_weights.npy \\
      --panel "prior only:/srv/data/.../prioronly/prod/tics.npy" \\
      --panel "epoch 100:/srv/data/.../ep100/tics.npy"
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deeptime.util import energy2d
from deeptime.plots import plot_energy2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", default=[], metavar="LABEL:PATH",
                    help="a TICs .npy to draw, with its title; repeatable. "
                         "Split on the LAST colon, so labels may contain colons.")
    ap.add_argument("--reference", default=None, help="reference TICs .npy (drawn first)")
    ap.add_argument("--weights", default=None, help="MSM weights for the REFERENCE only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, default=13)
    ap.add_argument("--vmax", type=float, default=None,
                    help="top of the shared colour scale; default = auto from the data")
    ap.add_argument("--title", default=None)
    ap.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("LO","HI"),
                    help="force TIC1 axis limits; use the SAME values across separate "
                         "figures so they stay comparable")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("LO","HI"),
                    help="force TIC2 axis limits")
    ap.add_argument("--theme", action="store_true",
                    help="use the thesis theme from tools/theme")
    args = ap.parse_args()

    if args.theme:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theme"))
        try:
            from thesis_style import apply_style
            apply_style("talk")
        except Exception as e:
            print(f"(theme unavailable, continuing unstyled: {e})")

    panels = []
    if args.reference:
        ref = np.load(args.reference)
        w = np.load(args.weights) if args.weights else None
        if w is not None and w.shape[0] != ref.shape[0]:
            raise SystemExit(f"weights ({w.shape[0]}) do not match reference ({ref.shape[0]})")
        panels.append(("Reference (all-atom)" + (" reweighted" if w is not None else ""), ref, w))
    for spec in args.panel:
        if ":" not in spec:
            raise SystemExit(f"--panel needs LABEL:PATH, got {spec!r}")
        label, path = spec.rsplit(":", 1)   # rsplit: labels may contain colons, paths rarely do
        panels.append((label, np.load(path), None))
    if not panels:
        raise SystemExit("nothing to plot: pass --reference and/or --panel")

    # Build every energy surface first, so the colour scale can span all of them.
    energies, vmax = [], 0.0
    for label, tics, w in panels:
        e = energy2d(tics[:, 0], tics[:, 1], weights=w)
        energies.append(e)
        finite = np.asarray(e.energies)[np.isfinite(np.asarray(e.energies))]
        vmax = max(vmax, float(finite.max()))
        print(f"  {label:<34} {tics.shape[0]:>9} frames   max F = {finite.max():.2f}")
    vmax = args.vmax if args.vmax is not None else vmax
    levels = np.linspace(0.0, vmax, args.levels)
    print(f"\nshared colour scale: 0 .. {vmax:.2f}")

    # Axis limits from the reference when present, else the union of all panels.
    base = panels[0][1] if args.reference else np.concatenate([p[1] for p in panels])
    pad = 0.08 * (base[:, 0].max() - base[:, 0].min())
    xlim = tuple(args.xlim) if args.xlim else (base[:, 0].min() - pad, base[:, 0].max() + pad)
    ylim = tuple(args.ylim) if args.ylim else (base[:, 1].min() - pad, base[:, 1].max() + pad)

    cmap = plt.get_cmap("turbo")
    cmap.set_bad(color="white")
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 4.4), sharex=True, sharey=True, squeeze=False)
    for ax, (label, _, _), e in zip(axs[0], panels, energies):
        plot_energy2d(e, ax=ax, levels=levels,
                      contourf_kws={"cmap": cmap, "extend": "both"})
        ax.set_title(label)
        ax.set_xlabel("TIC 1")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    axs[0][0].set_ylabel("TIC 2")
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
