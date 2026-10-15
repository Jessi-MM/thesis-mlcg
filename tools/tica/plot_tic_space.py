#!/usr/bin/env python3
"""First look at lambda in TICA space: where the sampling is, and what it means physically.

Panel A is the SAMPLING DENSITY, deliberately not called a free energy surface --
this dataset is adaptive sampling, so counts are not equilibrium populations.
Panel B colours the same plane by radius of gyration, which is what makes the
coordinate interpretable: it shows which region is compact and which is extended.
"""
import sys
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/theme")
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/tica")

import numpy as np
from thesis_style import apply_style, PALETTE, save_figure
from fit_reference_tica import contiguous_runs

TICA_DIR = "/srv/data/itzeljem79/mlcg-tk/lambda/tica"
CACHE = "/srv/data/itzeljem79/mlcg-tk/lambda/tica_cache"
MIN_BOND = 0.34

tics = np.load(f"{TICA_DIR}/lambda_both_lag100_ref_tics.npy")
print("tics:", tics.shape)

coords = np.load(f"{CACHE}/ca_coords_nm.npy", mmap_mode="r")
starts = np.load(f"{CACHE}/segment_starts.npy")
n_seg = len(starts) - 1

rg = []
for s in range(n_seg):
    seg = np.array(coords[starts[s]:starts[s + 1]], dtype=np.float32)
    for a, b in contiguous_runs(seg, MIN_BOND):
        if b - a < 2:
            continue
        blk = seg[a:b]
        cen = blk - blk.mean(axis=1, keepdims=True)
        rg.append(np.sqrt((cen ** 2).sum(axis=2).mean(axis=1)) * 10.0)
rg = np.concatenate(rg)
print("rg:", rg.shape, " aligned:", rg.shape[0] == tics.shape[0])
assert rg.shape[0] == tics.shape[0], "Rg and TICs are not frame-aligned"

apply_style("talk")
import matplotlib.pyplot as plt

x, y = tics[:, 0], tics[:, 1]
bins = 180
rng = [[x.min(), x.max()], [y.min(), y.max()]]

H, xe, ye = np.histogram2d(x, y, bins=bins, range=rng)
S, _, _ = np.histogram2d(x, y, bins=bins, range=rng, weights=rg)
with np.errstate(invalid="ignore", divide="ignore"):
    meanrg = np.where(H > 30, S / H, np.nan)
    logH = np.where(H > 0, np.log10(H), np.nan)

fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.6))
ext = [xe[0], xe[-1], ye[0], ye[-1]]

im0 = axs[0].imshow(logH.T, origin="lower", extent=ext, aspect="auto", cmap="magma_r")
cb0 = fig.colorbar(im0, ax=axs[0]); cb0.set_label("log$_{10}$ frames per bin", fontsize=13)
axs[0].set_title("Sampling density — NOT a free energy surface", fontsize=14)

im1 = axs[1].imshow(meanrg.T, origin="lower", extent=ext, aspect="auto", cmap="viridis")
cb1 = fig.colorbar(im1, ax=axs[1]); cb1.set_label("mean radius of gyration (Å)", fontsize=13)
axs[1].set_title("Same plane, coloured by compactness", fontsize=14)

for ax in axs:
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")

fig.suptitle("Lambda repressor reference, TICA(both, lag 100 = 10 ns), "
             f"{tics.shape[0]:,} frames", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save_figure(fig, "lambda_tic_space", output_dir=f"{TICA_DIR}/figures", formats=("png",))

# how well does TIC 1 track compactness?
ok = np.isfinite(rg)
print("corr(TIC1, Rg) = %.3f" % np.corrcoef(x[ok], rg[ok])[0, 1])
print("corr(TIC2, Rg) = %.3f" % np.corrcoef(y[ok], rg[ok])[0, 1])
for lo, hi, lab in [(-2.3, -1.0, "TIC1 very negative"), (-1.0, 0.0, "TIC1 negative"),
                    (0.0, 1.0, "TIC1 positive"), (1.0, 1.9, "TIC1 very positive")]:
    m = (x >= lo) & (x < hi)
    if m.sum():
        print("  %-20s %9d frames (%5.2f%%)  mean Rg %5.1f A" %
              (lab, m.sum(), 100 * m.mean(), rg[m].mean()))
