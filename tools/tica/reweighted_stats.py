#!/usr/bin/env python3
"""Quantify what the MSM reweighting actually changed, instead of eyeballing contours.

Also writes Rg per frame to disk so later scripts stop recomputing it.
"""
import sys, os
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/tica")
import numpy as np
from fit_reference_tica import contiguous_runs

T = "/srv/data/itzeljem79/mlcg-tk/lambda/tica"
CACHE = "/srv/data/itzeljem79/mlcg-tk/lambda/tica_cache"
MIN_BOND = 0.34

tics = np.load(f"{T}/lambda_both_lag100_ref_tics.npy")
w = np.load(f"{T}/lambda_both_lag100_msm_weights.npy")
w = w / w.sum()

rg_path = f"{T}/lambda_rg_per_frame.npy"
if os.path.exists(rg_path):
    rg = np.load(rg_path)
else:
    coords = np.load(f"{CACHE}/ca_coords_nm.npy", mmap_mode="r")
    starts = np.load(f"{CACHE}/segment_starts.npy")
    out = []
    for s in range(len(starts) - 1):
        seg = np.array(coords[starts[s]:starts[s + 1]], dtype=np.float32)
        for a, b in contiguous_runs(seg, MIN_BOND):
            if b - a < 2:
                continue
            blk = seg[a:b]
            cen = blk - blk.mean(axis=1, keepdims=True)
            out.append(np.sqrt((cen ** 2).sum(axis=2).mean(axis=1)) * 10.0)
    rg = np.concatenate(out).astype(np.float32)
    np.save(rg_path, rg)
    print(f"[saved] {rg_path}")

assert rg.shape[0] == tics.shape[0] == w.shape[0]
print(f"frames: {len(rg):,}\n")

print("%-26s %12s %12s" % ("quantity", "raw counts", "MSM-reweighted"))
print("-" * 52)
for name, v in [("mean TIC 1", tics[:, 0]), ("mean TIC 2", tics[:, 1]), ("mean Rg (A)", rg)]:
    print("%-26s %12.3f %12.3f" % (name, v.mean(), float((v * w).sum())))

print()
for thr in [13, 14, 16, 20, 25]:
    m = (rg < thr).astype(np.float64)
    print("%-26s %11.2f%% %11.2f%%" % (f"population Rg < {thr} A", 100 * m.mean(), 100 * (m * w).sum()))

print()
# where is the free energy minimum in each case?
bins = 120
rngs = [[tics[:, 0].min(), tics[:, 0].max()], [tics[:, 1].min(), tics[:, 1].max()]]
Hraw, xe, ye = np.histogram2d(tics[:, 0], tics[:, 1], bins=bins, range=rngs)
Hw, _, _ = np.histogram2d(tics[:, 0], tics[:, 1], bins=bins, range=rngs, weights=w)
xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
for lab, H in [("raw counts", Hraw), ("reweighted", Hw)]:
    i, j = np.unravel_index(np.argmax(H), H.shape)
    print("minimum of F (%-11s) at  TIC1 %+.2f   TIC2 %+.2f" % (lab, xc[i], yc[j]))

print()
occupied = Hw > 0
print("weight concentrated in the top 1%% of occupied bins: %.1f%%"
      % (100 * np.sort(Hw[occupied])[::-1][:max(1, int(0.01 * occupied.sum()))].sum()))
