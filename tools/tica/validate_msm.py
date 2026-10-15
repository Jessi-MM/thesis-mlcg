#!/usr/bin/env python3
"""The checks that decide whether an MSM/FES is trustworthy.

Four of them, in the order a referee would ask:

1. SPECTRAL GAP   -- how many metastable states does the data actually support?
                     A big ratio ts_i/ts_{i+1} says "there are i+1 basins". No gap
                     says the landscape is genuinely broad, not that you erred.
2. PCCA+          -- cut the 500 microstates into metastable sets and see where
                     they sit, so "one big basin" can be checked rather than eyeballed.
3. CHAPMAN-KOLMOGOROV -- the canonical MSM test. Build at lag tau, predict k*tau,
                     compare against a model estimated directly at k*tau. If the
                     Markov assumption is wrong, prediction and estimate diverge.
4. SPLIT-HALF     -- build on two disjoint halves of the trajectories and compare.
                     Catches "this surface is really just noise".
"""
import sys, os
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/theme")
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/tica")
import numpy as np
from fit_reference_tica import contiguous_runs
from deeptime.markov.msm import MaximumLikelihoodMSM

T = "/srv/data/itzeljem79/mlcg-tk/lambda/tica"
CACHE = "/srv/data/itzeljem79/mlcg-tk/lambda/tica_cache"
LAG, MIN_BOND = 100, 0.34
FRAME_PS = 100.0

tics = np.load(f"{T}/lambda_both_lag100_ref_tics.npy")
dtraj_all = np.load(f"{T}/lambda_both_lag100_dtraj.npy")

coords = np.load(f"{CACHE}/ca_coords_nm.npy", mmap_mode="r")
starts = np.load(f"{CACHE}/segment_starts.npy")
lengths = []
for s in range(len(starts) - 1):
    seg = np.array(coords[starts[s]:starts[s + 1]], dtype=np.float32)
    for a, b in contiguous_runs(seg, MIN_BOND):
        if b - a >= 2:
            lengths.append(b - a)
lengths = np.array(lengths)
bounds = np.concatenate([[0], np.cumsum(lengths)])
dtrajs = [dtraj_all[bounds[i]:bounds[i + 1]] for i in range(len(lengths))]
print(f"runs {len(dtrajs):,}  frames {lengths.sum():,}\n")

msm = MaximumLikelihoodMSM(lagtime=LAG).fit_fetch([d for d in dtrajs if len(d) > LAG])

# ---- 1. spectral gap -------------------------------------------------------
ts = msm.timescales() * FRAME_PS / 1000.0
print("=== 1. SPECTRAL GAP: how many basins does the data support? ===")
print("  slowest 8 timescales (ns):", np.round(ts[:8], 1))
ratio = ts[:7] / ts[1:8]
print("  ratio ts_i / ts_(i+1)    :", np.round(ratio, 2))
k = int(np.argmax(ratio)) + 1
print(f"  biggest gap after timescale {k}  (ratio {ratio.max():.2f})")
print(f"  -> the data supports about {k+1} metastable states")
print("     (a ratio near 1 everywhere would mean NO separated basins -- a broad landscape,")
print("      which is a physical result, not a mistake)\n")

# ---- 2. PCCA+ --------------------------------------------------------------
print("=== 2. PCCA+: where are those metastable sets? ===")
nsets = max(2, min(k + 1, 6))
pcca = msm.pcca(nsets)
assign = pcca.assignments
pi = msm.stationary_distribution
active = msm.count_model.state_symbols
centers = np.load(f"{T}/lambda_both_lag100_cluster_centers.npy")
print(f"  {nsets} metastable sets")
for m in range(nsets):
    sel = assign == m
    pop = pi[sel].sum()
    c = centers[active[sel]]
    print(f"   set {m}: {sel.sum():3d} microstates  population {100*pop:5.1f}%   "
          f"TIC1 {c[:,0].mean():+.2f}  TIC2 {c[:,1].mean():+.2f}")
print()

# ---- 3. Chapman-Kolmogorov -------------------------------------------------
print("=== 3. CHAPMAN-KOLMOGOROV: does the model predict its own future? ===")
ck_lags = [LAG * k for k in (1, 2, 3, 4)]
ck_models = []
for L in ck_lags:
    ck_models.append(MaximumLikelihoodMSM(lagtime=L).fit_fetch(
        [d for d in dtrajs if len(d) > L]))
ck = msm.ck_test(ck_models, n_metastable_sets=nsets)
pred, est = np.asarray(ck.predictions), np.asarray(ck.estimates)
print(f"  lags tested (frames): {ck_lags}  = {[l*FRAME_PS/1000 for l in ck_lags]} ns")
dev = np.abs(pred - est)
print(f"  max |predicted - estimated| over all sets and lags : {dev.max():.4f}")
print(f"  mean |predicted - estimated|                        : {dev.mean():.4f}")
for i, L in enumerate(ck_lags):
    d = np.abs(pred[i] - est[i])
    print(f"    at {L*FRAME_PS/1000:5.1f} ns : max dev {d.max():.4f}")
print("  rule of thumb: deviations under ~0.05 are good, over ~0.1 means the Markov")
print("  assumption is failing at this lag.\n")

# ---- 4. split-half ---------------------------------------------------------
print("=== 4. SPLIT-HALF: is the surface reproducible? ===")
rng = np.random.default_rng(3)
perm = rng.permutation(len(dtrajs))
halves = [perm[:len(perm) // 2], perm[len(perm) // 2:]]
pis = []
for h in halves:
    m = MaximumLikelihoodMSM(lagtime=LAG).fit_fetch(
        [dtrajs[i] for i in h if len(dtrajs[i]) > LAG])
    full = np.zeros(centers.shape[0])
    full[m.count_model.state_symbols] = m.stationary_distribution
    pis.append(full)
a, b = pis
both = (a > 0) & (b > 0)
print(f"  microstates populated in both halves : {both.sum()} of {centers.shape[0]}")
print(f"  correlation of stationary distributions: {np.corrcoef(a[both], b[both])[0,1]:.4f}")
Fa, Fb = -np.log(a[both]), -np.log(b[both])
Fa -= Fa.min(); Fb -= Fb.min()
print(f"  free energy per microstate: mean |dF| {np.abs(Fa-Fb).mean():.3f} kT, "
      f"max {np.abs(Fa-Fb).max():.3f} kT")
w = a[both] + b[both]
print(f"  population-weighted mean |dF|          : {np.average(np.abs(Fa-Fb), weights=w):.3f} kT")
print("  (a few kT of scatter in the sparsest states is normal; what matters is that the")
print("   well-populated states agree, which the weighted number reports.)")
