#!/usr/bin/env python3
"""Build the starting-structure ensemble for CG simulations of lambda.

`mlcg-tk-gen_sim_input` wants a multi-model CA-only PDB. bba's equivalent
(`bba_fes_ensemble_350K_charmm22star.pdb`) holds 100 models of 28 atoms, so this
writes 100 models of 80 atoms to match the convention. Aldo has no lambda
directory at all, so there was nothing to copy.

HOW THE 100 ARE CHOSEN
----------------------
k-means with 100 centres on the reference TICs, then the single real frame
nearest each centre. That spreads the starting points across the whole free
energy surface, which is what "fes_ensemble" means: simulations collectively
explore everywhere rather than all starting in the dominant basin.

The alternative -- drawing 100 frames according to the MSM equilibrium weights --
would start them where the protein actually lives, but with short simulations
that is close to circular: you would largely recover the starting distribution.
Spread starts plus long enough runs lets each simulation find its OWN
equilibrium, which is the thing we want to compare against the reference.
Either way, equilibration must be checked rather than assumed.

Frames are taken from the cis-split set used for the TICA projection, so none of
them contain a cis peptide bond.
"""
import sys
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/tica")
import numpy as np
import mdtraj as md
from fit_reference_tica import contiguous_runs

T = "/srv/data/itzeljem79/mlcg-tk/lambda/tica"
CACHE = "/srv/data/itzeljem79/mlcg-tk/lambda/tica_cache"
TOP = "/srv/data/itzeljem79/mlcg-tk/lambda/out/lambda_cg_structure.pdb"
OUT = "/storage/mi/itzeljem79/Projects/thesis-mlcg/datasets/lambda/lambda_fes_ensemble_350K.pdb"
N_MODELS, MIN_BOND, SEED = 100, 0.34, 11

tics = np.load(f"{T}/lambda_both_lag100_ref_tics.npy")
coords = np.load(f"{CACHE}/ca_coords_nm.npy", mmap_mode="r")
starts = np.load(f"{CACHE}/segment_starts.npy")

# map TIC row -> row in the cache (the cis split removed frames, so they differ)
idx = []
for s in range(len(starts) - 1):
    seg = np.array(coords[starts[s]:starts[s + 1]], dtype=np.float32)
    base = starts[s]
    for a, b in contiguous_runs(seg, MIN_BOND):
        if b - a >= 2:
            idx.append(np.arange(base + a, base + b))
idx = np.concatenate(idx)
print(f"tics {tics.shape[0]:,}   mapped cache rows {idx.shape[0]:,}   match {idx.shape[0] == tics.shape[0]}")
assert idx.shape[0] == tics.shape[0]

from deeptime.clustering import KMeans
rng = np.random.default_rng(SEED)
sub = rng.choice(tics.shape[0], size=400000, replace=False)
km = KMeans(n_clusters=N_MODELS, init_strategy="kmeans++", max_iter=500,
            fixed_seed=SEED, n_jobs=8).fit_fetch(tics[sub])
centres = km.cluster_centers

pick = np.empty(N_MODELS, dtype=np.int64)
for i, c in enumerate(centres):
    d = ((tics[sub] - c) ** 2).sum(1)
    pick[i] = sub[int(np.argmin(d))]
pick = np.unique(pick)
print(f"distinct starting frames: {len(pick)}")

xyz = np.array(coords[idx[pick]], dtype=np.float32)          # nm, (N, 80, 3)
top = md.load(TOP).topology
traj = md.Trajectory(xyz=xyz, topology=top)
traj.save_pdb(OUT)
print(f"[saved] {OUT}")

rg = np.sqrt(((xyz - xyz.mean(1, keepdims=True)) ** 2).sum(2).mean(1)) * 10
print(f"\nstarting ensemble: {len(pick)} structures")
print(f"  TIC1 {tics[pick,0].min():+.2f} .. {tics[pick,0].max():+.2f}"
      f"   (reference {tics[:,0].min():+.2f} .. {tics[:,0].max():+.2f})")
print(f"  TIC2 {tics[pick,1].min():+.2f} .. {tics[pick,1].max():+.2f}"
      f"   (reference {tics[:,1].min():+.2f} .. {tics[:,1].max():+.2f})")
print(f"  Rg   {rg.min():.1f} .. {rg.max():.1f} A   mean {rg.mean():.1f}")
np.save(f"{T}/lambda_sim_start_tics.npy", tics[pick])
