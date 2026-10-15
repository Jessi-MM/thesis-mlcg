#!/usr/bin/env python3
"""Helicity and compactness for each CG model, against the reweighted reference.

TIC 1 is helicity-dominated, so a model can score well on TIC 1 while being
structurally wrong -- an extended helical rod has high helicity AND high Rg.
Measuring both separates "forms helices" from "folds".
"""
import glob
import numpy as np
import mdtraj as md

TOP = "/srv/data/itzeljem79/mlcg-tk/lambda/out/lambda_cg_structure.pdb"
T = "/srv/data/itzeljem79/mlcg-tk/lambda/tica"
top = md.load(TOP).topology
n = top.n_atoms
dih = np.array([[k, k + 1, k + 2, k + 3] for k in range(n - 3)])

def obs_from_xyz_nm(xyz):
    tr = md.Trajectory(xyz=xyz, topology=top)
    ang = np.degrees(md.compute_dihedrals(tr, dih))
    hel = ((ang > 30) & (ang < 70)).mean(1)
    cen = xyz - xyz.mean(1, keepdims=True)
    rg = np.sqrt((cen ** 2).sum(2).mean(1)) * 10.0
    return hel, rg

rows = []

# reference, equilibrium-weighted
rg_ref = np.load(f"{T}/lambda_rg_per_frame.npy")
w = np.load(f"{T}/lambda_both_lag100_msm_weights.npy"); w = w / w.sum()
cache = np.load("/srv/data/itzeljem79/mlcg-tk/lambda/tica_cache/ca_coords_nm.npy", mmap_mode="r")
sub = np.linspace(0, cache.shape[0] - 1, 60000).astype(int)
hel_ref, _ = obs_from_xyz_nm(np.array(cache[sub], dtype=np.float32))
rows.append(("all-atom reference (raw)", hel_ref.mean(), rg_ref.mean()))
rows.append(("all-atom reference (reweighted)", float("nan"), float((rg_ref * w).sum())))

for tag, path in (("prior only", "/srv/data/itzeljem79/mlcg-tk/lambda/prioronly/prod/lambda_prioronly"),
                  ("trained ep0", "/srv/data/itzeljem79/mlcg-tk/lambda/trained_sims/ep0/prod/lambda_ep0"),
                  ("trained ep15", "/srv/data/itzeljem79/mlcg-tk/lambda/trained_sims/ep15/prod/lambda_ep15")):
    fs = sorted(glob.glob(path + "_coords_*.npy"))
    fs = fs[len(fs) // 2:]                       # equilibrated half
    H, R = [], []
    for fn in fs[::4]:                           # every 4th chunk is plenty
        x = np.load(fn).reshape(-1, n, 3) * 0.1  # Angstrom -> nm
        h, r = obs_from_xyz_nm(np.ascontiguousarray(x, dtype=np.float32))
        H.append(h); R.append(r)
    H = np.concatenate(H); R = np.concatenate(R)
    rows.append((tag, H.mean(), R.mean()))

print("%-34s %14s %12s" % ("", "helical frac", "mean Rg (A)"))
print("-" * 62)
for name, h, r in rows:
    hs = "    n/a   " if np.isnan(h) else "%8.3f  " % h
    print("%-34s %14s %10.1f" % (name, hs, r))
