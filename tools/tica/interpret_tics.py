#!/usr/bin/env python3
"""What do TIC 1 and TIC 2 actually measure?

Takes the saved reference TICA model, re-featurises a subsample of segments,
projects them, and then correlates each TIC against (a) every individual input
feature and (b) a set of interpretable physical observables.

The point: a TICA component is a linear combination of 3314 features and means
nothing until you ask what moves with it. Rg alone correlated only -0.26 with
TIC 1, so the slow mode is NOT simply folding/unfolding, and this script finds
out what it is instead.
"""
import sys
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/theme")
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/tica")

import pickle as pkl
import numpy as np
import mdtraj as md
from thesis_style import apply_style, PALETTE, save_figure
from fit_reference_tica import featurise, contiguous_runs

TICA_DIR = "/srv/data/itzeljem79/mlcg-tk/lambda/tica"
CACHE = "/srv/data/itzeljem79/mlcg-tk/lambda/tica_cache"
TOP = "/srv/data/itzeljem79/mlcg-tk/lambda/out/lambda_cg_structure.pdb"
N_SEG, STRIDE, MIN_BOND = 600, 5, 0.34

with open(f"{TICA_DIR}/lambda_both_lag100.pkl", "rb") as f:
    saved = pkl.load(f)
model, recipe = saved["model"], saved["recipe"]
pairs = np.array(recipe["pairs"])
n_dist = len(pairs)
print("model lag %d, recipe %s, %d features" % (saved["lag"], recipe["features"], recipe["n_features"]))

top = md.load(TOP).topology
resnames = [r.name for r in top.residues]
resseq = [r.resSeq for r in top.residues]

coords = np.load(f"{CACHE}/ca_coords_nm.npy", mmap_mode="r")
starts = np.load(f"{CACHE}/segment_starts.npy")
rng = np.random.default_rng(7)
sel = rng.permutation(len(starts) - 1)[:N_SEG]

F_all, T_all = [], []
for s in sel:
    seg = np.array(coords[starts[s]:starts[s + 1]], dtype=np.float32)
    for a, b in contiguous_runs(seg, MIN_BOND):
        if b - a < 2:
            continue
        blk = seg[a:b]
        F = featurise(blk, top, recipe)
        T_all.append(model.transform(F)[:, :2][::STRIDE])
        F_all.append(F[::STRIDE])
F = np.concatenate(F_all); T = np.concatenate(T_all)
del F_all, T_all
print("frames used: %d   features: %d" % F.shape)

# ---- correlation of every feature with each TIC ----------------------------
Fz = (F - F.mean(0)) / (F.std(0) + 1e-12)
Tz = (T - T.mean(0)) / (T.std(0) + 1e-12)
C = (Fz.T @ Tz) / F.shape[0]          # (n_features, 2)
print()
for comp in (0, 1):
    c = C[:, comp]
    cd, ch = c[:n_dist], c[n_dist:]
    print("TIC %d : |corr| max over distances %.3f, over dihedral sin/cos %.3f"
          % (comp + 1, np.abs(cd).max(), np.abs(ch).max() if ch.size else float("nan")))
    print("        mean |corr| distances %.3f, dihedrals %.3f"
          % (np.abs(cd).mean(), np.abs(ch).mean() if ch.size else float("nan")))

print()
print("top 15 distance pairs correlated with TIC 1:")
order = np.argsort(-np.abs(C[:n_dist, 0]))[:15]
for k in order:
    i, j = pairs[k]
    print("   %s%-3d - %s%-3d   |i-j|=%2d   corr %+0.3f"
          % (resnames[i], resseq[i], resnames[j], resseq[j], abs(i - j), C[k, 0]))

# ---- physical observables ---------------------------------------------------
D = F[:, :n_dist]
sep = np.abs(pairs[:, 0] - pairs[:, 1])
obs = {}
N = top.n_atoms
# Rg^2 = (1/N^2) * sum_{i<j} d_ij^2 for equal masses (the sum here is over i<j only,
# which is exactly half the full double sum, cancelling the 1/2 in the usual formula).
obs["Rg (A)"] = np.sqrt((D ** 2).sum(1)) / N * 10.0
obs["end-to-end (A)"] = D[:, np.where((pairs[:, 0] == 0) & (pairs[:, 1] == N - 1))[0][0]] * 10.0
for cut, lab in [(0.8, "contacts<8A"), (1.0, "contacts<10A")]:
    m = sep >= 3
    obs[lab] = (D[:, m] < cut).sum(1).astype(np.float32)
# helicity from the dihedral part: alpha-helical CA pseudo-dihedral ~ +50 deg
if recipe["features"] in ("dihedrals", "both"):
    n_dih = len(recipe["dihedral_atoms"])
    sin_, cos_ = F[:, n_dist:n_dist + n_dih], F[:, n_dist + n_dih:]
    ang = np.degrees(np.arctan2(sin_, cos_))
    hel = ((ang > 30) & (ang < 70))
    obs["helical fraction"] = hel.mean(1).astype(np.float32)
    nw = 5
    for w in range(nw):
        lo, hi = w * n_dih // nw, (w + 1) * n_dih // nw
        obs[f"helicity res {resseq[lo]}-{resseq[min(hi+2, N-1)]}"] = hel[:, lo:hi].mean(1).astype(np.float32)

print()
print("%-28s %9s %9s" % ("observable", "corr TIC1", "corr TIC2"))
for k, v in obs.items():
    print("%-28s %+9.3f %+9.3f"
          % (k, np.corrcoef(v, T[:, 0])[0, 1], np.corrcoef(v, T[:, 1])[0, 1]))

# ---- figure: where in the contact map does TIC 1 live? ---------------------
apply_style("talk")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.8))
for comp, ax in zip((0, 1), axs):
    M = np.full((N, N), np.nan)
    M[pairs[:, 0], pairs[:, 1]] = C[:n_dist, comp]
    M[pairs[:, 1], pairs[:, 0]] = C[:n_dist, comp]
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[resseq[0], resseq[-1], resseq[0], resseq[-1]])
    cb = fig.colorbar(im, ax=ax); cb.set_label(f"corr(distance, TIC {comp+1})", fontsize=13)
    ax.set_xlabel("residue"); ax.set_ylabel("residue")
    ax.set_title(f"TIC {comp+1}: which CA-CA distances move with it", fontsize=14)
fig.suptitle("Lambda repressor: what the TICA components measure "
             f"({F.shape[0]:,} frames from {N_SEG} segments)", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save_figure(fig, "lambda_tic_interpretation", output_dir=f"{TICA_DIR}/figures",
            formats=("png",))
