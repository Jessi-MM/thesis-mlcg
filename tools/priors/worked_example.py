#!/usr/bin/env python3
"""Step 2 by hand, for ONE bond type, on the real BBA data.

Measure -> histogram -> Boltzmann invert -> fit a harmonic -> compare with
what mlcg-tk actually put in the prior model.
"""
import glob
import pickle
import numpy as np
import torch
import mdtraj as md

OUT = "/srv/data/itzeljem79/mlcg-tk/bba/out"
TAG = "badn_poly_min_pair_4"
T = 350.0
kB = 0.0019872041                 # kcal/mol/K
kT = kB * T
print(f"kT at {T:.0f} K = {kT:.4f} kcal/mol\n")

nl = pickle.load(open(f"{OUT}/1FME_prior_nls_{TAG}.pkl", "rb"))
im = np.asarray(nl["bonds"]["index_mapping"])
emb = np.load(f"{OUT}/1FME_cg_embeds.npy")
res = [r.name for r in md.load(f"{OUT}/1FME_cg_structure.pdb").topology.residues]

# ---- 1. pick one bond TYPE and find every instance of it
t0 = (emb[im[0, 0]], emb[im[1, 0]])
cols = [j for j in range(im.shape[1]) if (emb[im[0, j]], emb[im[1, j]]) == t0]
print(f"bond type {t0} = {res[im[0,0]]}-{res[im[1,0]]}")
print(f"  instances of this type in the chain: {cols}  (beads "
      f"{[(int(im[0,j]), int(im[1,j])) for j in cols]})\n")

# ---- 2. MEASURE: the distance, every frame, every instance of this type
x = np.concatenate([np.asarray(np.load(f, mmap_mode="r")[::10])
                    for f in sorted(glob.glob(f"{OUT}/1FME_batch_*_cg_coords.npy"))[:3]])
d = np.linalg.norm(x[:, im[0, cols]] - x[:, im[1, cols]], axis=-1).ravel()
print(f"1. MEASURED {d.size:,} distances   mean {d.mean():.3f}  std {d.std():.3f} A\n")

# ---- 3. HISTOGRAM -> probability
NB, LO, HI = 100, 0.0, 6.0
counts, edges = np.histogram(d, bins=NB, range=(LO, HI))
centres = 0.5 * (edges[1:] + edges[:-1])
P = counts / counts.sum()
ok = counts > 50                                   # ignore near-empty bins
print(f"2. HISTOGRAM  {NB} bins over {LO}-{HI} A;  {ok.sum()} bins usable\n")

# ---- 4. BOLTZMANN INVERSION
U = np.full(NB, np.nan)
U[ok] = -kT * np.log(P[ok])
U[ok] -= np.nanmin(U[ok])                          # shift so the minimum is 0
print("3. BOLTZMANN INVERSION   U(r) = -kT ln P(r)")
print(f"{'r (A)':>8} {'count':>9} {'P(r)':>10} {'U (kcal/mol)':>13}   profile")
for i in np.where(ok)[0]:
    bar = "#" * int(min(U[i], 6.0) / 6.0 * 40)
    print(f"{centres[i]:8.2f} {counts[i]:9d} {P[i]:10.5f} {U[i]:13.3f}   {bar}")

# ---- 5. FIT  U = k/2 (r - r0)^2
w = np.where(ok & (U < 3.0))[0]                    # fit the well, not the noisy tails
A = np.c_[centres[w] ** 2, centres[w], np.ones(w.size)]
c2, c1, c0 = np.linalg.lstsq(A, U[w], rcond=None)[0]
k_fit = 2 * c2
r0_fit = -c1 / (2 * c2)
print(f"\n4. FIT a parabola over the {w.size} bins with U < 3 kcal/mol")
print(f"   U(r) = k/2 (r - r0)^2  ->  k = {k_fit:8.2f} kcal/mol/A^2   r0 = {r0_fit:.4f} A")

# ---- 6. compare with what mlcg-tk stored
m = torch.load(f"{OUT}/{TAG}_prior_model.pt", weights_only=False).models["bonds"].model
print(f"\n5. mlcg-tk's fitted values for this type:")
print(f"   k = {float(m.k[t0]):8.2f} kcal/mol/A^2   x_0 = {float(m.x_0[t0]):.4f} A")
print(f"\n   difference: k {100*abs(k_fit-float(m.k[t0]))/float(m.k[t0]):.1f}% , "
      f"r0 {abs(r0_fit-float(m.x_0[t0])):.4f} A")
