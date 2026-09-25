#!/usr/bin/env python3
"""How much does the factorised prior actually throw away?

The prior is a SUM of independent 1D terms, which means it assumes
P(a,b) = P(a)P(b) for every pair of features. The mutual information
I(a;b) measures exactly how wrong that is -- and kT*I is the average
energy error, in kcal/mol.

A shuffled control gives the finite-sample bias floor.
"""
import glob
import pickle
import numpy as np

OUT = "/srv/data/itzeljem79/mlcg-tk/bba/out"
TAG = "badn_poly_min_pair_4"
kT = 0.0019872041 * 350
NB = 30

nl = pickle.load(open(f"{OUT}/1FME_prior_nls_{TAG}.pkl", "rb"))
x = np.concatenate([np.asarray(np.load(f, mmap_mode="r")[::8])
                    for f in sorted(glob.glob(f"{OUT}/1FME_batch_*_cg_coords.npy"))[:5]])
print(f"{x.shape[0]:,} frames\n")

bo = np.asarray(nl["bonds"]["index_mapping"])
an = np.asarray(nl["angles"]["index_mapping"])
di = np.asarray(nl["pseudo_ca_dihedral"]["index_mapping"])

bond = np.linalg.norm(x[:, bo[0]] - x[:, bo[1]], axis=-1)
u, v = x[:, an[0]] - x[:, an[1]], x[:, an[2]] - x[:, an[1]]
ang = np.arccos(np.clip((u * v).sum(-1) /
                        (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)), -1, 1))
A, B, C, D = (x[:, di[i]] for i in range(4))
b0, b1, b2 = B - A, C - B, D - C
n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
m = np.cross(n1, b1 / np.linalg.norm(b1, axis=-1, keepdims=True))
dih = np.arctan2((m * n2).sum(-1), (n1 * n2).sum(-1))


def mi(a, b, nb=NB):
    """Mutual information in nats, from a 2D histogram."""
    h, _, _ = np.histogram2d(a, b, bins=nb)
    p = h / h.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def report(label, a, b):
    a, b = a.ravel(), b.ravel()
    val = mi(a, b)
    rng = np.random.default_rng(0)
    floor = mi(a, rng.permutation(b))          # shuffled control
    net = max(val - floor, 0.0)
    print(f"  {label:<34s} I = {val:6.4f} nats  (noise floor {floor:6.4f})"
          f"  ->  kT*I = {kT*net:6.3f} kcal/mol")
    return net


print("COUPLING between features the prior treats as independent:")
report("angle_i  vs  dihedral_i", ang[:, :di.shape[1]], dih)
report("dihedral_i  vs  dihedral_i+1", dih[:, :-1], dih[:, 1:])
report("angle_i  vs  angle_i+1", ang[:, :-1], ang[:, 1:])
report("bond_i  vs  angle_i", bond[:, :an.shape[1]], ang)
print()
print("CONTROL (features that share no beads, should be ~0):")
report("angle_i  vs  dihedral_i+8", ang[:, :di.shape[1]-8], dih[:, 8:])

# ---- the CA-trace analogue of a Ramachandran map
print("\nJoint P(angle, dihedral), pooled  [rows: angle 60-150 deg, cols: dihedral -180..180]")
a = np.degrees(ang[:, :di.shape[1]]).ravel()
d = np.degrees(dih).ravel()
h, ae, de = np.histogram2d(a, d, bins=[9, 24], range=[[60, 150], [-180, 180]])
U = -kT * np.log(np.maximum(h / h.sum(), 1e-12))
U -= U.min()
bars = "@%#*+=-:. "
print("        " + "".join(f"{int(de[j]):<3d}" for j in range(0, 24, 3)))
for i in range(9):
    row = "".join(bars[min(int(U[i, j] / 4 * 9), 9)] for j in range(24))
    print(f"  {int(ae[i]):3d}-{int(ae[i+1]):3d} {row}")
print("  (dark = populated / low free energy)")
