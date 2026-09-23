#!/usr/bin/env python3
"""Reconstruct a continuous folding pathway from the adaptive-sampling tree.

Each trajectory file records its parent as (epoch, sim, frame). Because
child frame 0 IS the parent's spawn frame, a lineage can be stitched into
one continuous path with no discontinuity.
"""
import glob
import os
import re
import numpy as np

D = "/group/ag_clementi_cmb/projects/single_protein_datasets/charmm/raw_charmm22star_bba/coords_nowater"

# bba_coor_folding-bba_<fam>_e<ep>s<sim>_e<pep>s<psim>p0f<pf>.npy      (spawned)
# bba_coor_folding-bba_<fam>_e1s<sim>_bba_50ns_<n>.npy                 (root)
SPAWN = re.compile(r"bba_coor_folding-bba_(\d+)_e(\d+)s(\d+)_e(\d+)s(\d+)p\d+f(\d+)\.npy$")
ROOT = re.compile(r"bba_coor_folding-bba_(\d+)_e(\d+)s(\d+)_bba_50ns_(\d+)\.npy$")

node = {}   # (fam, ep, sim) -> dict(path=..., parent=(fam,ep,sim) or None, pframe=int)
for p in sorted(glob.glob(D + "/*.npy")):
    b = os.path.basename(p)
    m = SPAWN.match(b)
    if m:
        fam, ep, sim, pep, psim, pf = (int(x) for x in m.groups())
        node[(fam, ep, sim)] = dict(path=p, parent=(fam, pep, psim), pframe=pf)
        continue
    m = ROOT.match(b)
    if m:
        fam, ep, sim, _ = (int(x) for x in m.groups())
        node[(fam, ep, sim)] = dict(path=p, parent=None, pframe=None)

print(f"parsed {len(node)} of {len(glob.glob(D + '/*.npy'))} files")
fams = sorted({k[0] for k in node})
eps = sorted({k[1] for k in node})
print(f"families: {fams}   epochs: {eps[0]}..{eps[-1]}")


def rg(x):
    c = x - x.mean(1)[:, None, :]
    return np.sqrt((np.linalg.norm(c, axis=-1) ** 2).mean(1))


def lineage(key):
    """Walk from a leaf back to its root. Returns [root, ..., leaf]."""
    chain = []
    while key is not None and key in node:
        chain.append(key)
        par = node[key]["parent"]
        key = par
    return chain[::-1]


# ---- score deepest-epoch leaves by how compact and stable they end up
deepest = max(eps)
leaves = [k for k in node if k[1] == deepest]
print(f"\n{len(leaves)} candidate leaves at epoch {deepest}; scoring…")

scored = []
for k in leaves:
    try:
        x = np.asarray(np.load(node[k]["path"], mmap_mode="r")[-120:])
    except Exception:
        continue
    r = rg(x)
    chain = lineage(k)
    if len(chain) < 4 or node[chain[0]]["parent"] is not None:
        continue                      # needs a full lineage back to a root
    root_rg = rg(np.asarray(np.load(node[chain[0]]["path"], mmap_mode="r")[:60])).mean()
    scored.append((r.mean(), r.std(), root_rg, len(chain), k))

scored.sort(key=lambda t: t[0] - 0.35 * t[2])     # compact end, extended start
print(f"{'final Rg':>9} {'std':>6} {'root Rg':>8} {'gens':>5}  leaf")
for m, s, rr, n, k in scored[:8]:
    print(f"{m:9.2f} {s:6.2f} {rr:8.2f} {n:5d}  fam{k[0]} e{k[1]}s{k[2]}")

best = scored[0][4]
chain = lineage(best)
print(f"\nchosen lineage ({len(chain)} generations):")
for i, k in enumerate(chain):
    nd = node[k]
    print(f"  {i}: fam{k[0]} e{k[1]}s{k[2]}   spawned at parent frame {nd['pframe']}"
          if nd["parent"] else f"  {i}: fam{k[0]} e{k[1]}s{k[2]}   ROOT")

np.save("/srv/data/itzeljem79/mlcg-tk/bba/viz/_chain.npy",
        np.array([[k[0], k[1], k[2]] for k in chain]))
print("\nsaved chain to viz/_chain.npy")
