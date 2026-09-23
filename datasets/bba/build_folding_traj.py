#!/usr/bin/env python3
"""Stitch one lineage of the adaptive-sampling tree into a continuous folding movie."""
import glob
import os
import re
import numpy as np
import mdtraj as md

D = "/group/ag_clementi_cmb/projects/single_protein_datasets/charmm/raw_charmm22star_bba"
C = D + "/coords_nowater"
OUT = "/srv/data/itzeljem79/mlcg-tk/bba/out"
DEST = "/srv/data/itzeljem79/mlcg-tk/bba/viz"

SPAWN = re.compile(r"bba_coor_folding-bba_(\d+)_e(\d+)s(\d+)_e(\d+)s(\d+)p\d+f(\d+)\.npy$")
ROOT = re.compile(r"bba_coor_folding-bba_(\d+)_e(\d+)s(\d+)_bba_50ns_(\d+)\.npy$")

node = {}
for p in sorted(glob.glob(C + "/*.npy")):
    b = os.path.basename(p)
    m = SPAWN.match(b)
    if m:
        fam, ep, sim, pep, psim, pf = (int(x) for x in m.groups())
        node[(fam, ep, sim)] = dict(path=p, parent=(fam, pep, psim), pframe=pf)
    else:
        m = ROOT.match(b)
        if m:
            fam, ep, sim, _ = (int(x) for x in m.groups())
            node[(fam, ep, sim)] = dict(path=p, parent=None, pframe=None)

chain = [tuple(r) for r in np.load(f"{DEST}/_chain.npy")]

# ---- stitch. child[0] == parent[pframe+1], so parent contributes [0 : pframe+1]
pieces, marks, t = [], [], 0
for i, key in enumerate(chain):
    x = np.asarray(np.load(node[key]["path"], mmap_mode="r"))
    if i < len(chain) - 1:
        cut = node[chain[i + 1]]["pframe"] + 1
        x = x[:cut]
    pieces.append(x)
    marks.append((t, f"e{key[1]}s{key[2]}"))
    t += len(x)
xyz = np.concatenate(pieces, 0)
print(f"stitched {len(chain)} runs -> {len(xyz)} frames "
      f"({len(xyz) * 0.1:.1f} ns of continuous dynamics)")

# ---- continuity check across every junction
full = md.load(f"{D}/bba_50ns_0/structure.pdb")
prot = full.topology.select("protein")
aa_top = full.atom_slice(prot).topology
traj = md.Trajectory(xyz / 10.0, aa_top)

M = np.load(f"{OUT}/1FME_cg_coord_map.npy")
ca_idx = np.array([np.nonzero(M[i])[0][0] for i in range(M.shape[0])])

ref = traj[-1]                       # the folded end state
traj.superpose(ref, 0, atom_indices=ca_idx)

rmsd = md.rmsd(traj, ref, 0, atom_indices=ca_idx) * 10
c = traj.xyz[:, ca_idx] * 10
cc = c - c.mean(1)[:, None, :]
rg = np.sqrt((np.linalg.norm(cc, axis=-1) ** 2).mean(1))

print("\njunction continuity (RMSD between consecutive frames, A):")
step = np.sqrt(((c[1:] - c[:-1]) ** 2).sum(-1).mean(-1))
for start, label in marks[1:]:
    print(f"  at frame {start:5d} ({label:9s}) step = {step[start-1]:6.3f}   "
          f"typical = {np.median(step):.3f}")

# ---- write both resolutions
traj[0].save_pdb(f"{DEST}/fold_allatom_top.pdb")
traj.save_dcd(f"{DEST}/fold_allatom.dcd")

cg_top = md.load(f"{OUT}/1FME_cg_structure.pdb").topology
atoms = list(cg_top.atoms)
for a, b in zip(atoms[:-1], atoms[1:]):
    cg_top.add_bond(a, b)
cg = md.Trajectory(traj.xyz[:, ca_idx], cg_top)
cg[0].save_pdb(f"{DEST}/fold_cg_top.pdb")
cg.save_dcd(f"{DEST}/fold_cg.dcd")

np.savetxt(f"{DEST}/fold_rg_rmsd.txt", np.c_[np.arange(len(rg)) * 0.1, rg, rmsd],
           header="time_ns  Rg_A  RMSD_to_folded_A", fmt="%10.3f")

# ---- text trace
bars = " .:-=+*#%@"
print("\nRg (top) and RMSD-to-folded (bottom), 7 A ... 22 A:")
for name, v in (("Rg  ", rg), ("RMSD", rmsd)):
    idx = np.clip(((v - 2) / 20 * 9).astype(int), 0, 9)
    s = "".join(bars[i] for i in idx[::max(1, len(idx) // 100)])
    print(f"  {name} {s}")
print(f"\n  start: Rg {rg[0]:.1f} A, RMSD {rmsd[0]:.1f} A")
print(f"  end:   Rg {rg[-1]:.1f} A, RMSD {rmsd[-1]:.1f} A")
print(f"\nwrote {traj.n_frames} frames to {DEST}")
