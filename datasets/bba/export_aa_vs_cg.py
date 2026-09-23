#!/usr/bin/env python3
"""Export the SAME frames of BBA twice: all-atom, and the CG beads mapped from them.

The CG coordinate map is a pure CA selection, so the beads sit exactly on the
all-atom CA atoms. Both files are aligned with the same transform, so loading
them together in PyMOL/VMD overlays them perfectly.
"""
import glob
import numpy as np
import mdtraj as md

RAW = "/group/ag_clementi_cmb/projects/single_protein_datasets/charmm/raw_charmm22star_bba"
OUT = "/srv/data/itzeljem79/mlcg-tk/bba/out"
DEST = "/srv/data/itzeljem79/mlcg-tk/bba/viz"
NFRAMES = 300

import os
os.makedirs(DEST, exist_ok=True)

# ---- all-atom topology: the pdb is the full solvated system, keep the protein
full = md.load(f"{RAW}/bba_50ns_0/structure.pdb")
prot_idx = full.topology.select("protein")
aa_top = full.atom_slice(prot_idx).topology
assert aa_top.n_atoms == 513

# ---- one raw trajectory file, first NFRAMES frames
raw_fn = sorted(glob.glob(f"{RAW}/coords_nowater/*.npy"))[0]
raw = np.load(raw_fn, mmap_mode="r")[:NFRAMES]
print(f"source: {os.path.basename(raw_fn)}  frames={raw.shape[0]} atoms={raw.shape[1]}")

# raw data is in angstroms; mdtraj wants nm
aa = md.Trajectory(np.asarray(raw) / 10.0, aa_top)

# ---- which atoms the pipeline's coord map selects (the CG beads)
M = np.load(f"{OUT}/1FME_cg_coord_map.npy")          # (28, 513)
ca_idx = np.array([np.nonzero(M[i])[0][0] for i in range(M.shape[0])])
print(f"CG beads: {ca_idx.size}  e.g. {[str(list(aa_top.atoms)[i]) for i in ca_idx[:3]]}")

# ---- align the whole all-atom trajectory on its CA atoms, frame 0 as reference
aa.superpose(aa, 0, atom_indices=ca_idx)

# ---- CG is taken from the ALREADY ALIGNED all-atom coords => exact overlay
cg_top = md.load(f"{OUT}/1FME_cg_structure.pdb").topology
atoms = list(cg_top.atoms)
for a, b in zip(atoms[:-1], atoms[1:]):        # CG pdb has no CONECT records
    cg_top.add_bond(a, b)
cg = md.Trajectory(aa.xyz[:, ca_idx], cg_top)

# ---- write
aa[0].save_pdb(f"{DEST}/bba_allatom_top.pdb")
aa.save_dcd(f"{DEST}/bba_allatom.dcd")
cg[0].save_pdb(f"{DEST}/bba_cg_top.pdb")
cg.save_dcd(f"{DEST}/bba_cg.dcd")

# ---- verify they really coincide
err = np.abs(aa.xyz[:, ca_idx] - cg.xyz).max() * 10
print(f"max |CG bead - all-atom CA| = {err:.2e} A  (should be 0)")
print(f"wrote {aa.n_frames} frames: {aa.n_atoms} atoms / {cg.n_atoms} beads -> {DEST}")
