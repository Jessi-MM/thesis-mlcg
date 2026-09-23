"""Export the CG trajectory produced by step 1 for viewing in VMD / PyMOL."""
import numpy as np
import mdtraj as md

OUT = "/srv/data/itzeljem79/mlcg-tk/demo_ca/out"
STRIDE = 10  # 5000 frames is a lot to scrub through; every 10th is plenty

top = md.load(f"{OUT}/1L2Y_cg_structure.pdb").topology

# The CG pdb has no CONECT records, so VMD/PyMOL would draw 20 loose dots.
# The chain connectivity is real (see the bonds neighbour list), so add it.
atoms = list(top.atoms)
for a, b in zip(atoms[:-1], atoms[1:]):
    top.add_bond(a, b)

xyz = np.load(f"{OUT}/1L2Y_cg_coords.npy")[::STRIDE]
traj = md.Trajectory(xyz / 10.0, top)   # mlcg-tk stores angstroms, mdtraj wants nm

traj[0].save_pdb(f"{OUT}/cg_traj_top.pdb")
traj.save_dcd(f"{OUT}/cg_traj.dcd")
print(f"wrote {traj.n_frames} frames, {traj.n_atoms} beads")
