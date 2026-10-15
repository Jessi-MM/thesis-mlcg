#!/usr/bin/env python3
"""Project a CG simulation onto the reference TICA axes.

Takes the `*_coords_????.npy` chunks that `mlcg-nvt_langevin` writes, featurises
them exactly as the reference analysis did, and pushes them through the SAVED
reference TICA model. Output is a single `(n_frames, 2)` array of TIC 1 / TIC 2.

WHY A SAVED TICA MODEL, NEVER A FRESH FIT
-----------------------------------------
A free energy surface is only comparable to another one if both are drawn on the
SAME axes. TICA fitted on your model's own trajectories would give axes that mean
something different from the reference's, and the two pictures could not be laid
side by side. So the TICA model is an input here, never something this script fits.

THE UNIT TRAP
-------------
mlcg's simulation code writes coordinates in ANGSTROM. The reference features were
built in NANOMETRES. Forgetting the factor of 10 does not raise an error -- it
silently projects every frame to a meaningless place. This script converts, and
prints the mean coordinate magnitude so you can see which unit came in.

USAGE
-----
  python project_sim.py --sim-dir /srv/data/.../prioronly/prod \
                        --out     /srv/data/.../prioronly/prod/tics.npy
"""
import argparse
import os
import pickle as pkl
from glob import glob
from itertools import combinations

import numpy as np

DEFAULT_TICA = ("/group/ag_clementi_cmb/users/sayeg84/mlcg-proteins/bba_ca/"
                "bba_cadist_diheds_lag_100.pkl")
DEFAULT_TOP = "/srv/data/itzeljem79/mlcg-tk/bba/out/1FME_cg_structure.pdb"


def featurise(xyz_nm, top, pairs, dihedral_atoms):
    """The reference feature vector: all pair distances + sin/cos of dihedrals."""
    import mdtraj as md

    traj = md.Trajectory(xyz=xyz_nm, topology=top)
    dih = md.compute_dihedrals(traj, dihedral_atoms)
    return np.concatenate(
        [md.compute_distances(traj, pairs, periodic=False), np.sin(dih), np.cos(dih)],
        axis=1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True,
                    help="directory holding *_coords_????.npy from mlcg-nvt_langevin")
    ap.add_argument("--out", required=True, help="where to write the (n_frames, 2) TICs")
    ap.add_argument("--tica", default=DEFAULT_TICA, help="pickled reference TICA model")
    ap.add_argument("--topology", default=DEFAULT_TOP, help="CG structure pdb")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    args = ap.parse_args()

    import mdtraj as md

    files = sorted(glob(os.path.join(args.sim_dir, "*coords_*.npy")))
    if not files:
        raise SystemExit(f"no *coords_*.npy found in {args.sim_dir}")
    print(f"chunks found : {len(files)}")

    top = md.load(args.topology).topology
    n_beads = top.n_atoms
    pairs = list(combinations(range(n_beads), 2))
    dihedral_atoms = np.array([[k, k + 1, k + 2, k + 3] for k in range(n_beads - 3)])
    print(f"beads        : {n_beads}  ->  {len(pairs)} distances + "
          f"{2*len(dihedral_atoms)} sin/cos = {len(pairs)+2*len(dihedral_atoms)} features")

    with open(args.tica, "rb") as f:
        tica = pkl.load(f)

    # Unit check on the first chunk, reported rather than assumed.
    probe = np.load(files[0])
    mag = float(np.abs(probe).mean())
    unit = "ANGSTROM" if mag > 2.0 else "nanometre"
    print(f"mean |coord| : {mag:.2f}  ->  read as {unit}")
    scale = 0.1 if mag > 2.0 else 1.0

    tics = []
    for fn in files:
        chunk = np.load(fn)                       # (n_traj, n_frames, n_beads, 3)
        for k in range(chunk.shape[0]):           # each replica separately
            xyz = chunk[k][::args.stride] * scale
            tics.append(tica.transform(featurise(xyz, top, pairs, dihedral_atoms))[:, :2])
    tics = np.concatenate(tics)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, tics)
    print(f"\nprojected    : {tics.shape[0]} frames")
    print(f"TIC 1 range  : {tics[:,0].min():.2f} .. {tics[:,0].max():.2f}")
    print(f"TIC 2 range  : {tics[:,1].min():.2f} .. {tics[:,1].max():.2f}")
    print(f"written      : {args.out}")


if __name__ == "__main__":
    main()
