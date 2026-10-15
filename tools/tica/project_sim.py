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

THE FEATURISATION MUST MATCH, AND IS NO LONGER GUESSED
------------------------------------------------------
Projecting with a different feature vector than the reference was fitted with is
silent and fatal -- the numbers come out, they just mean nothing. Models written
by `fit_reference_tica.py` therefore carry their own recipe (which pairs, which
dihedral quartets, which feature set, which units), and this script USES it rather
than reconstructing one. Older pickles that hold a bare model -- Aldo's bba
`bba_cadist_diheds_lag_100.pkl` -- have no recipe, so for those the legacy
all-pairs-plus-sin/cos-dihedrals recipe is rebuilt from the bead count, which is
what that model was fitted with.

THE UNIT TRAP
-------------
mlcg's simulation code writes coordinates in ANGSTROM. The reference features were
built in NANOMETRES. Forgetting the factor of 10 does not raise an error -- it
silently projects every frame to a meaningless place. This script converts, and
prints the mean coordinate magnitude so you can see which unit came in.

USAGE
-----
  python project_sim.py --sim-dir /srv/data/.../prioronly/prod \
                        --tica    /srv/data/.../lambda_both_lag100.pkl \
                        --out     /srv/data/.../prioronly/prod/tics.npy
"""
import argparse
import os
import pickle as pkl
import sys
from glob import glob
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TICA = ("/group/ag_clementi_cmb/users/sayeg84/mlcg-proteins/bba_ca/"
                "bba_cadist_diheds_lag_100.pkl")
DEFAULT_TOP = "/srv/data/itzeljem79/mlcg-tk/bba/out/1FME_cg_structure.pdb"


def legacy_recipe(n_beads):
    """What a pickle without an embedded recipe was fitted with: every pair
    distance plus sin/cos of every consecutive CA pseudo-dihedral."""
    return {"features": "both",
            "pairs": list(combinations(range(n_beads), 2)),
            "dihedral_atoms": np.array([[k, k + 1, k + 2, k + 3]
                                        for k in range(n_beads - 3)]),
            "n_beads": n_beads, "units": "nanometre"}


def featurise(xyz_nm, top, recipe):
    import mdtraj as md
    traj = md.Trajectory(xyz=xyz_nm, topology=top)
    parts = []
    if recipe["features"] in ("distances", "both"):
        parts.append(md.compute_distances(traj, recipe["pairs"], periodic=False))
    if recipe["features"] in ("dihedrals", "both"):
        h = md.compute_dihedrals(traj, recipe["dihedral_atoms"])
        parts += [np.sin(h), np.cos(h)]
    return np.concatenate(parts, axis=1)


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

    with open(args.tica, "rb") as f:
        saved = pkl.load(f)
    if isinstance(saved, dict) and "model" in saved and "recipe" in saved:
        tica, recipe = saved["model"], saved["recipe"]
        print(f"tica model   : {os.path.basename(args.tica)}  (lag {saved.get('lag')})")
        print(f"recipe       : EMBEDDED -- {recipe['features']}, "
              f"{len(recipe['pairs'])} pairs, {len(recipe['dihedral_atoms'])} dihedrals")
    else:
        tica, recipe = saved, None
        print(f"tica model   : {os.path.basename(args.tica)}  (bare model, no recipe)")

    top = md.load(args.topology).topology
    n_beads = top.n_atoms
    if recipe is None:
        recipe = legacy_recipe(n_beads)
        print(f"recipe       : LEGACY reconstruction -- all pairs + sin/cos dihedrals")
    if recipe["n_beads"] != n_beads:
        raise SystemExit(
            f"topology has {n_beads} beads but the TICA recipe was built for "
            f"{recipe['n_beads']}; this would project onto meaningless axes")
    n_feat = (len(recipe["pairs"]) if recipe["features"] in ("distances", "both") else 0) \
             + (2 * len(recipe["dihedral_atoms"]) if recipe["features"] in ("dihedrals", "both") else 0)
    print(f"beads        : {n_beads}  ->  {n_feat} features")

    probe = np.load(files[0])
    mag = float(np.abs(probe).mean())
    unit = "ANGSTROM" if mag > 2.0 else "nanometre"
    print(f"mean |coord| : {mag:.2f}  ->  read as {unit}")
    scale = 0.1 if mag > 2.0 else 1.0

    tics = []
    for fn in files:
        chunk = np.load(fn)                       # (n_traj, n_frames, n_beads, 3)
        for k in range(chunk.shape[0]):           # each replica separately
            xyz = np.array(chunk[k][::args.stride], dtype=np.float32) * scale
            tics.append(tica.transform(featurise(xyz, top, recipe))[:, :2])
    tics = np.concatenate(tics)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, tics)
    print(f"\nprojected    : {tics.shape[0]} frames")
    print(f"TIC 1 range  : {tics[:,0].min():.2f} .. {tics[:,0].max():.2f}")
    print(f"TIC 2 range  : {tics[:,1].min():.2f} .. {tics[:,1].max():.2f}")
    print(f"written      : {args.out}")


if __name__ == "__main__":
    main()
