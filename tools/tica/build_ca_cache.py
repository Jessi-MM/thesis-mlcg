#!/usr/bin/env python3
"""Build a local CA-only cache of an adaptive-sampling dataset, one segment per raw file.

Reads the raw all-atom `coords_nowater/*.npy`, keeps only the CA atoms, converts
Angstrom -> nanometre, and writes ONE flat float32 array plus the segment
boundaries. Everything downstream (featurisation, TICA, MSM) then reads this
local cache and never touches /group again.

WHY A CACHE AT ALL
------------------
Three separate reasons, any one of which would justify it:

1. SEGMENT BOUNDARIES ARE THE WHOLE BALLGAME. This dataset is an adaptive-sampling
   tree: each raw file is an independent 501-frame segment spawned from some parent
   segment at some frame. Time continuity exists ONLY within a file. Fitting TICA
   across a boundary is not a small bias -- measured on synthetic data, ignoring
   boundaries moved the leading eigenvalue by 24.5% and reordered the spectrum.
   One raw file = one trajectory is the only layout that makes that impossible to
   get wrong, and the packaged `*_cg_coords.npy` can NOT give it: the cis and
   corrupt-force filtering punched holes that do not align with file boundaries.

2. /group IS SLOW AND NOT ALWAYS THERE. Measured 11 MB/s, so one pass over the
   ~125 GB of coordinates is ~3 h single-threaded. It also went down for a whole
   day on 2026-10-06 and took a pipeline step with it. Pay that cost once.

3. THE CACHE IS 7.6 GB. 15700 x 501 x 80 x 3 float32. Trivially re-readable, so
   the three featurisations under test cost nothing extra to compare.

THE SOLVENT TRAP
----------------
`lambda_aa_structure.pdb` has 32236 atoms (protein + water) while the raw .npy
files have 1256 (protein only). Selecting CA from the un-stripped topology happens
to give the right indices here -- water is appended AFTER the protein -- but that
is luck, not design. This script calls remove_solvent() and then ASSERTS that the
atom count matches the raw files, so the day a dataset interleaves solvent it
fails loudly instead of silently featurising water.

USAGE
-----
  python build_ca_cache.py \
      --raw-dir  /group/.../raw_charmm22star_lambda/coords_nowater \
      --pdb      /storage/.../datasets/lambda/lambda_aa_structure.pdb \
      --out-dir  /srv/data/itzeljem79/mlcg-tk/lambda/tica_cache \
      --workers  8
"""
import argparse
import json
import os
from glob import glob
from multiprocessing import Pool

import numpy as np

_CTX = {}


def _init(pdb, n_expected):
    import mdtraj as md
    t = md.load(pdb).remove_solvent()
    if t.topology.n_atoms != n_expected:
        raise SystemExit(
            f"topology has {t.topology.n_atoms} atoms after remove_solvent() but the raw "
            f"files have {n_expected}; refusing to guess an atom mapping")
    _CTX["ca"] = t.topology.select("name CA")


def _one(fn):
    """Load one raw segment, keep CA, Angstrom -> nm, float32."""
    a = np.load(fn)
    return os.path.basename(fn), (a[:, _CTX["ca"], :] / 10.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="directory of raw all-atom *.npy segments")
    ap.add_argument("--pdb", required=True, help="all-atom topology pdb")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel readers; NFS single-stream is ~11 MB/s so this matters a lot")
    ap.add_argument("--limit", type=int, default=None, help="only the first N segments (for testing)")
    args = ap.parse_args()

    fns = sorted(glob(os.path.join(args.raw_dir, "*.npy")))
    if args.limit:
        fns = fns[:args.limit]
    if not fns:
        raise SystemExit(f"no *.npy in {args.raw_dir}")

    probe = np.load(fns[0], mmap_mode="r")
    n_atoms_raw = probe.shape[1]
    mag = float(np.abs(np.load(fns[0])).mean())
    unit = "ANGSTROM" if mag > 2.0 else "nanometre"
    print(f"segments        : {len(fns)}")
    print(f"raw shape       : {probe.shape}  ({n_atoms_raw} atoms)")
    print(f"mean |coord|    : {mag:.2f}  -> read as {unit}")
    if mag <= 2.0:
        raise SystemExit("coordinates look like nanometres already; check the dataset before caching")

    os.makedirs(args.out_dir, exist_ok=True)

    names, lengths, chunks = [], [], []
    with Pool(args.workers, initializer=_init, initargs=(args.pdb, n_atoms_raw)) as pool:
        for i, (name, arr) in enumerate(pool.imap(_one, fns, chunksize=4)):
            names.append(name)
            lengths.append(arr.shape[0])
            chunks.append(arr)
            if (i + 1) % 500 == 0:
                print(f"  {i+1:6d}/{len(fns)} segments, {sum(lengths):,} frames", flush=True)

    coords = np.concatenate(chunks, axis=0)
    del chunks
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    np.save(os.path.join(args.out_dir, "ca_coords_nm.npy"), coords)
    np.save(os.path.join(args.out_dir, "segment_starts.npy"), starts)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"raw_dir": args.raw_dir, "pdb": args.pdb, "units": "nanometre",
                   "n_segments": len(names), "n_frames": int(coords.shape[0]),
                   "n_beads": int(coords.shape[2]), "segment_names": names,
                   "segment_lengths": [int(x) for x in lengths]}, f, indent=1)

    uniq = sorted(set(lengths))
    shown = uniq if len(uniq) <= 5 else str(uniq[:5]) + " ..."
    print()
    print(f"cache written   : {args.out_dir}")
    print(f"coords          : {coords.shape} {coords.dtype}  ({coords.nbytes/1e9:.2f} GB)")
    print(f"segment lengths : {shown}")
    print(f"total frames    : {coords.shape[0]:,}")


if __name__ == "__main__":
    main()
