#!/usr/bin/env python3
"""Fit a reference TICA model on an adaptive-sampling dataset, by streaming.

Reads the local CA cache from `build_ca_cache.py`, featurises each segment, and
accumulates TICA covariances with `partial_fit`. Two modes:

  --lags A B C ...   scan several lag times in ONE pass and report, per lag, the
                     implied timescales and the cross-validated VAMP-2 score.
  --save-lag N       fit at that single lag on every segment, then project every
                     frame and save the model + TICs for the MSM step.

WHY STREAMING, NOT fit()
------------------------
The full feature matrix for lambda would be 7.85M frames x 3314 features x 4 B =
104 GB. Streaming never materialises it. Verified exact: streaming `partial_fit`
per segment against `fit()` on a list of segments agreed to 9.0e-17 (float64
round-off) with identical timescales, on deeptime 0.4.5.

WHY THE LAGGED PAIRS ARE BUILT HERE AND NOT HANDED TO deeptime
--------------------------------------------------------------
`TICA.partial_fit` takes an explicit (X_t, X_t+tau) tuple, so this script decides
which frames may be paired. That is the point: the dataset is an adaptive-sampling
tree and time continuity exists ONLY within one segment. Pairing across a boundary
is not a mild bias -- measured on synthetic data it moved the leading eigenvalue
24.5% and reordered the spectrum, an error larger than the largest true eigenvalue.
Every pair this script forms comes from inside one segment.

WHY THE RECIPE IS SAVED INSIDE THE PICKLE
-----------------------------------------
A free energy surface is comparable to another only if both are drawn on the same
axes, which means the simulation must be featurised *identically* to the reference.
Passing the recipe separately is how that silently drifts. So the saved file holds
the model AND the exact pairs, dihedral atoms, feature name and units, and
`project_sim.py` reads them from there rather than reconstructing them.

COST (measured, lambda: 80 beads, 15700 segments of ~501 frames)
----------------------------------------------------------------
Covariance work is O(d^2) and the distances dominate it completely:
    distances   d=3160  -> 101 min per full pass
    dihedrals   d= 154  -> under 1 min
    both        d=3314  ->  94 min        (same as distances; d^2 only +10%)
Featurisation is negligible (18 ms per segment, ~5 min total). So "both" is free
relative to "distances", and a lag scan should use --max-segments to subsample.

USAGE
-----
  # scan lags on a subsample, for the implied-timescale test
  python fit_reference_tica.py --cache <dir> --features both \
      --lags 20 50 100 160 250 --max-segments 2000

  # final model at the chosen lag, on everything
  python fit_reference_tica.py --cache <dir> --features both --save-lag 100 \
      --out-prefix /srv/data/.../lambda_tica_both
"""
import argparse
import json
import os
import pickle as pkl
from itertools import combinations

import numpy as np

FRAME_PS = 100.0  # lambda/bba charmm22star: xtcfreq 25000 x timestep 4 fs


def build_recipe(n_beads, features):
    pairs = list(combinations(range(n_beads), 2))
    dihedral_atoms = np.array([[k, k + 1, k + 2, k + 3] for k in range(n_beads - 3)])
    n_dist = len(pairs) if features in ("distances", "both") else 0
    n_dih = 2 * len(dihedral_atoms) if features in ("dihedrals", "both") else 0
    return {"features": features, "pairs": pairs, "dihedral_atoms": dihedral_atoms,
            "n_beads": n_beads, "units": "nanometre", "n_features": n_dist + n_dih}


def featurise(xyz_nm, top, recipe):
    """The one definition of the feature vector. project_sim.py must call THIS."""
    import mdtraj as md
    traj = md.Trajectory(xyz=xyz_nm, topology=top)
    parts = []
    if recipe["features"] in ("distances", "both"):
        parts.append(md.compute_distances(traj, recipe["pairs"], periodic=False))
    if recipe["features"] in ("dihedrals", "both"):
        h = md.compute_dihedrals(traj, recipe["dihedral_atoms"])
        parts += [np.sin(h), np.cos(h)]
    return np.concatenate(parts, axis=1)


def contiguous_runs(seg_xyz, min_bond_nm):
    """Split a segment at frames whose consecutive CA-CA distance is too short.

    Cis peptide frames show up at CA resolution as a short consecutive distance
    (cis ~0.28 nm vs trans ~0.38 nm, with an empty gap at 0.34). Dropping such a
    frame breaks time continuity, so the segment is split into maximal runs rather
    than having the hole stitched over.
    """
    if min_bond_nm <= 0:
        return [(0, seg_xyz.shape[0])]
    b = np.linalg.norm(seg_xyz[:, 1:, :] - seg_xyz[:, :-1, :], axis=2).min(axis=1)
    ok = b >= min_bond_nm
    runs, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(ok)))
    return runs


def iter_segments(coords, starts, idx, min_bond_nm):
    """Yield contiguous, time-continuous blocks of frames for the given segments."""
    for s in idx:
        # np.array() not asarray(): the cache is memmapped read-only and mdtraj's
        # Cython kernels require a writable buffer.
        seg = np.array(coords[starts[s]:starts[s + 1]], dtype=np.float32)
        for a, b in contiguous_runs(seg, min_bond_nm):
            if b - a >= 2:
                yield seg[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="directory from build_ca_cache.py")
    ap.add_argument("--topology", required=True, help="CG structure pdb (n_beads atoms)")
    ap.add_argument("--features", choices=("distances", "dihedrals", "both"), required=True)
    ap.add_argument("--lags", type=int, nargs="+", default=None, help="lag scan, in frames")
    ap.add_argument("--save-lag", type=int, default=None, help="fit and save at this one lag")
    ap.add_argument("--out-prefix", default=None, help="required with --save-lag")
    ap.add_argument("--max-segments", type=int, default=None, help="subsample segments (scan mode)")
    ap.add_argument("--train-frac", type=float, default=0.8, help="segment split for VAMP-2")
    ap.add_argument("--score-dim", type=int, default=2, help="components kept when scoring")
    ap.add_argument("--epsilon", type=float, default=1e-6,
                    help="deeptime TICA epsilon; raise it if 3160 collinear distances "
                         "make C_0 ill-conditioned")
    ap.add_argument("--min-bond-nm", type=float, default=0.0,
                    help="split segments at consecutive CA-CA below this (0.34 = the cis cut "
                         "used on the training data); 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if (args.lags is None) == (args.save_lag is None):
        raise SystemExit("give exactly one of --lags (scan) or --save-lag (final fit)")
    if args.save_lag is not None and not args.out_prefix:
        raise SystemExit("--save-lag needs --out-prefix")

    import mdtraj as md
    from deeptime.decomposition import TICA

    coords = np.load(os.path.join(args.cache, "ca_coords_nm.npy"), mmap_mode="r")
    starts = np.load(os.path.join(args.cache, "segment_starts.npy"))
    with open(os.path.join(args.cache, "manifest.json")) as f:
        manifest = json.load(f)
    n_seg = len(starts) - 1
    top = md.load(args.topology).topology
    if top.n_atoms != coords.shape[1]:
        raise SystemExit(f"topology has {top.n_atoms} beads, cache has {coords.shape[1]}")

    recipe = build_recipe(top.n_atoms, args.features)
    print(f"cache        : {args.cache}")
    print(f"segments     : {n_seg:,}   frames: {coords.shape[0]:,}   beads: {coords.shape[1]}")
    print(f"featurisation: {args.features}  ->  {recipe['n_features']} features")
    print(f"units        : {manifest.get('units')}   frame interval: {FRAME_PS:.0f} ps")
    if args.min_bond_nm > 0:
        print(f"cis split    : segments split where consecutive CA-CA < {args.min_bond_nm} nm")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_seg)
    if args.max_segments:
        order = order[:args.max_segments]
        print(f"subsampled   : {len(order):,} of {n_seg:,} segments")
    n_tr = int(len(order) * args.train_frac)
    tr_idx, te_idx = order[:n_tr], order[n_tr:]

    lags = args.lags if args.lags is not None else [args.save_lag]
    print(f"lags         : {lags}  (frames)   = {[l*FRAME_PS/1000 for l in lags]} ns")
    print()

    def stream_fit(idx, lag_list, label):
        est = {l: TICA(lagtime=l, dim=args.score_dim, epsilon=args.epsilon) for l in lag_list}
        n_blocks = n_used = 0
        for block in iter_segments(coords, starts, idx, args.min_bond_nm):
            F = featurise(block, top, recipe)
            n_blocks += 1
            for l in lag_list:
                if F.shape[0] > l:
                    est[l].partial_fit((F[:-l], F[l:]))
                    n_used += 1
            if n_blocks % 250 == 0:
                print(f"  [{label}] {n_blocks:,} blocks", flush=True)
        print(f"  [{label}] done: {n_blocks:,} blocks, {n_used:,} (block,lag) updates")
        return {l: e.fetch_model() for l, e in est.items()}

    m_tr = stream_fit(tr_idx, lags, "train")
    m_te = stream_fit(te_idx, lags, "test") if len(te_idx) else {}

    print()
    print(f"{'lag':>6s} {'lag/ns':>8s} {'ts1/frames':>11s} {'ts1/ns':>9s} {'ts2/ns':>9s} {'VAMP-2(CV)':>11s}")
    for l in lags:
        ts = m_tr[l].timescales()
        ts1 = ts[0] if len(ts) else float("nan")
        ts2 = ts[1] if len(ts) > 1 else float("nan")
        sc = m_tr[l].score(r=2, test_model=m_te[l]) if l in m_te else float("nan")
        print(f"{l:6d} {l*FRAME_PS/1000:8.1f} {ts1:11.1f} {ts1*FRAME_PS/1000:9.2f} "
              f"{ts2*FRAME_PS/1000:9.2f} {sc:11.5f}")

    if args.save_lag is None:
        print()
        print("scan only; nothing saved. Pick a lag where the timescales have flattened,")
        print("then rerun with --save-lag. Report timescales AND VAMP-2: they can disagree,")
        print("and if they do that disagreement is the result, not something to resolve by taste.")
        return

    lag = args.save_lag
    model = m_tr[lag]
    print()
    print(f"refitting on ALL {n_seg:,} segments at lag {lag} for the saved model ...")
    final = stream_fit(np.arange(n_seg), [lag], "all")[lag]

    print("projecting every frame ...")
    tics, seg_of = [], []
    for s in range(n_seg):
        seg = np.array(coords[starts[s]:starts[s + 1]], dtype=np.float32)
        for a, b in contiguous_runs(seg, args.min_bond_nm):
            if b - a < 2:
                continue
            t = final.transform(featurise(seg[a:b], top, recipe))[:, :2]
            tics.append(t)
            seg_of.append(np.full(t.shape[0], s, dtype=np.int32))
    tics = np.concatenate(tics)
    seg_of = np.concatenate(seg_of)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    with open(f"{args.out_prefix}_lag{lag}.pkl", "wb") as f:
        pkl.dump({"model": final, "recipe": recipe, "lag": lag, "frame_ps": FRAME_PS,
                  "cache": args.cache, "topology": args.topology,
                  "min_bond_nm": args.min_bond_nm}, f)
    np.save(f"{args.out_prefix}_lag{lag}_ref_tics.npy", tics)
    np.save(f"{args.out_prefix}_lag{lag}_ref_segment.npy", seg_of)

    print()
    print(f"model  : {args.out_prefix}_lag{lag}.pkl   (model + recipe embedded)")
    print(f"tics   : {args.out_prefix}_lag{lag}_ref_tics.npy  {tics.shape}")
    print(f"segment: {args.out_prefix}_lag{lag}_ref_segment.npy  (for per-segment MSM dtrajs)")
    print(f"TIC 1  : {tics[:,0].min():.2f} .. {tics[:,0].max():.2f}")
    print(f"TIC 2  : {tics[:,1].min():.2f} .. {tics[:,1].max():.2f}")
    print()
    print("NEXT: the raw density of these TICs is NOT a free energy surface -- this dataset")
    print("is adaptive sampling. Cluster, build an MSM at this lag, and use")
    print("compute_trajectory_weights before plotting with plot_fes.py --weights.")


if __name__ == "__main__":
    main()
