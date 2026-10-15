#!/usr/bin/env python3
"""Cluster the reference TICs, build an MSM, and produce equilibrium frame weights.

This is the step that turns a sampling density into a free energy surface. The
lambda dataset is adaptive sampling, so raw frame counts are NOT equilibrium
populations -- some regions were deliberately given 31 child trajectories. The
MSM re-weights each frame by its stationary probability, and only then does
-kT ln P mean anything.

WHY RUN BOUNDARIES ARE RE-DERIVED HERE
--------------------------------------
`*_ref_segment.npy` labels each frame with its SEGMENT id. Where the cis filter
split a segment into two runs, both runs carry the same id, so grouping by it
would silently concatenate them -- gluing together frames that are not adjacent
in time and reintroducing exactly the boundary error the whole pipeline avoids.
So the contiguous runs are recomputed from the cache with the same rule, and the
total is asserted against the TIC count before anything else happens.

TWO MODES
---------
  --lags A B C ...   MSM implied-timescale test. This is how the lag gets chosen
                     for lambda, because the TICA timescales never plateau within
                     the 25 ns ceiling that 50 ns segments allow.
  --save-lag N       build at that lag and write per-frame equilibrium weights.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_reference_tica import contiguous_runs

FRAME_PS = 100.0


def derive_runs(cache, min_bond_nm):
    """Recompute contiguous-run lengths from the cache, in cache order."""
    coords = np.load(os.path.join(cache, "ca_coords_nm.npy"), mmap_mode="r")
    starts = np.load(os.path.join(cache, "segment_starts.npy"))
    lengths = []
    for s in range(len(starts) - 1):
        seg = np.array(coords[starts[s]:starts[s + 1]], dtype=np.float32)
        for a, b in contiguous_runs(seg, min_bond_nm):
            if b - a >= 2:
                lengths.append(b - a)
    return np.array(lengths, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tics", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--min-bond-nm", type=float, default=0.34)
    ap.add_argument("--n-clusters", type=int, default=500)
    ap.add_argument("--cluster-sample", type=int, default=500000,
                    help="frames used to FIT kmeans; all frames are then assigned")
    ap.add_argument("--lags", type=int, nargs="+", default=None)
    ap.add_argument("--save-lag", type=int, default=None)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    if (args.lags is None) == (args.save_lag is None):
        raise SystemExit("give exactly one of --lags or --save-lag")
    if args.save_lag is not None and not args.out_prefix:
        raise SystemExit("--save-lag needs --out-prefix")

    from deeptime.clustering import KMeans
    from deeptime.markov.msm import MaximumLikelihoodMSM

    tics = np.load(args.tics)
    print(f"tics            : {tics.shape}")

    lengths = derive_runs(args.cache, args.min_bond_nm)
    print(f"contiguous runs : {len(lengths):,}   frames {lengths.sum():,}")
    if lengths.sum() != tics.shape[0]:
        raise SystemExit(
            f"run frames ({lengths.sum():,}) != TIC frames ({tics.shape[0]:,}); "
            "the cache or the cis cut does not match the one used for the projection")
    print("                  (matches the TIC count exactly)")
    print(f"run length      : min {lengths.min()}  max {lengths.max()}  median {int(np.median(lengths))}")

    rng = np.random.default_rng(args.seed)
    sub = rng.choice(tics.shape[0], size=min(args.cluster_sample, tics.shape[0]), replace=False)
    print(f"\nkmeans          : {args.n_clusters} centres, fitted on {len(sub):,} frames")
    clst = KMeans(n_clusters=args.n_clusters, init_strategy="kmeans++", max_iter=500,
                  fixed_seed=args.seed, n_jobs=8).fit_fetch(tics[sub])
    dtraj_all = clst.transform(tics).astype(np.int32)
    occ = np.bincount(dtraj_all, minlength=args.n_clusters)
    print(f"state occupancy : min {occ.min():,}  max {occ.max():,}  empty {int((occ == 0).sum())}")

    bounds = np.concatenate([[0], np.cumsum(lengths)])
    dtrajs = [dtraj_all[bounds[i]:bounds[i + 1]] for i in range(len(lengths))]

    lags = args.lags if args.lags is not None else [args.save_lag]
    print(f"\n{'lag':>6s} {'lag/ns':>8s} {'states':>8s} {'frac':>7s} "
          f"{'ts1/ns':>10s} {'ts2/ns':>10s} {'ts3/ns':>10s}")
    models = {}
    for lag in lags:
        usable = int((lengths > lag).sum())
        try:
            msm = MaximumLikelihoodMSM(lagtime=lag).fit_fetch(
                [d for d in dtrajs if len(d) > lag])
        except Exception as e:
            print(f"{lag:6d} {lag*FRAME_PS/1000:8.1f}   FAILED: {type(e).__name__}: {e}")
            continue
        models[lag] = msm
        cs = msm.count_model
        n_act = msm.n_states
        frac = cs.selected_state_fraction if hasattr(cs, "selected_state_fraction") else float("nan")
        ts = msm.timescales() * FRAME_PS / 1000.0
        print(f"{lag:6d} {lag*FRAME_PS/1000:8.1f} {n_act:8d} {frac:7.3f} "
              f"{ts[0]:10.1f} {ts[1]:10.1f} {ts[2]:10.1f}   (runs used {usable:,})")

    if args.save_lag is None:
        print("\nITS scan only; nothing written. Pick a lag where the MSM timescales have")
        print("flattened -- unlike TICA's, these can converge, because the Markov assumption")
        print("on discrete states is a different approximation from TICA's linear one.")
        return

    lag = args.save_lag
    msm = models[lag]
    print(f"\nweights at lag {lag} ...")
    keep = [i for i, d in enumerate(dtrajs) if len(d) > lag]
    w_list = msm.compute_trajectory_weights([dtrajs[i] for i in keep])

    weights = np.zeros(tics.shape[0], dtype=np.float64)
    for i, w in zip(keep, w_list):
        weights[bounds[i]:bounds[i + 1]] = w
    nz = weights > 0
    print(f"frames weighted : {nz.sum():,} of {len(weights):,} ({100*nz.mean():.2f}%)")
    print(f"weight sum      : {weights.sum():.6f}  (1.0 if normalised)")
    print(f"weight range    : {weights[nz].min():.3e} .. {weights[nz].max():.3e}")
    eff = weights.sum() ** 2 / (weights ** 2).sum()
    print(f"effective sample size: {eff:,.0f} frames "
          f"({100*eff/len(weights):.2f}% of the raw count)")

    np.save(f"{args.out_prefix}_lag{lag}_msm_weights.npy", weights)
    np.save(f"{args.out_prefix}_lag{lag}_dtraj.npy", dtraj_all)
    np.save(f"{args.out_prefix}_lag{lag}_cluster_centers.npy", clst.cluster_centers)
    print(f"\nwritten: {args.out_prefix}_lag{lag}_msm_weights.npy  (+ dtraj, centers)")
    print("These weights belong to the REFERENCE only. CG Langevin simulations are")
    print("equilibrium runs and must be histogrammed raw -- reweighting them would be wrong.")


if __name__ == "__main__":
    main()
