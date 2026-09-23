#!/usr/bin/env python3
"""Sanity-check a fitted prior model and the delta forces derived from it.

Handles both plain and mol_num_batches output. Memory-safe: everything is
memory-mapped and subsampled.

Run after every step 3, before packaging. Three levels:
  1. are the delta forces smaller than the true forces? (if not, something is broken)
  2. are any fitted coefficients absurd compared to their peers?
  3. which histograms are too sparse to fit safely?

Usage:
    python check_priors.py --save_dir out --name 1FME --prior_tag badn_poly_min_pair_4 --stride 10
"""
import argparse
import glob
import os
import pickle

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--save_dir", required=True)
p.add_argument("--name", required=True)
p.add_argument("--prior_tag", required=True)
p.add_argument("--stride", type=int, default=1, help="stride used in the stats config")
p.add_argument("--bins", type=int, default=100)
p.add_argument("--max_frames", type=int, default=200000,
               help="subsample to at most this many frames for levels 1 and 3")
a = p.parse_args()

J = lambda *x: os.path.join(a.save_dir, *x)
rms = lambda v: float(np.sqrt((v ** 2).mean()))
RULE = "=" * 72


def find(suffix):
    """Return the file list for <name>_<suffix>, batched or not."""
    plain = J(f"{a.name}_{suffix}")
    if os.path.exists(plain):
        return [plain]
    batched = sorted(glob.glob(J(f"{a.name}_batch_*_{suffix}")),
                     key=lambda s: int(s.split("_batch_")[1].split("_")[0]))
    return batched


def load(files, max_frames):
    """Memory-map, subsample evenly across all files, concatenate."""
    if not files:
        return None
    total = sum(np.load(f, mmap_mode="r").shape[0] for f in files)
    step = max(1, total // max_frames)
    return np.concatenate([np.asarray(np.load(f, mmap_mode="r")[::step]) for f in files], 0)


# ------------------------------------------------------------------ level 1
print(RULE)
print("LEVEL 1  delta forces vs true CG forces")
print(RULE)
ff = find("cg_forces.npy")
df = find(f"{a.prior_tag}_delta_forces.npy")
print(f"  force files: {len(ff)}   delta-force files: {len(df)}")
if ff and df:
    F = load(ff, a.max_frames)
    D = load(df, a.max_frames)
    if F.shape != D.shape:
        print(f"  !! shape mismatch {F.shape} vs {D.shape} — cannot compare")
    else:
        P = F - D
        ratio = rms(D) / rms(F)
        print(f"  sampled {F.shape[0]} frames")
        print("  RMS |F_CG|     = %10.2f kcal/mol/A" % rms(F))
        print("  RMS |F_prior|  = %10.2f" % rms(P))
        print("  RMS |dF|       = %10.2f" % rms(D))
        verdict = "OK" if ratio < 1 else "*** BAD: priors ADD force instead of removing it"
        print("  ratio dF/F     = %10.2f   -> %s" % (ratio, verdict))
        mag = np.sqrt((P ** 2).sum(-1)).max(1)
        print("  per-frame max |F_prior|: " + "  ".join(
            "%g%%=%.0f" % (q, np.percentile(mag, q)) for q in (50, 90, 99, 99.9, 100)))
        if np.percentile(mag, 100) > 50 * np.percentile(mag, 50):
            print("    -> heavy tail: a few terms are pathological, not a global scale error")
        del F, D, P
else:
    print("  (delta forces not produced yet — run step 3 first)")

# ------------------------------------------------------------------ level 2
print()
print(RULE)
print("LEVEL 2  fitted coefficients (only the type tuples this molecule uses)")
print(RULE)
model = torch.load(J(f"{a.prior_tag}_prior_model.pt"), weights_only=False)
emb = np.load(J(f"{a.name}_cg_embeds.npy"))
nls = pickle.load(open(J(f"{a.name}_prior_nls_{a.prior_tag}.pkl"), "rb"))

for term, nl in nls.items():
    if term not in model.models:
        continue
    mod = model.models[term].model
    im = np.asarray(nl["index_mapping"])
    order = im.shape[0]
    idx = tuple(emb[im[i]] for i in range(order))
    for pname, t in list(mod.named_buffers()) + list(mod.named_parameters()):
        v = t.detach().numpy()
        vals = v[(slice(None),) + idx] if v.ndim > order else v[idx]
        mx = float(np.abs(vals).max())
        med = float(np.median(np.abs(vals)))
        blown = med > 0 and mx > 1e3 * med
        print("  %-20s %-6s max=%12.2f  median=%10.3f%s"
              % (term, pname, mx, med, "   *** DIVERGED" if blown else ""))
        if blown:
            per = np.abs(vals).reshape(-1, im.shape[1]).max(0)
            bad = np.argsort(per)[::-1][:3]
            print("      worst instances: " + ", ".join(
                "#%d(max=%.3g)" % (j, per[j]) for j in bad))

# ------------------------------------------------------------------ level 3
print()
print(RULE)
print("LEVEL 3  histogram occupancy per instance (%d bins)" % a.bins)
print(RULE)
x = load(find("cg_coords.npy"), a.max_frames)
print(f"  sampled {x.shape[0]} frames from {len(find('cg_coords.npy'))} file(s)")


def feature(x, im):
    """Internal coordinate for a neighbour list of order 2, 3 or 4."""
    order = im.shape[0]
    if order == 2:
        return np.linalg.norm(x[:, im[0]] - x[:, im[1]], axis=-1), None
    if order == 3:
        u = x[:, im[0]] - x[:, im[1]]
        v = x[:, im[2]] - x[:, im[1]]
        cs = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        return cs, (-1.0, 1.0)
    A, B, C, Dd = (x[:, im[i]] for i in range(4))
    b0, b1, b2 = B - A, C - B, Dd - C
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1, axis=-1, keepdims=True))
    return np.arctan2((m * n2).sum(-1), (n1 * n2).sum(-1)), (-np.pi, np.pi)


for term, nl in nls.items():
    im = np.asarray(nl["index_mapping"])
    f, rng = feature(x, im)
    if rng is None:
        rng = (float(f.min()), float(f.max()))
    occ = np.array([(np.histogram(f[:, j], bins=a.bins, range=rng)[0] > 0).sum()
                    for j in range(im.shape[1])])
    sparse = occ.min() < 0.4 * a.bins
    print("  %-20s occupied bins: min=%3d median=%3d max=%3d  of %d%s"
          % (term, occ.min(), int(np.median(occ)), occ.max(), a.bins,
             "   *** sparse -> fit may diverge" if sparse else ""))
    if sparse:
        bad = np.argsort(occ)[:3]
        print("      worst instances: " + ", ".join("#%d(%d bins)" % (j, occ[j]) for j in bad))
