#!/usr/bin/env python3
"""How much of the reference landscape does a CG model actually reproduce?

Reports three numbers that are quotable in a thesis, rather than "it looks smaller":
  area coverage  -- fraction of the reference's populated bins the model ever visits
  mass coverage  -- fraction of the reference's EQUILIBRIUM PROBABILITY that lives
                    in the region the model visits (the number that matters: visiting
                    empty corners is worthless, missing the dominant basin is fatal)
  overlap        -- sum of min(p_ref, p_model), 1.0 = identical, 0.0 = disjoint
"""
import sys
import numpy as np

T = "/srv/data/itzeljem79/mlcg-tk/lambda/tica"
ref = np.load(f"{T}/lambda_both_lag100_ref_tics.npy")
w = np.load(f"{T}/lambda_both_lag100_msm_weights.npy"); w = w / w.sum()
sim = np.load(sys.argv[1])
label = sys.argv[2] if len(sys.argv) > 2 else "model"

bins = 100
rng = [[ref[:, 0].min(), ref[:, 0].max()], [ref[:, 1].min(), ref[:, 1].max()]]
Pref, xe, ye = np.histogram2d(ref[:, 0], ref[:, 1], bins=bins, range=rng, weights=w)
Psim, _, _ = np.histogram2d(sim[:, 0], sim[:, 1], bins=bins, range=rng)
Psim = Psim / Psim.sum()

ref_occ = Pref > 0
sim_occ = Psim > 0
area = sim_occ.sum() / ref_occ.sum()
mass = Pref[sim_occ].sum()
overlap = np.minimum(Pref, Psim).sum()
outside = Psim[~ref_occ].sum()

print(f"=== {label} vs all-atom reference ===")
print(f"frames compared        : {sim.shape[0]:,}")
print(f"reference bins occupied: {ref_occ.sum()}   model bins occupied: {sim_occ.sum()}")
print()
print(f"AREA coverage  : {100*area:6.2f}%   (of the reference's populated region)")
print(f"MASS coverage  : {100*mass:6.2f}%   (of the reference's equilibrium probability)")
print(f"OVERLAP        : {100*overlap:6.2f}%   (1.0 would be identical distributions)")
print(f"outside the reference entirely: {100*outside:5.2f}% of the model's own probability")
print()
print(f"mean TIC1  reference {float((ref[:,0]*w).sum()):+.3f}   {label} {sim[:,0].mean():+.3f}")
print(f"mean TIC2  reference {float((ref[:,1]*w).sum()):+.3f}   {label} {sim[:,1].mean():+.3f}")
print(f"std  TIC1  reference {float(np.sqrt((w*(ref[:,0]-(ref[:,0]*w).sum())**2).sum())):.3f}"
      f"   {label} {sim[:,0].std():.3f}")
