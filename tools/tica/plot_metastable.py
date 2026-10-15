#!/usr/bin/env python3
"""The FES with its two metastable states marked.

Answers the natural objection "I only see one big dark region": the two states
are there, but the barrier between them is low, so they merge visually on a
contour plot. PCCA+ separates them from the dynamics rather than from the picture.
"""
import sys
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/theme")
sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/tica")
import numpy as np
from fit_reference_tica import contiguous_runs
from deeptime.markov.msm import MaximumLikelihoodMSM
from thesis_style import apply_style, PALETTE, save_figure

T = "/srv/data/itzeljem79/mlcg-tk/lambda/tica"
CACHE = "/srv/data/itzeljem79/mlcg-tk/lambda/tica_cache"
LAG, MIN_BOND = 100, 0.34

tics = np.load(f"{T}/lambda_both_lag100_ref_tics.npy")
w = np.load(f"{T}/lambda_both_lag100_msm_weights.npy"); w = w / w.sum()
dtraj_all = np.load(f"{T}/lambda_both_lag100_dtraj.npy")
centers = np.load(f"{T}/lambda_both_lag100_cluster_centers.npy")
rg = np.load(f"{T}/lambda_rg_per_frame.npy")

coords = np.load(f"{CACHE}/ca_coords_nm.npy", mmap_mode="r")
starts = np.load(f"{CACHE}/segment_starts.npy")
lengths = []
for s in range(len(starts) - 1):
    seg = np.array(coords[starts[s]:starts[s + 1]], dtype=np.float32)
    for a, b in contiguous_runs(seg, MIN_BOND):
        if b - a >= 2:
            lengths.append(b - a)
bounds = np.concatenate([[0], np.cumsum(lengths)])
dtrajs = [dtraj_all[bounds[i]:bounds[i + 1]] for i in range(len(lengths))]

msm = MaximumLikelihoodMSM(lagtime=LAG).fit_fetch([d for d in dtrajs if len(d) > LAG])
pcca = msm.pcca(2)
sym = msm.count_model.state_symbols
state_of_micro = np.full(centers.shape[0], -1, dtype=int)
state_of_micro[sym] = pcca.assignments
frame_state = state_of_micro[dtraj_all]

apply_style("talk")
import matplotlib.pyplot as plt

bins = 150
rngs = [[tics[:, 0].min(), tics[:, 0].max()], [tics[:, 1].min(), tics[:, 1].max()]]
Hw, xe, ye = np.histogram2d(tics[:, 0], tics[:, 1], bins=bins, range=rngs, weights=w)
with np.errstate(divide="ignore"):
    F = -np.log(np.where(Hw > 0, Hw, np.nan))
F -= np.nanmin(F)

fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.6))
ext = [xe[0], xe[-1], ye[0], ye[-1]]

cf = axs[0].contourf(F.T, levels=np.linspace(0, 12, 13), extent=ext, cmap="jet", extend="max")
cb = fig.colorbar(cf, ax=axs[0]); cb.set_label("free energy (kT)", fontsize=13)
axs[0].set_title("Reweighted FES — looks like one broad basin", fontsize=14)

cols = [PALETTE["variant_5"], PALETTE["variant_1"]]
for m in (0, 1):
    sel = state_of_micro == m
    pop = msm.stationary_distribution[pcca.assignments == m].sum()
    mean_rg = np.average(rg[frame_state == m], weights=w[frame_state == m])
    axs[1].scatter(centers[sel, 0], centers[sel, 1], s=42, c=cols[m], alpha=0.85,
                   edgecolors="none",
                   label=f"state {m}: {100*pop:.1f}%,  Rg {mean_rg:.1f} Å")
axs[1].contour(F.T, levels=np.linspace(0, 12, 7), extent=ext,
               colors=PALETTE["spine"], linewidths=0.6, alpha=0.45)
axs[1].set_title("PCCA+ on the dynamics — it is two states", fontsize=14)
axs[1].legend(loc="upper left", fontsize=12, frameon=False)

for ax in axs:
    ax.set_xlabel("TIC 1"); ax.set_ylabel("TIC 2")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

fig.suptitle("Lambda repressor at 350 K: the two metastable states are real, "
             "the barrier between them is just low", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save_figure(fig, "lambda_metastable_states", output_dir=f"{T}/figures", formats=("png",))

print()
for m in (0, 1):
    sel = frame_state == m
    ww = w[sel] / w[sel].sum()
    print("state %d: weighted population %5.1f%%   mean Rg %5.2f A   mean TIC1 %+.2f"
          % (m, 100 * w[sel].sum(), float((rg[sel] * ww).sum()), float((tics[sel, 0] * ww).sum())))
print("\nmean first passage times (ns):")
mfpt01 = msm.mfpt(np.where(pcca.assignments == 0)[0], np.where(pcca.assignments == 1)[0]) * 0.1
mfpt10 = msm.mfpt(np.where(pcca.assignments == 1)[0], np.where(pcca.assignments == 0)[0]) * 0.1
print("  state 0 -> state 1 : %.1f" % mfpt01)
print("  state 1 -> state 0 : %.1f" % mfpt10)
