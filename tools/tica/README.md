# tools/tica — free energy surfaces on reference axes

Two stages live here:

```
STAGE A   build a reference from all-atom data      (new, for lambda)
          raw AA trajectories -> TICA model -> MSM -> equilibrium weights

STAGE B   compare a CG model against that reference (was already here)
          mlcg-nvt_langevin output -> project onto saved axes -> FES panels
```

For **bba** only stage B was ever needed — Aldo had already built the reference
model. For **lambda** no reference existed anywhere in the group, so stage A had
to be built from scratch. That is what most of this folder now is.

## The rule that makes any of this meaningful

**Every surface must be drawn on the same axes.** TICA fitted on your own model's
trajectories would produce axes that mean something different from the
reference's, and the two pictures could not be compared at all. So
`project_sim.py` takes a *saved* TICA model as input and never fits one.

---

## Scripts

### Stage A — building a reference

| script | does |
|---|---|
| `build_ca_cache.py` | raw AA trajectories on /group -> local CA-only cache (nm, float32) plus the segment boundaries |
| `fit_reference_tica.py` | cache -> TICA model + per-frame TICs. `--lags` scans several lags, `--save-lag` fits the final model |
| `build_msm.py` | TICs -> 500 k-means states -> MSM -> **equilibrium weights**. `--lags` runs the MSM implied-timescale test |
| `validate_msm.py` | the four checks: spectral gap, PCCA+, Chapman-Kolmogorov, split-half |
| `make_sim_ensemble.py` | 100 starting structures spread over the FES, for `mlcg-tk-gen_sim_input` |

### Stage B — comparing a CG model

| script | does |
|---|---|
| `project_sim.py` | simulation chunks -> `(n_frames, 2)` TICs on the reference axes |
| `plot_fes.py` | one or more tics.npy -> an N-panel figure, shared colour scale and limits |
| `overlap.py` | area / mass / overlap coverage against the reference — quotable numbers |
| `sim_observables.py` | helicity and radius of gyration per model |

### Diagnostics and figures

| script | does |
|---|---|
| `interpret_tics.py` | what TIC 1 and TIC 2 physically measure (found: helicity, not compactness) |
| `reweighted_stats.py` | what the MSM reweighting changed |
| `plot_scan.py` | featurisation comparison figure |
| `plot_its.py` | TICA vs MSM implied timescales |
| `plot_tic_space.py` | sampling density and compactness in TIC space |
| `plot_metastable.py` | the FES with its metastable states marked |
| `run_scan.sh` | driver: three featurisations x five lags |

---

## Reference files for lambda

Everything under `/srv/data/itzeljem79/mlcg-tk/lambda/`. **These four are the
whole reference** — the only things stage B needs:

| file | size | used by |
|---|---|---|
| `tica/lambda_both_lag100.pkl` | 351 MB | `project_sim.py --tica` |
| `tica/lambda_both_lag100_ref_tics.npy` | 126 MB | `plot_fes.py --reference` |
| `tica/lambda_both_lag100_msm_weights.npy` | 63 MB | `plot_fes.py --weights` |
| `out/lambda_cg_structure.pdb` | small | `project_sim.py --topology` |

Supporting, not needed for comparison:

| file | why it exists |
|---|---|
| `tica/..._dtraj.npy`, `..._cluster_centers.npy` | the 500 MSM states — to rebuild the MSM or redo PCCA+ |
| `tica/..._ref_segment.npy` | superseded; `build_msm.py` re-derives boundaries instead |
| `tica/lambda_rg_per_frame.npy` | cached Rg so scripts stop recomputing it |
| `tica_cache/` (7.5 GB) | only needed to **refit** TICA; regenerable in ~3.4 h |

The lag is in every filename on purpose: refitting at another lag cannot silently
overwrite these, and you can always tell which lag a result came from.

Per simulation you get `tics.npy` (all frames) and `tics_equilibrated.npy`
(second half). **Use the equilibrated one** — CG runs start from spread-out
structures and need time to find their own equilibrium.

---

## Comparing a new model: the whole loop

```bash
# 1. merge the checkpoint with the prior   (--out is a FILE path, despite --help)
mlcg-combine_model \
  --ckpt  .../ckpt/epoch=NN-validation_loss=XXX.ckpt \
  --prior /srv/data/itzeljem79/mlcg-tk/lambda/out/badn_poly_min_pair_4_prior_model.pt \
  --out   /srv/data/itzeljem79/mlcg-tk/lambda/trained_sims/epNN/model_with_prior.pt

# 2. simulate
mlcg-nvt_langevin --config datasets/lambda/sim/trained/sim_epNN.yaml

# 3. project onto the reference axes
python project_sim.py \
  --sim-dir  /srv/data/itzeljem79/mlcg-tk/lambda/trained_sims/epNN/prod \
  --tica     /srv/data/itzeljem79/mlcg-tk/lambda/tica/lambda_both_lag100.pkl \
  --topology /srv/data/itzeljem79/mlcg-tk/lambda/out/lambda_cg_structure.pdb \
  --out      /srv/data/itzeljem79/mlcg-tk/lambda/trained_sims/epNN/prod/tics.npy

# 4. drop the first half (equilibration), then plot
T=/srv/data/itzeljem79/mlcg-tk/lambda/tica
python plot_fes.py --out fes.png --theme --vmax 12 \
  --reference $T/lambda_both_lag100_ref_tics.npy \
  --weights   $T/lambda_both_lag100_msm_weights.npy \
  --panel "prior only:.../prioronly/prod/tics_equilibrated.npy" \
  --panel "epoch NN:.../trained_sims/epNN/prod/tics_equilibrated.npy"

# 5. numbers to go with the picture
python overlap.py .../trained_sims/epNN/prod/tics_equilibrated.npy "epoch NN"
```

`--panel` is repeatable, so the same command scales from two panels to an epoch
sweep. It splits on the **last** colon, so labels may contain colons.

**Always run a prior-only simulation first.** It tests the whole chain while
nothing is at stake, and it is the control the trained model is read against —
without it you cannot say what the network added.

---

## Traps these scripts close

**Units.** mlcg writes simulation coordinates in **Angstrom**; the reference
features were built in **nanometres**. Getting this wrong raises no error — it
silently projects every frame somewhere meaningless. `project_sim.py` detects the
scale from the coordinate magnitudes and prints which unit it read.

**Featurisation recipe.** Projecting with a different feature vector than the
reference was fitted with is equally silent and equally fatal. Models written by
`fit_reference_tica.py` therefore carry their own recipe inside the pickle —
which pairs, which dihedral quartets, which feature set, which units — and
`project_sim.py` reads it rather than reconstructing one. It refuses outright if
the topology's bead count disagrees. Older pickles without a recipe (Aldo's bba
one) fall back to the legacy all-pairs reconstruction, so bba still works.

**Trajectory boundaries.** The all-atom data is an adaptive-sampling tree: time
runs forwards inside one 501-frame file and has no meaning across files. Pairing
frames across a boundary is not a small bias — measured on synthetic data it
moved the leading eigenvalue 24.5% and reordered the spectrum. The cache stores
the boundaries, and `build_msm.py` re-derives them and asserts the total before
doing anything.

**Colour scale.** deeptime auto-scales each panel independently, which makes two
surfaces look far more different than they are. `plot_fes.py` computes every
surface first, then draws them on one scale. Use `--vmax` when one panel's tail
would flatten the others — the lambda reference reaches 17.2 kT while a CG model
may only reach 5, so `--vmax 12` is more readable.

**Axis limits.** Taken from the reference when given, so a model that explores
less of the space shows up as a smaller blob instead of being silently zoomed in.

**Reweighting belongs to the reference only.** `--weights` applies per-frame MSM
weights, and the all-atom data came from adaptive sampling so its raw frame
counts are not equilibrium populations. CG simulations are Langevin runs at
equilibrium and are histogrammed raw — reweighting them would be wrong. Whether
they are *converged* is a separate question, which the 100 spread-out starting
structures and the equilibrated-half rule address.

---

## Cost, for planning

| step | lambda (80 beads) | bba (28 beads) |
|---|---|---|
| build the CA cache | ~3.4 h (one time, /group is 11 MB/s) | — |
| one TICA pass, all segments | ~100 min | minutes |
| MSM + weights | ~3 min | fast |
| prior-only simulation | 43 min, GPU | ~29 min, CPU |
| trained SchNet simulation | **~2 h 10 min**, GPU | — |
| projection | under a minute per 200k frames | same |

TICA covariance work scales as features², and lambda has 3,314 features against
bba's 428 — about 60x the arithmetic per frame. The feature matrix itself is
never built: `fit_reference_tica.py` streams segment by segment, so the 104 GB it
would occupy never exists.

A trained SchNet evaluates a network at every step, so **measure one epoch's
simulation before planning a sweep**. Every training epoch is on disk
(`save_top_k: -1`), so checkpoints can be evaluated retrospectively in any order.
