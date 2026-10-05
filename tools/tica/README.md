# tools/tica — free energy surfaces on the reference axes

Two scripts, one job each:

```
project_sim.py   mlcg-nvt_langevin output  ->  tics.npy   (n_frames, 2)
plot_fes.py      one or more tics.npy       ->  an N-panel figure
```

Splitting them matters: projection is slow and done once per simulation, while
plotting gets re-run constantly as you change labels, panels and colour scales.

## The rule that makes any of this meaningful

**Every surface must be drawn on the same axes.** TICA fitted on your own model's
trajectories would produce axes that mean something different from the reference's,
and the two pictures could not be compared at all. So `project_sim.py` takes a
*saved* TICA model as input and never fits one. The default is the group's
lag-100 reference model:

```
/group/ag_clementi_cmb/users/sayeg84/mlcg-proteins/bba_ca/bba_cadist_diheds_lag_100.pkl
```

Its reference TICs are frame-aligned with the BBA training data (3 652 333 both),
so it is the right projection for this system.

## Typical use

```bash
# 1. project a simulation  (once per simulation)
python project_sim.py \
  --sim-dir /srv/data/itzeljem79/mlcg-tk/bba/prioronly/prod \
  --out     /srv/data/itzeljem79/mlcg-tk/bba/prioronly/prod/tics.npy

# 2. plot, as many panels as you like
A=/group/ag_clementi_cmb/users/sayeg84/mlcg-proteins/bba_ca
python plot_fes.py --out fes.png --vmax 9.5 \
  --reference $A/bba_cadist_diheds_lag_100_ref_tics.npy \
  --weights   $A/bba_cadist_diheds_lag_100_msm_weights.npy \
  --panel "prior only:/srv/data/.../prioronly/prod/tics.npy" \
  --panel "epoch 100:/srv/data/.../ep100/tics.npy"
```

`--panel` is repeatable, so the same command scales from two panels to an epoch
sweep. It splits on the **last** colon, so labels may contain colons.

## Three traps these scripts close

**Units.** mlcg writes simulation coordinates in **Ångström**; the reference
features were built in **nanometres**. Getting this wrong raises no error — it
silently projects every frame somewhere meaningless. `project_sim.py` detects the
scale from the coordinate magnitudes and prints which unit it read.

**Colour scale.** deeptime auto-scales each panel independently, which makes two
surfaces look far more different than they are. `plot_fes.py` computes every
surface first, then draws them all on one scale. Use `--vmax` when one panel's
tail would otherwise flatten the others — the reference's reweighted tail reaches
~14.4, while a CG model may only reach ~5, so `--vmax 9.5` is often more readable.

**Axis limits.** Taken from the reference when given, so a model that explores
less of the space shows up as a smaller blob instead of being silently zoomed in
to fill the panel.

## Reweighting belongs to the reference only

`--weights` applies per-frame MSM weights and should be used **only** on the
all-atom reference, which came from adaptive sampling and therefore does not
carry equilibrium populations in its raw frame counts.

CG simulations are Langevin runs at equilibrium — their frames are already
Boltzmann-sampled by construction, so they are histogrammed raw. Reweighting them
would be wrong. (Whether they are *converged* is a separate question, which is
what the 100 spread-out starting structures address.)

## Cost, for planning an epoch sweep

Projection is cheap: 200 000 frames took well under a minute. The expensive part
is generating the trajectories — a prior-only run of 100 replicas × 200 000 steps
took ~29 minutes on CPU. A trained SchNet evaluates a network at every step, so
measure **one** epoch's simulation before planning a sweep over many.

Every training epoch is on disk (`save_top_k: -1`), so checkpoints can be
evaluated retrospectively, in any order, at any time — no need to hook anything
into the training loop.
