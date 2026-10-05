# tools/viz — structures, trajectories and force vectors

Export scripts for looking at the system in PyMOL or VMD, rather than at plots.

| script | produces |
|---|---|
| `export_aa_vs_cg.py` | the same frames at both resolutions, to see the mapping |
| `find_folding_path.py` | locates a trajectory that actually folds |
| `build_folding_traj.py` | assembles that into a movie |
| `make_force_viz.py` | one frame's force vectors as PyMOL CGO arrows, both resolutions |

## Heads-up: paths are hardcoded

These were written against BBA and carry absolute paths in a block at the top:

```python
D    = "/group/ag_clementi_cmb/projects/single_protein_datasets/charmm/raw_charmm22star_bba"
OUT  = "/srv/data/itzeljem79/mlcg-tk/bba/out"
DEST = "/srv/data/itzeljem79/mlcg-tk/bba/viz"
STEM = "bba_0_e10s10_e8s8p0f443"
```

Nothing in the logic is BBA-specific — edit the block to point at another system.
They live under `tools/` rather than `datasets/bba/` for that reason.

## Two things worth knowing

**Both files are needed in PyMOL.** A `.pdb` carries the topology (which atoms,
how bonded); the `.dcd` carries only coordinates per frame. Load the pdb first,
then the dcd into the same object.

**Forces are drawn at one shared scale.** `make_force_viz.py` picks the scale so
the *median all-atom* arrow is ~2.5 Å and uses that same scale for the CG arrows,
so the two pictures are directly comparable. Changing it per-panel would make the
CG forces look artificially similar in magnitude to the all-atom ones.

**Frames are not aligned by default.** The raw trajectories carry real rotational
and translational diffusion. That is correct for force matching — the network
never sees absolute coordinates, so it is unaffected — but it makes a movie hard
to watch. Align for visualisation; do not align the training data.
