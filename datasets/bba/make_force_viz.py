#!/usr/bin/env python3
"""Draw the force vectors that live in the raw .npy files, at both resolutions.

Produces, for ONE frame:
  forces_allatom.pdb + forces_cg.pdb   the structures
  forces.py                            a PyMOL script drawing CGO arrows
"""
import numpy as np
import mdtraj as md

D = "/group/ag_clementi_cmb/projects/single_protein_datasets/charmm/raw_charmm22star_bba"
OUT = "/srv/data/itzeljem79/mlcg-tk/bba/out"
DEST = "/srv/data/itzeljem79/mlcg-tk/bba/viz"
STEM = "bba_0_e10s10_e8s8p0f443"        # same run used for the earlier export
FRAME = 0

x = np.asarray(np.load(f"{D}/coords_nowater/bba_coor_folding-{STEM}.npy", mmap_mode="r")[FRAME])
f = np.asarray(np.load(f"{D}/forces_nowater/bba_force_folding-{STEM}.npy", mmap_mode="r")[FRAME])
print(f"frame {FRAME}: coords {x.shape}, forces {f.shape}")

full = md.load(f"{D}/bba_50ns_0/structure.pdb")
top = full.atom_slice(full.topology.select("protein")).topology

# CG: the pipeline's own maps
Mx = np.load(f"{OUT}/1FME_cg_coord_map.npy")     # (28, 513)
Mf = np.load(f"{OUT}/1FME_cg_force_map.npy")
xc, fc = Mx @ x, Mf @ f
cg_top = md.load(f"{OUT}/1FME_cg_structure.pdb").topology

mag_a = np.linalg.norm(f, axis=1)
mag_c = np.linalg.norm(fc, axis=1)
print(f"  all-atom |F|: median {np.median(mag_a):6.1f}  max {mag_a.max():7.1f} kcal/mol/A")
print(f"  CG bead  |F|: median {np.median(mag_c):6.1f}  max {mag_c.max():7.1f} kcal/mol/A")

# scale so the median arrow is ~2.5 A long, same scale for both so they are comparable
SCALE = 2.5 / float(np.median(mag_a))
print(f"  arrow scale: {SCALE:.4f} A per kcal/mol/A")

md.Trajectory(x[None] / 10.0, top).save_pdb(f"{DEST}/forces_allatom.pdb")
md.Trajectory(xc[None] / 10.0, cg_top).save_pdb(f"{DEST}/forces_cg.pdb")


def arrows(pos, vec, name, rgb, scale):
    """CGO cylinder + cone per vector."""
    R, HR, HL = 0.06, 0.16, 0.5
    out = [f"{name} = ["]
    for p, v in zip(pos, vec):
        q = p + v * scale
        n = np.linalg.norm(q - p)
        if n < 1e-6:
            continue
        head = q - (q - p) / n * min(HL, 0.45 * n)
        out.append(
            f"  CYLINDER, {p[0]:.3f},{p[1]:.3f},{p[2]:.3f}, {head[0]:.3f},{head[1]:.3f},{head[2]:.3f},"
            f" {R}, {rgb[0]},{rgb[1]},{rgb[2]}, {rgb[0]},{rgb[1]},{rgb[2]},")
        out.append(
            f"  CONE, {head[0]:.3f},{head[1]:.3f},{head[2]:.3f}, {q[0]:.3f},{q[1]:.3f},{q[2]:.3f},"
            f" {HR}, 0.0, {rgb[0]},{rgb[1]},{rgb[2]}, {rgb[0]},{rgb[1]},{rgb[2]}, 1.0, 0.0,")
    out.append("]")
    return "\n".join(out)


script = f'''# Forces stored in the raw .npy files, one frame.
# Run with:  pymol forces.py
from pymol import cmd
from pymol.cgo import CYLINDER, CONE

cmd.load("forces_allatom.pdb", "aa")
cmd.load("forces_cg.pdb", "cg")
cmd.hide("everything")
cmd.show("lines", "aa"); cmd.color("grey50", "aa")
cmd.show("spheres", "cg"); cmd.set("sphere_scale", 0.35, "cg"); cmd.color("grey80", "cg")
cmd.bg_color("white")

{arrows(x, f, "F_atom", (0.20, 0.45, 0.85), SCALE)}

{arrows(xc, fc, "F_bead", (0.85, 0.20, 0.20), SCALE)}

cmd.load_cgo(F_atom, "force_allatom")
cmd.load_cgo(F_bead, "force_cg")
cmd.orient("aa")

# toggles:
#   disable force_allatom / enable force_allatom
#   disable aa            -> only the 28 beads and their forces
# Blue = the 513 raw atomic forces.  Red = the 28 mapped CG forces.
# Same scale, so lengths are directly comparable.
'''
open(f"{DEST}/forces.py", "w").write(script)
print(f"wrote {DEST}/forces.py  (+ two pdb files)")
