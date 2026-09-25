"""
tools/theme/thesis_style.py
Matplotlib theme for the CG force-field thesis.

Three modes, one palette:
    draft  - Nunito Regular + stixsans, 10 pt. Fast; for diagnostics.
    talk   - Nunito + stixsans, 18 pt, heavy. Matches the slide templates.
    paper  - real LaTeX (Computer Modern), 9 pt, 3.31" column. Matches the thesis.

IMPORTANT: call apply_style() BEFORE plt.subplots(). Tick labels are rendered at
save time from the current rcParams, so changing style afterwards gives a mix of
fonts (the body text changes, the tick numbers do not).

Nunito is installed per-user on seagull at ~/.local/share/fonts/nunito as STATIC
weights cut from the variable font by make_static_nunito.py (which lives beside
this file). The variable original defaulted to ExtraLight, which rendered far too
light - do not reinstall it.

Usage:
    import sys; sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/theme")
    from thesis_style import apply_style, PALETTE, AXIS_LABELS, term_color, save_figure
    apply_style("draft")

Structure follows the author's earlier paper theme.
Inspiration for the base recipe: https://www.steven-braun.com/blog/2021/matplotlib-viz/
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Palette
# -----------------------------------------------------------------------------
# Starter values only - chosen to be functional, not final. Two constraints:
#   (a) colour-vision safe (built on the Okabe-Ito qualitative set)
#   (b) the four term colours differ in LIGHTNESS as well as hue, so they stay
#       distinguishable in greyscale print
PALETTE = {
    # --- the four prior terms: permanent colours, used in every figure ---
    "bonds":       "#863A6F",   # deep plum      (darkest)
    "angles":      "#975C8D",   # muted purple
    "dihedral":    "#D989B5",   # pink
    "non_bonded":  "#F87B92",   # coral          (lightest)

    # --- the recurring "measurement vs model" pair ---
    "data":        "#2E073F",   # dark plum: filled markers, contrasts with the fit line
    "prior_fit":   "#0072B2",   # default curve colour; term_color() overrides per term

    # --- comparing prior variants ---
    "baseline":    "#4A4A4A",   # neutral grey: the reference is never a "colour"
    "variant_1":   "#0072B2",
    "variant_2":   "#E69F00",
    "variant_3":   "#009E73",
    "variant_4":   "#CC79A7",
    "variant_5":   "#D55E00",   # vermillion

    # --- resolutions, for the all-atom vs CG figures ---
    "all_atom":    "#9AA3AB",   # muted: it is context, not the subject
    "coarse":      "#D55E00",   # the beads are the subject

    # --- structural ---
    "grid":        "#E5E7EB",
    "spine":       "#374151",
    "muted":       "#9AA3AB",
    "fill":        "#0072B233",  # ~20% alpha, for error bands
}

TERM_ORDER = ["bonds", "angles", "dihedral", "non_bonded"]
VARIANT_CYCLE = ["baseline", "variant_1", "variant_2", "variant_3", "variant_4", "variant_5"]


def term_color(name: str) -> str:
    """Colour for a prior term. Accepts mlcg-tk's names, e.g. 'pseudo_ca_dihedral'."""
    key = name.lower()
    if "bond" in key and "non" not in key:
        return PALETTE["bonds"]
    if "angle" in key:
        return PALETTE["angles"]
    if "dihedral" in key or "torsion" in key:
        return PALETTE["dihedral"]
    if "non_bonded" in key or "nonbonded" in key or "repulsion" in key:
        return PALETTE["non_bonded"]
    return PALETTE["prior_fit"]


# -----------------------------------------------------------------------------
# 2. Widths
# -----------------------------------------------------------------------------
TEX_WIDTH_TWO_COLUMN = 3.31314   # IEEE / Phys. Rev. single column
TEX_WIDTH_SINGLE_COLUMN = 6.5    # full text width
SLIDE_WIDTH = 10.0               # 16:9 slide, comfortable

_MODE_WIDTH = {"draft": 6.0, "paper": TEX_WIDTH_TWO_COLUMN, "talk": SLIDE_WIDTH}
_MODE = {"current": "draft"}


# -----------------------------------------------------------------------------
# 3. Theme
# -----------------------------------------------------------------------------
def apply_style(mode: str = "draft", use_tex: bool = None, base_font_size: int = None):
    """
    Configure matplotlib for one of three output contexts.

    Parameters
    ----------
    mode
        'draft'  : Nunito Regular + stixsans, 10 pt, dpi 110, PNG. Fast.
        'talk'   : Nunito + stixsans, 18 pt, heavy lines, dpi 200.
        'paper'  : real LaTeX (Computer Modern), 9 pt, 3.31" column, dpi 300, PDF + PNG.
    use_tex
        Override the mode default (draft/talk: False, paper: True). LaTeX is
        typographically exact and matches the thesis, but adds a round-trip per
        figure - pass use_tex=False in paper mode while iterating.
    base_font_size
        Override the mode default.
    """
    if mode not in _MODE_WIDTH:
        raise ValueError(f"mode must be one of {list(_MODE_WIDTH)}")
    _MODE["current"] = mode

    defaults = {
        "draft": dict(font=10, lw=1.5, ms=4.5, dpi=110, axlw=0.8, tick=0.8, tex=False,
                      family="Nunito", mathset="stixsans"),
        "talk":  dict(font=18, lw=2.5, ms=8.0, dpi=200, axlw=1.2, tick=1.2, tex=False,
                      family="Nunito", mathset="stixsans"),
        "paper": dict(font=9,  lw=1.25, ms=4.0, dpi=300, axlw=0.6, tick=0.6, tex=True,
                      family="serif", mathset="cm"),
    }[mode]

    fs = base_font_size if base_font_size is not None else defaults["font"]
    tex = defaults["tex"] if use_tex is None else use_tex

    mpl.rcParams.update({
        # --- type ---
        # draft/talk: Nunito with sans math, matching the slide templates.
        # paper: hand everything to LaTeX so figures match the thesis exactly.
        #   (pass use_tex=False for fast iteration; LaTeX adds a round-trip per figure)
        "font.family": defaults["family"],
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.sans-serif": ["Nunito", "Lato", "DejaVu Sans"],
        "font.weight": "regular",
        "mathtext.fontset": defaults["mathset"],
        "text.usetex": tex,
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
        "legend.title_fontsize": fs - 1,

        # --- axes & grid ---
        "axes.linewidth": defaults["axlw"],
        "axes.edgecolor": PALETTE["spine"],
        "axes.labelcolor": PALETTE["spine"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "axes.prop_cycle": mpl.cycler(color=[PALETTE[k] for k in VARIANT_CYCLE]),

        # --- lines ---
        "lines.linewidth": defaults["lw"],
        "lines.markersize": defaults["ms"],

        # --- ticks ---
        "xtick.color": PALETTE["spine"],
        "ytick.color": PALETTE["spine"],
        "xtick.major.width": defaults["tick"],
        "ytick.major.width": defaults["tick"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        # --- output ---
        "figure.dpi": defaults["dpi"],
        "savefig.dpi": defaults["dpi"],
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


# -----------------------------------------------------------------------------
# 4. Sizing
# -----------------------------------------------------------------------------
def get_figure_dim(width_in: float = None, aspect_ratio: float = 0.75, scale: float = 1.0):
    """Figure size in inches. Defaults to the width appropriate for the active mode."""
    if width_in is None:
        width_in = _MODE_WIDTH[_MODE["current"]]
    w = width_in * scale
    return (w, w * aspect_ratio)


# -----------------------------------------------------------------------------
# 5. Recurring figure: measured points + fitted curve
# -----------------------------------------------------------------------------
MARKER_ALPHA = 0.50   # filled, no edge (see fit_panel)


def fit_panel(ax, x_data, y_data, x_curve=None, y_curve=None, term="prior_fit",
              data_label="data", fit_label="prior fit"):
    """The panel drawn hundreds of times in this project.

    Filled semi-transparent markers in PALETTE["data"] for what was measured; a
    dashed line in the TERM colour for the model. Fixed dark markers against a
    coloured fit keeps the contrast constant whatever the term.

    alpha 0.50 with no edge: where points pile up the fill darkens, so density is
    readable, while isolated points stay visible. (The fit is drawn at zorder 3,
    above the markers, so the line is legible regardless of marker opacity.)
    """
    c = term_color(term)
    ax.scatter(x_data, y_data, s=mpl.rcParams["lines.markersize"] ** 2 * 0.8,
               color=PALETTE["data"], alpha=MARKER_ALPHA, linewidths=0,
               zorder=2, label=data_label)
    if x_curve is not None:
        # zorder 3: the fit must read ON TOP, or dense data hides it entirely
        ax.plot(x_curve, y_curve, "--", color=c, zorder=3, label=fit_label)
    return ax


# -----------------------------------------------------------------------------
# 6. Legend & saving
# -----------------------------------------------------------------------------
def format_legend(ax, loc: str = "best", title: str = None):
    legend = ax.legend(loc=loc, title=title, frameon=True, fancybox=False,
                       edgecolor="#CCCCCC")
    if legend:
        legend.get_frame().set_linewidth(0.5)
    return legend


def save_figure(fig, filename: str, output_dir: str = "figures", formats: tuple = None):
    """Save in the formats appropriate for the active mode.

    draft -> png only (fast). paper -> pdf + png. talk -> png + pdf.
    """
    os.makedirs(output_dir, exist_ok=True)
    if formats is None:
        formats = ("png",) if _MODE["current"] == "draft" else ("pdf", "png")
    paths = []
    for ext in formats:
        p = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(p, pad_inches=0.02)
        paths.append(p)
    print(f"[saved] {'  '.join(paths)}")
    return paths


# -----------------------------------------------------------------------------
# 7. Labels - one place, so units are never wrong twice
# -----------------------------------------------------------------------------
AXIS_LABELS = {
    "distance":      r"$r$ ($\mathrm{\AA}$)",
    "bond_length":   r"bond length $r$ ($\mathrm{\AA}$)",
    "cos_angle":     r"$\cos\theta$",
    "angle":         r"$\theta$ (deg)",
    "dihedral":      r"$\phi$ (rad)",
    "free_energy":   r"free energy (kcal mol$^{-1}$)",
    "energy":        r"$U$ (kcal mol$^{-1}$)",
    "force":         r"$|F|$ (kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)",
    "delta_force":   r"$|\Delta F|$ (kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)",
    "rg":            r"$R_g$ ($\mathrm{\AA}$)",
    "rmsd":          r"RMSD ($\mathrm{\AA}$)",
    "time_ns":       r"$t$ (ns)",
    "probability":   r"$P$",
    "counts":        r"counts",
    "bin_occupancy": r"occupied bins",
}

TERM_LABELS = {
    "bonds":              "bonds",
    "angles":             "angles",
    "pseudo_ca_dihedral": r"CA pseudo-dihedral",
    "non_bonded":         "non-bonded",
}


# -----------------------------------------------------------------------------
# 8. Palette preview
# -----------------------------------------------------------------------------
def show_palette(path="palette_preview.png"):
    """Render the palette as swatches, so choices can be judged by eye."""
    groups = [
        ("prior terms", TERM_ORDER),
        ("measurement vs model", ["data", "prior_fit"]),
        ("variants", VARIANT_CYCLE),
        ("resolutions", ["all_atom", "coarse"]),
        ("structural", ["grid", "spine", "muted"]),
    ]
    fig, axs = plt.subplots(len(groups), 1, figsize=(8, 0.9 * len(groups)))
    for ax, (title, keys) in zip(axs, groups):
        for i, k in enumerate(keys):
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=PALETTE[k]))
            ax.text(i + 0.45, -0.35, k, ha="center", va="top", fontsize=7)
            ax.text(i + 0.45, 0.5, PALETTE[k], ha="center", va="center",
                    fontsize=6, color="white" if k not in ("grid", "angles") else "#333")
        ax.set_xlim(-0.1, max(len(k) for _, k in groups)); ax.set_ylim(-1.1, 1.1)
        ax.set_title(title, loc="left", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")
    return fig
