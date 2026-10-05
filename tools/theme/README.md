# tools/theme — the thesis plotting style

```python
import sys; sys.path.insert(0, "/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/theme")
from thesis_style import apply_style, PALETTE, fit_panel, save_figure

apply_style("talk")            # BEFORE plt.subplots()
```

## The one rule

**Call `apply_style()` before `plt.subplots()`.**

Tick labels are rendered at *save* time from the rcParams current *then*, but
axis and title text is created when the axes are. Setting the style after
creating the figure gives you a figure whose ticks follow the new style and
whose labels follow the old one — which looks like the style silently not
working. This cost a debugging session: a font specimen showed no differences
between fonts because every panel's ticks rendered in whichever font was set
last.

## Modes

| mode | font | line | marker | dpi | usetex |
|---|---|---|---|---|---|
| `draft` | 10 | 1.5 | 4.5 | 110 | no |
| `talk` | 18 | 2.5 | 8.0 | 200 | no |
| `paper` | 9 | 1.25 | 4.0 | 300 | **yes** |

Only `usetex` costs real time — 1.45 s per figure against 0.24 s. Draft and talk
are the same speed, so use `talk` by default and switch to `paper` at the end.

`save_figure()` writes PNG only in every mode.

## Palette

A single plum→coral ramp ordered by lightness, so the four prior terms stay
distinguishable in greyscale:

```
bonds       #863A6F   deep plum      L* 36.4
angles      #975C8D   muted purple   L* 47.2
dihedral    #D989B5   pink           L* 66.5
non_bonded  #F87B92   coral          L* 66.7
data        #2E073F   dark plum      L*  9.7   filled markers, alpha 0.50
```

**Stale entries:** `prior_fit`, `coarse`, `fill` and `variant_1…5` are still
Okabe-Ito starter values and clash with this ramp. The variant colours are the
one group that genuinely needs categorical rather than sequential separation —
fix them before plotting a variant sweep.

Annotations currently inherit the term colour and are hard to read on the light
terms; they should use `PALETTE["muted"]`.

## `fit_panel()`

Draws scatter + fitted curve with the fit on top:

```python
ax.scatter(..., alpha=MARKER_ALPHA, linewidths=0, zorder=2)
ax.plot(x_curve, y_curve, "--", color=c, zorder=3)   # must read ON TOP
```

Dense data will otherwise hide the curve completely.

## Fonts

`make_static_nunito.py` cuts static instances from the variable font. The
variable `Nunito[wght].ttf` defaults to **ExtraLight** in matplotlib, which is
far too faint — so weights 400/600/700 are instanced out with
`fontTools.varLib.instancer` and the variable originals deleted. Re-run it if
the fonts are ever reinstalled.
