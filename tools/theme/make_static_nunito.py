"""Cut real static weights out of the Nunito variable font."""
import os, glob
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

D = os.path.expanduser("~/.local/share/fonts/nunito")
JOBS = [("Nunito[wght].ttf",        {400: "Regular", 600: "SemiBold", 700: "Bold"}),
        ("Nunito-Italic[wght].ttf", {400: "Italic",  700: "BoldItalic"})]

for src, weights in JOBS:
    path = os.path.join(D, src)
    for w, label in weights.items():
        f = TTFont(path)
        inst = instancer.instantiateVariableFont(f, {"wght": w}, updateFontNames=True,
                                                 inplace=True)
        out = os.path.join(D, f"Nunito-{label}.ttf")
        inst.save(out)
        print(f"  wght={w:<4} -> {os.path.basename(out)}")

# drop the variable originals so matplotlib cannot pick the ExtraLight default
for p in glob.glob(os.path.join(D, "Nunito*[[]wght[]].ttf")):
    os.remove(p); print(f"  removed {os.path.basename(p)}")
