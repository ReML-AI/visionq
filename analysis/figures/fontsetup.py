# Register the system Inter font files with matplotlib (it is not found by family name).
import glob
from matplotlib import font_manager

for f in glob.glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    if "Display" not in f and "Italic" not in f:
        font_manager.fontManager.addfont(f)
