# Shared style for the paper's diagrams: drawn in inches at the page's text width (5.5 in),
# so they are included at 100% and every figure prints with the same font sizes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Polygon, Circle

NOTO = "/usr/share/fonts/truetype/noto/"
for f in ("NotoSans-Regular", "NotoSans-Bold", "NotoSans-Italic", "NotoSans-BoldItalic",
          "NotoSansMono-Regular", "NotoSansMono-Bold"):
    font_manager.fontManager.addfont(NOTO + f + ".ttf")
plt.rcParams.update({
    "font.family": "Noto Sans", "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Noto Sans",
    "mathtext.it": "Noto Sans:italic", "mathtext.bf": "Noto Sans:bold",
})

TEXT_W = 5.5                    # NeurIPS text width, inches
TITLE, BODY, NOTE = 9.0, 8.5, 7.8
INK, MUTED, ARROW = "#1f1f1f", "#4d5561", "#262626"
PAL = {"blue": ("#e1e8fa", "#3d5fc4"), "green": ("#e2f1da", "#56963a"), "red": ("#fbe3e1", "#c9463d"),
       "yellow": ("#fdf0cf", "#d29a22"), "teal": ("#d9f1ef", "#258b85"), "purple": ("#efe5fb", "#7a3fc2")}


def canvas(height):
    fig = plt.figure(figsize=(TEXT_W, height))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, TEXT_W); ax.set_ylim(0, height); ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, colour, lw=1.3, r=0.07, dashed=False):
    fc, ec = ("white", INK) if colour is None else PAL[colour]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=(0, (4, 2.5)) if dashed else "-"))
    return ec


def arrow(ax, pts, lw=1.5):
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=ARROW, lw=lw, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=lw, mutation_scale=10))


# ---- small flat icons ------------------------------------------------------
def ic_db(ax, cx, cy, s, c):
    for k in range(3):
        ax.add_patch(Rectangle((cx - 0.3 * s, cy - 0.34 * s + k * 0.2 * s), 0.6 * s, 0.2 * s, fc="white", ec=c, lw=1.0))
    ax.add_patch(Ellipse((cx, cy + 0.27 * s), 0.6 * s, 0.16 * s, fc=c, ec=c, lw=1.0))


def ic_funnel(ax, cx, cy, s, c):
    ax.add_patch(Polygon([(cx - 0.4 * s, cy + 0.34 * s), (cx + 0.4 * s, cy + 0.34 * s), (cx + 0.08 * s, cy - 0.02 * s),
                          (cx + 0.08 * s, cy - 0.38 * s), (cx - 0.08 * s, cy - 0.3 * s), (cx - 0.08 * s, cy - 0.02 * s)],
                         fc=c, ec=c, lw=1.0, alpha=0.9))


def ic_doc(ax, cx, cy, s, c):
    ax.add_patch(FancyBboxPatch((cx - 0.3 * s, cy - 0.4 * s), 0.6 * s, 0.8 * s, boxstyle="round,pad=0,rounding_size=0.02",
                                fc="white", ec=c, lw=1.0))
    for k in range(3):
        y = cy + 0.18 * s - k * 0.18 * s
        ax.plot([cx - 0.17 * s, cx + 0.17 * s], [y, y], color=c, lw=0.9)


def ic_book(ax, cx, cy, s, c):
    for sx in (-1, 1):
        ax.add_patch(Polygon([(cx, cy - 0.3 * s), (cx + sx * 0.44 * s, cy - 0.22 * s), (cx + sx * 0.44 * s, cy + 0.32 * s),
                              (cx, cy + 0.24 * s)], fc="white", ec=c, lw=1.0))


def ic_pairs(ax, cx, cy, s, c):
    pts = [(cx - 0.34 * s, cy + 0.28 * s), (cx + 0.34 * s, cy + 0.28 * s), (cx - 0.34 * s, cy - 0.28 * s), (cx + 0.34 * s, cy - 0.28 * s)]
    for p in pts:
        ax.plot([cx, p[0]], [cy, p[1]], color=c, lw=1.0)
    for p in pts + [(cx, cy)]:
        ax.add_patch(Circle(p, 0.09 * s, fc="white", ec=c, lw=1.0))


def ic_palette(ax, cx, cy, s, c):
    for k, col in enumerate(("#c9463d", "#d29a22", "#56963a", "#3d5fc4")):
        ax.add_patch(Rectangle((cx - 0.36 * s + k * 0.19 * s, cy - 0.3 * s), 0.15 * s, 0.6 * s, fc=col, ec="none"))


def ic_gear(ax, cx, cy, s, c):
    import numpy as np
    t = np.linspace(0, 2 * np.pi, 49)
    r = 0.34 * s * (1 + 0.18 * (np.cos(8 * t) > 0))
    ax.fill(cx + r * np.cos(t), cy + r * np.sin(t), color=c)
    ax.add_patch(Circle((cx, cy), 0.13 * s, fc="white", ec="none"))
