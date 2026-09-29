# Redraw of figs/DPO-dataset.png. Changes vs. the original: "questions" instead of "records",
# reference_pair (as in Appendix D), a note on why 3,911 data points yield 4,524 questions,
# and the pair constructor placed after the dataset (pairs are built from questions).
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fontsetup  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, Rectangle, Circle

plt.rcParams.update({"font.family": "Inter", "pdf.fonttype": 42})
GREEN = ("#dcefd0", "#6aa84f")
BLUE = ("#d3dbf6", "#5b6fd6")
ORANGE = ("#fde3a8", "#dba23a")
INK, MUTED = "#222222", "#4a5a6a"

fig, ax = plt.subplots(figsize=(10.5, 6.6))
ax.set_xlim(0, 16.5); ax.set_ylim(0, 10.4); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.15, 0.15), 16.2, 10.1, boxstyle="round,pad=0,rounding_size=0.35",
                            fc="white", ec=INK, lw=1.6, ls=(0, (5, 3))))
ax.plot([0.15, 16.35], [6.35, 6.35], color=INK, lw=1.2, ls=(0, (5, 3)))


def box(x, y, w, h, colors, lines, icon=None, fs=14):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=fc, ec=ec, lw=1.8))
    left = x + (1.05 if icon else 0)  # text is centred in the space right of the icon
    ax.text((left + x + w) / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)
    if icon:
        icon(x + 0.52, y + h / 2)


def database(cx, cy, s=0.34):
    for k in range(3):
        y = cy - 0.3 + k * 0.24
        ax.add_patch(Rectangle((cx - s, y), 2 * s, 0.24, fc="white", ec=INK, lw=1.3))
    ax.add_patch(Ellipse((cx, cy + 0.42), 2 * s, 0.22, fc="white", ec=INK, lw=1.3))


def funnel(cx, cy):
    ax.add_patch(Polygon([(cx - 0.36, cy + 0.32), (cx + 0.36, cy + 0.32), (cx + 0.07, cy - 0.02),
                          (cx + 0.07, cy - 0.36), (cx - 0.07, cy - 0.28), (cx - 0.07, cy - 0.02)],
                         closed=True, fc=BLUE[0], ec=INK, lw=1.4))


def book(cx, cy):
    for sx in (-1, 1):
        ax.add_patch(Polygon([(cx, cy - 0.26), (cx + sx * 0.42, cy - 0.18), (cx + sx * 0.42, cy + 0.3), (cx, cy + 0.22)],
                             closed=True, fc="#f6c56a", ec=INK, lw=1.3))


def network(cx, cy):
    pts = [(cx, cy), (cx - 0.35, cy + 0.28), (cx + 0.35, cy + 0.3), (cx - 0.3, cy - 0.3), (cx + 0.32, cy - 0.28)]
    for p in pts[1:]:
        ax.plot([cx, p[0]], [cy, p[1]], color=INK, lw=1.2)
    for p in pts:
        ax.add_patch(Circle(p, 0.1, fc="#f6c56a", ec=INK, lw=1.2))


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#444", lw=2.2, mutation_scale=16))


# Top row: source + three filters
xs, W, Y, H = [0.55, 4.55, 8.6, 12.65], 3.35, 7.45, 1.2
for x, head in zip(xs, ["Source", "Filter", "Filter", "Filter"]):
    ax.text(x + W / 2, 9.35, head, ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
box(xs[0], Y, W, H, GREEN, ["data points", "(3,911)"], database)
box(xs[1], Y, W, H, BLUE, ["Mode &", "status filter"], funnel)
box(xs[2], Y, W, H, BLUE, ["Winner", "detection"], funnel)
box(xs[3], Y, W, H, BLUE, ["Distractor", "selection"], funnel)
for a, b in zip(xs, xs[1:]):
    arrow(a + W, Y + H / 2, b, Y + H / 2)
cap = dict(ha="center", va="center", fontsize=12.5)
ax.text(xs[0] + W / 2, 6.9, "Raw JSON", color=MUTED, **cap)
ax.text(xs[1] + W / 2, 6.85, "pair_compare\nreference_pair\njoint_pair", family="DejaVu Sans Mono", color=INK, linespacing=1.15, **{**cap, "fontsize": 10})
ax.text(xs[2] + W / 2, 6.85, "SELF_REF regex\nbbox slug match", color=INK, linespacing=1.3, **cap)
ax.text(xs[3] + W / 2, 6.85, "SKIP_GT regex\nmax 4 methods", color=INK, linespacing=1.3, **cap)

# Bottom column: question builder -> dataset -> pair constructor
BX, BW = 5.9, 4.7
box(BX, 4.55, BW, 0.95, ORANGE, ["Question builder"], book, fs=15)
box(BX, 2.55, BW, 1.15, GREEN, ["VisionQ-MCQ", "(4,524 questions)"], database, fs=15)
box(BX, 0.6, BW, 0.95, ORANGE, ["Pair constructor"], network, fs=15)
arrow(BX + BW / 2, 4.55, BX + BW / 2, 3.72)
arrow(BX + BW / 2, 2.55, BX + BW / 2, 1.57)
note = dict(ha="right", va="center", fontsize=11.5, color=MUTED, linespacing=1.25)
ax.text(BX - 0.3, 5.02, "1–N questions per data point:\nwith / without reference panel,\nplus 2-choice pairs", **note)
ax.text(BX - 0.3, 3.12, "after human review:\n760 rejected, 200 added", **note)
ax.text(BX - 0.3, 1.07, "one DPO pair per wrong option", **note)

# Side inputs into the question builder
wx = xs[2] + W / 2
ax.plot([wx, wx], [6.35, 6.0], color="#444", lw=2.2)  # starts below the captions, at the divider
ax.plot([wx, BX + 1.6], [6.0, 6.0], color="#444", lw=2.2)
arrow(BX + 1.6, 6.0, BX + 1.6, 5.52)
ax.text(BX + 1.75, 5.78, "winner slug, method slugs", ha="left", va="center", fontsize=11.5, color=MUTED)
ax.text(BX + 1.45, 5.78, "note", ha="right", va="center", fontsize=11.5, color=MUTED, style="italic")
dx = xs[3] + W / 2
ax.plot([dx, dx], [6.35, 5.02], color="#444", lw=2.2)
ax.annotate("", xy=(BX + BW + 0.02, 5.02), xytext=(dx, 5.02), arrowprops=dict(arrowstyle="-|>", color="#444", lw=2.2, mutation_scale=16))
ax.text(dx - 0.15, 5.6, "input", ha="right", va="center", fontsize=11.5, color=MUTED)

from pathlib import Path
_d = Path(__file__).resolve().parents[2] / "results" / "figures"
_d.mkdir(parents=True, exist_ok=True)
out = str(_d / "DPO-dataset")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote", out)
