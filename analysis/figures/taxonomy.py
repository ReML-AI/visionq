# Redraw of figs/taxonomy_2.png with leaf names matching the Appendix A codebook.
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fontsetup  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams.update({"font.family": "Inter", "pdf.fonttype": 42})
AXES = [  # (number, title lines, stroke, fill, leaves) — leaves in the original figure's order
    (1, ["Image", "Appearance"], "#1d3fc4", "#eef3ff",
     ["Blur", "Noise", "Exposure", "Contrast", "Color", "Lighting", "Global Artifact", "Realism", "Style"]),
    (2, ["Object", "Form"], "#3b8a1e", "#f2faea",
     ["Shape", "Boundary", "Texture", "Surface", "Material", "Completeness", "Pose", "Face", "Anatomy", "Detail", "Segmentation"]),
    (3, ["Scene", "Layout"], "#5a1fb0", "#f6f0ff",
     ["Composition", "Depth", "Scale", "View", "Spatial Distribution", "Occlusion Layout", "Camera Geometry", "Physical Plausibility"]),
    (4, ["Relation"], "#d9861c", "#fff7ec",
     ["Contact", "Correspondence", "Attribute Binding", "Spatial Relation", "Interaction", "Part-Whole Relation"]),
    (5, ["Reference", "Fidelity"], "#cf1f1f", "#fff1f1",
     ["Identity Preservation", "Background Preservation", "Reconstruction Fidelity", "Edit Preservation",
      "Geometry Fidelity", "Style Fidelity", "Mask Fidelity", "Landmark Fidelity", "Segmentation Fidelity"]),
    (6, ["Prompt", "Match"], "#1d6b6b", "#eef8f8",
     ["Object Presence", "Attribute Match", "Spatial Instruction", "Semantic Match", "Text Legibility",
      "Count Match", "Action Match", "Expression Match"]),
]
assert sum(len(a[4]) for a in AXES) == 51

W, GAP, PILL_H, STEP = 3.0, 0.22, 0.42, 0.505
n_max = max(len(a[4]) for a in AXES)
col_top, head_h = 7.3, 1.55
col_h = head_h + 0.25 + n_max * STEP + 0.15
total_w = 6 * W + 5 * GAP
fig, ax = plt.subplots(figsize=(total_w / 1.05, (col_top + 1.75) / 1.05))
ax.set_xlim(-0.1, total_w + 0.1); ax.set_ylim(col_top - col_h - 0.1, col_top + 1.75); ax.axis("off")

# Root box and connectors
rx, rw = total_w / 2 - 2.3, 4.6
ax.add_patch(FancyBboxPatch((rx, col_top + 0.75), rw, 0.85, boxstyle="round,pad=0,rounding_size=0.12", fc="white", ec="#222", lw=1.4))
ax.text(total_w / 2, col_top + 1.36, "ROOT", ha="center", va="center", fontsize=17, fontweight="bold")
ax.text(total_w / 2, col_top + 0.98, "Qualitative evaluation criteria", ha="center", va="center", fontsize=12, fontweight="semibold")
bus_y = col_top + 0.45
ax.plot([total_w / 2, total_w / 2], [col_top + 0.75, bus_y], color="#222", lw=1.4)
xs = [i * (W + GAP) + W / 2 for i in range(6)]
ax.plot([xs[0], xs[-1]], [bus_y, bus_y], color="#222", lw=1.4)
for x in xs:
    ax.annotate("", xy=(x, col_top + 0.02), xytext=(x, bus_y), arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.4, mutation_scale=12))

for i, (num, title, ec, fc, leaves) in enumerate(AXES):
    x0 = i * (W + GAP)
    ax.add_patch(FancyBboxPatch((x0, col_top - col_h), W, col_h, boxstyle="round,pad=0,rounding_size=0.18", fc=fc, ec=ec, lw=1.6))
    ax.add_patch(Circle((x0 + 0.42, col_top - 0.5), 0.26, fc=ec, ec="none"))
    ax.text(x0 + 0.42, col_top - 0.5, str(num), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ty = col_top - 0.38 if len(title) == 2 else col_top - 0.5
    for k, line in enumerate(title):
        ax.text(x0 + W / 2 + 0.25, ty - k * 0.36, line, ha="center", va="center", fontsize=14.5, fontweight="bold", color="#111")
    ax.text(x0 + W / 2 + 0.25, col_top - 1.12, f"{len(leaves)} leaves", ha="center", va="center", fontsize=12.5, fontweight="bold", color=ec)
    ax.plot([x0 + 0.15, x0 + W - 0.15], [col_top - head_h + 0.12, col_top - head_h + 0.12], color=ec, lw=1.8)
    for j, leaf in enumerate(leaves):
        y = col_top - head_h - 0.12 - j * STEP - PILL_H
        ax.add_patch(FancyBboxPatch((x0 + 0.12, y), W - 0.24, PILL_H, boxstyle="round,pad=0,rounding_size=0.1", fc="white", ec=ec, lw=1.1))
        ax.text(x0 + 0.24, y + PILL_H / 2, leaf, ha="left", va="center", fontsize=10.2 if len(leaf) > 21 else 11, color="#111")

from pathlib import Path
_d = Path(__file__).resolve().parents[2] / "results" / "figures"
_d.mkdir(parents=True, exist_ok=True)
out = str(_d / "taxonomy")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.03)
print("wrote", out)
