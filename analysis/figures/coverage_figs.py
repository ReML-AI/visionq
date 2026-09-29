# Redraws of figs/fig1_papers_per_task_type and figs/fig4_top5_leaves_per_task_type.
# Same data and colours as the originals; changes: paper-ready titles (no "Step 1/2"),
# task names with spaces (as in fig2/5/6), and full leaf names (no truncation).
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
(_REPO / "results" / "figures").mkdir(parents=True, exist_ok=True)
OUT = str(_REPO / "results" / "figures") + "/"
EDA = str(_REPO / "data" / "eda") + "/"
SIX = {"Image Appearance", "Object Form", "Scene Layout", "Relation", "Reference Fidelity", "Prompt Match"}

# Paper counts per task type, as annotated on the original fig1 (v4.4 master index, N = 1,403).
PAPERS = [("neural_radiance_field", 195), ("generative_2d", 166), ("point_cloud_analysis", 144),
          ("segmentation_detection", 141), ("reconstruction_3d", 138), ("novel_view_synthesis", 81),
          ("human_pose_mesh", 75), ("generative_3d", 63), ("human_avatar", 56), ("scene_understanding", 50),
          ("depth_estimation", 43), ("motion_generation", 39), ("gaussian_splatting", 38), ("hand_centric", 32),
          ("optical_flow_stereo", 29), ("image_restoration", 27), ("tracking", 25), ("face_centric", 22),
          ("other", 20), ("slam_localization", 19)]
assert sum(n for _, n in PAPERS) == 1403
COLOR = {t: plt.cm.tab20(i) for i, (t, _) in enumerate(PAPERS)}
nice = lambda t: t.replace("_", " ")

# fig1
fig, ax = plt.subplots(figsize=(10.5, 5.2))
names = [t for t, _ in PAPERS][::-1]
vals = [n for _, n in PAPERS][::-1]
ax.barh([nice(t) for t in names], vals, color=[COLOR[t] for t in names])
for i, v in enumerate(vals):
    ax.text(v + 2, i, str(v), va="center", fontsize=9)
ax.set_xlabel("Number of papers")
ax.set_title("Papers per task type (N = 1,403 papers with an annotated figure)")
ax.set_xlim(0, 215)
ax.margins(y=0.01)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT + f"fig1_papers_per_task_type.{ext}", dpi=150)
plt.close(fig)

# fig4
counts = defaultdict(lambda: defaultdict(int))
axes_seen = set()
for r in csv.DictReader(open(EDA + "coverage_matrix.csv")):
    axes_seen.add(r["axis"])
    if r["axis"] in SIX:
        counts[r["task_type"]][r["leaf"]] += int(r["n_records"])
print("axis values in coverage_matrix:", sorted(axes_seen))
print("in-scope data points:", sum(sum(v.values()) for v in counts.values()))
fig, axs = plt.subplots(5, 4, figsize=(20, 18.5))
for axp, (t, _) in zip(axs.flat, PAPERS):
    top = sorted(counts[t].items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    labels, v = [k for k, _ in top][::-1], [n for _, n in top][::-1]
    axp.barh(labels, v, color=COLOR[t])
    axp.set_title(nice(t), fontsize=12)
    axp.tick_params(axis="y", labelsize=10)
    axp.tick_params(axis="x", labelsize=9)
    if t in ("neural_radiance_field", "reconstruction_3d", "motion_generation"):
        print(t, top)
fig.suptitle("Top-5 taxonomy leaves per task type (number of data points)", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.975))
for ext in ("pdf", "png"):
    fig.savefig(OUT + f"fig4_top5_leaves_per_task_type.{ext}", dpi=120)
print("wrote fig1/fig4 v2")
