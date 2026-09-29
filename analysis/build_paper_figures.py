#!/usr/bin/env python3
"""Render the VisionQ benchmark dashboard charts as publication-quality
matplotlib figures (PDF for LaTeX + PNG for preview), targeting NeurIPS
column widths.

Outputs into benchmark/figures/:

  fig_accuracy_lollipop.{pdf,png}  — primary headline figure
  fig_axis_heatmap.{pdf,png}       — model × axis norm_score
  fig_scaling.{pdf,png}            — params vs accuracy
  fig_axis_forest.{pdf,png}        — per-axis forest panels (n>=30 only)
  fig_mcnemar.{pdf,png}            — paired χ² (appendix; optional)

Usage:
    python build_paper_figures.py
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

EXP_DIR = Path(__file__).resolve().parent
PRED_DIR = EXP_DIR.parent / "results" / "benchmark" / "predictions"
FIG_DIR  = EXP_DIR.parent / "results" / "figures"

# NeurIPS column widths (inches)
COL_W = 3.25       # single column
TWO_COL_W = 6.75   # full-text width

# Reuse helpers + name maps from the dashboard
import sys
sys.path.insert(0, str(EXP_DIR))
from build_profile_dashboard import (
    NAME_MAP, AGGREGATED_MARKER, MODEL_PARAMS, SHORT_LABEL_MAP,
    wilson, build_per_category, metrics, mcnemar_chi2, chi2_to_p_one_df,
    display_name,
)


# Publication style — Times-ish serif, light grid, no top/right spines
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Liberation Serif", "DejaVu Serif", "Times New Roman", "Times"],
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.35,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,    # TrueType in PDF for editability
    "ps.fonttype":      42,
})


def load_models() -> tuple[dict[str, list[dict]], set[str], list[dict]]:
    """Returns (models name → eligible rows, eligible_ids, haiku rows for baseline)."""
    pred_paths = sorted(PRED_DIR.glob("test_questions_*_p1.jsonl"))

    haiku_path = PRED_DIR / "test_questions_haiku_p1.jsonl"
    haiku_rows = [json.loads(l) for l in haiku_path.open()]
    elig_ids = {r["id"] for r in haiku_rows
                if (r.get("quality_flags") or {}).get("benchmark_eligible")}

    models: dict[str, list[dict]] = {}
    for path in pred_paths:
        name = display_name(path.stem)
        if name in models:
            continue
        rows = [json.loads(l) for l in open(path)]
        models[name] = [r for r in rows if r["id"] in elig_ids]
    return models, elig_ids, haiku_rows


def overall_metrics(rows: list[dict]) -> dict:
    agg = {"n": 0, "correct": 0, "exp_rand": 0.0}
    for r in rows:
        if False:  # every eligible question counts; unparsed answers are scored wrong
            continue
        agg["n"] += 1
        agg["correct"] += int(bool(r.get("correct")))
        agg["exp_rand"] += 1.0 / max(1, r.get("n_choices", 4))
    return metrics(agg)


# --------------------------------------------------------------------------- #
# Figure 1 — Headline lollipop
# --------------------------------------------------------------------------- #

def fig_lollipop(models: dict, out_stem: Path) -> None:
    rows = []
    for name, recs in models.items():
        m = overall_metrics(recs)
        if m["n"] == 0: continue
        lo, hi = wilson(m["correct"], m["n"])
        rows.append((name, m["raw"], lo, hi, m["correct"], m["n"], m["random"]))
    rows.sort(key=lambda r: r[1])  # ascending so best is at top

    n = len(rows)
    fig, ax = plt.subplots(figsize=(TWO_COL_W, max(3.0, 0.22 * n + 0.6)))

    rand_baseline = sum(r[6] * r[5] for r in rows) / sum(r[5] for r in rows)
    ax.axvline(rand_baseline, color="#c33", ls="--", lw=0.8, label=f"random ≈ {100*rand_baseline:.0f}%", alpha=0.8)

    ys = np.arange(n)
    for i, (name, pt, lo, hi, c, total, rand) in enumerate(rows):
        is_ours = "Ours" in name
        color = "#c63" if is_ours else "#0050b3"
        ax.hlines(i, 100*lo, 100*hi, color=color, lw=2.5,
                  alpha=0.85 if is_ours else 0.55)
        if is_ours:
            ax.plot(100*pt, i, "*", color="#c63", ms=11,
                    mec="#7a3a00", mew=1.0, zorder=4)
        else:
            ax.plot(100*pt, i, "o", color=color, ms=5.5,
                    mec="white", mew=1.0, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [(r[0], "Ours" in r[0]) for r in rows]  # placeholder, replaced below
    )
    # Bold Ours in y-tick labels
    ax.set_yticklabels([r[0] for r in rows])
    for tick_label in ax.get_yticklabels():
        if "Ours" in tick_label.get_text():
            tick_label.set_fontweight("bold")
            tick_label.set_color("#7a3a00")
    ax.set_xlim(20, 80)
    ax.set_xlabel("Accuracy on eligible split (%)")
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ax.set_xticklabels([f"{v}%" for v in [20, 30, 40, 50, 60, 70, 80]])
    ax.legend(loc="lower right", frameon=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax.set_title("Headline accuracy with Wilson 95% CI", loc="left")
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png")
    plt.close(fig)
    print(f"  wrote {out_stem}.{{pdf,png}}")


# --------------------------------------------------------------------------- #
# Figure 2 — model × axis heatmap
# --------------------------------------------------------------------------- #

def fig_heatmap(models: dict, out_stem: Path, min_axis_n: int = 30) -> None:
    axis_stats = {n: build_per_category(rs, "axis") for n, rs in models.items()}
    overall = {n: overall_metrics(rs) for n, rs in models.items()}

    axis_n: dict[str, int] = {}
    for d in axis_stats.values():
        for ax_, s in d.items():
            axis_n[ax_] = max(axis_n.get(ax_, 0), s["n"])
    all_axes = sorted(axis_n, key=lambda a: -axis_n[a])

    # Sort models by overall norm_score
    model_order = sorted(models, key=lambda n: -(overall[n]["norm"] or -1))

    # Matrix
    mat = np.full((len(model_order), len(all_axes) + 1), np.nan)
    for i, name in enumerate(model_order):
        d = axis_stats[name]
        for j, ax_ in enumerate(all_axes):
            m = metrics(d.get(ax_, {"n": 0, "correct": 0, "exp_rand": 0.0}))
            if m["norm"] is not None:
                mat[i, j] = m["norm"]
        mat[i, -1] = overall[name]["norm"]

    # Diverging colormap centered at 0
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#dc5050", "#eaeaea", "#329646"], N=256,
    )
    norm = Normalize(vmin=-0.3, vmax=0.6)

    fig, ax = plt.subplots(figsize=(TWO_COL_W, 0.30 * len(model_order) + 1.2))
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(all_axes) + 1))
    ax.set_xticklabels(
        [f"{a}\n(n={axis_n[a]}{'*' if axis_n[a] < min_axis_n else ''})" for a in all_axes]
        + ["Weighted\n(overall)"],
        rotation=0, fontsize=7,
    )
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=8)
    for tick_label in ax.get_yticklabels():
        if "Ours" in tick_label.get_text():
            tick_label.set_fontweight("bold")
            tick_label.set_color("#7a3a00")
    ax.set_title("Model × axis norm_score (= (acc − random) / (1 − random))", loc="left")

    # Annotate cells with norm score
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                tcolor = "white" if abs(v) > 0.35 else "#222"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color=tcolor)

    # Vertical separator before "Weighted" column
    ax.axvline(len(all_axes) - 0.5, color="white", lw=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("norm_score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    # Footnote: ★ = low-n axis
    if any(axis_n[a] < min_axis_n for a in all_axes):
        suppr = [a for a in all_axes if axis_n[a] < min_axis_n]
        fig.text(0.01, 0.005,
                 f"* axes with n < {min_axis_n} (low confidence): {', '.join(suppr)}",
                 fontsize=6.5, color="#666")
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png")
    plt.close(fig)
    print(f"  wrote {out_stem}.{{pdf,png}}")


# --------------------------------------------------------------------------- #
# Figure 3 — scaling: params vs accuracy
# --------------------------------------------------------------------------- #

def fig_scaling(models: dict, out_stem: Path) -> None:
    rows = []
    excluded = []
    for name, recs in models.items():
        active_b, total_b = MODEL_PARAMS.get(name, (None, None))
        if active_b is None:
            excluded.append(name); continue
        m = overall_metrics(recs)
        if m["n"] == 0: continue
        # Plot at TOTAL params (capacity / advertised size), with a smaller
        # inner dot at active for MoE. This matches how models are commonly
        # named ("Nemotron-30B", "Mistral-128B") and is the standard
        # scaling-law convention.
        plot_b = float(total_b or active_b)
        rows.append((name, float(active_b), plot_b,
                     m["raw"], total_b is not None and total_b > active_b))

    fig, ax = plt.subplots(figsize=(TWO_COL_W, 4.0))
    rand_baseline = sum(overall_metrics(rs)["random"] * overall_metrics(rs)["n"]
                        for rs in models.values()) / \
                    max(1, sum(overall_metrics(rs)["n"] for rs in models.values()))
    ax.axhline(100*rand_baseline, color="#c33", ls="--", lw=0.8, alpha=0.8,
               label=f"random ≈ {100*rand_baseline:.0f}%")

    # Per-point label offset overrides for clusters (in display offset points).
    LABEL_OFFSETS = {
        "Gemma-4-E4B-it (Ours, DPO)":   (12,  18),
        "Gemma-4-E4B-it (base)":        (12, -22),
        "Gemma-4-26B-A4B":              (38,   2),
        "Nemotron-Omni-30B-A3B":        (-95, -16),
        "Qwen 3.6-35B-A3B":             (12, -14),
        "Qwen 3.5-9B":                  (12,   8),
        "Qwen 3.6-27B":                 (12,   8),
        "Gemma-4-31B":                  (12,   3),
        "MiMo v2.5 (15B/310B)":         (12,   8),
        "Mistral Medium 3.5 (128B)":    (-110,-12),
    }

    # First pass: plot non-Ours points and capture Ours points to draw later
    ours_points = []  # (name, plot_x, acc)
    for (name, active_b, plot_b, acc, has_moe) in rows:
        is_ours = "Ours" in name
        if is_ours:
            ours_points.append((name, plot_b, acc))
        color = "#c63" if is_ours else "#0050b3"
        # MoE: small inner dot at active params + thin connector to total
        if has_moe and plot_b > active_b:
            ax.scatter([active_b], [100*acc], s=18, color="#888",
                       marker="o", alpha=0.65, zorder=1)
            ax.plot([active_b, plot_b], [100*acc, 100*acc], color="#888",
                    lw=0.5, ls=":", alpha=0.5, zorder=1)
        # Different marker / size for Ours: orange star, larger
        if is_ours:
            ax.scatter([plot_b], [100*acc], s=180, color="#c63",
                       marker="*", edgecolors="#7a3a00", linewidths=1.4,
                       zorder=5)
        else:
            ax.scatter([plot_b], [100*acc], s=42, color=color,
                       edgecolors="white", linewidths=1.2, zorder=3)
        # Label
        short = SHORT_LABEL_MAP.get(name, name)
        offset = LABEL_OFFSETS.get(short, (8, 4))
        weight = "bold" if is_ours else "normal"
        text_color = "#7a3a00" if is_ours else "#222"
        font_size = 8 if is_ours else 7
        ax.annotate(short, (plot_b, 100*acc), xytext=offset,
                    textcoords="offset points",
                    fontsize=font_size, color=text_color, fontweight=weight,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#aaa",
                                    shrinkA=2, shrinkB=2)
                    if max(abs(offset[0]), abs(offset[1])) > 14 else None)

    # Connector arrow from base → DPO showing the lift, with delta callout
    if len(ours_points) == 2:
        ours_points.sort(key=lambda r: r[2])
        (base_name, base_x, base_y), (dpo_name, dpo_x, dpo_y) = ours_points
        delta_pp = (dpo_y - base_y) * 100
        ax.annotate("", xy=(dpo_x, 100 * dpo_y),
                    xytext=(base_x, 100 * base_y),
                    arrowprops=dict(arrowstyle="->", color="#c63", lw=1.6,
                                    alpha=0.85, mutation_scale=14),
                    zorder=4)
        callout_x = dpo_x * 1.5
        callout_y = (100 * (base_y + dpo_y) / 2) + 0.5
        ax.text(callout_x, callout_y,
                f"+{delta_pp:.1f} pp\n(DPO)",
                fontsize=8, color="#7a3a00", fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e6",
                          edgecolor="#c63", lw=0.8, alpha=0.9))

    ax.set_xscale("log")
    ax.set_xlim(1, 500)
    ax.set_ylim(40, 65)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(["1B", "2B", "5B", "10B", "20B", "50B", "100B", "200B", "500B"])
    ax.set_xlabel("Total parameters (log scale) — primary; small dot = MoE active params")
    ax.set_ylabel("Accuracy on eligible split (%)")
    ax.set_title(f"Scaling — params vs accuracy (n={len(rows)} models with disclosed sizes)",
                 loc="left")
    ax.legend(loc="lower right", frameon=False)
    if excluded:
        fig.text(0.01, 0.005,
                 f"Excluded ({len(excluded)} models with undisclosed sizes): "
                 f"{', '.join(excluded[:6])}{'...' if len(excluded) > 6 else ''}",
                 fontsize=6.5, color="#666")
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png")
    plt.close(fig)
    print(f"  wrote {out_stem}.{{pdf,png}}")


# --------------------------------------------------------------------------- #
# Figure 4 — per-axis forest panels (n>=min)
# --------------------------------------------------------------------------- #

def fig_forest(models: dict, out_stem: Path, min_axis_n: int = 30) -> None:
    axis_stats = {n: build_per_category(rs, "axis") for n, rs in models.items()}
    overall = {n: overall_metrics(rs) for n, rs in models.items()}
    axis_n: dict[str, int] = {}
    for d in axis_stats.values():
        for ax_, s in d.items():
            axis_n[ax_] = max(axis_n.get(ax_, 0), s["n"])
    plot_axes = [ax_ for ax_ in axis_n if axis_n[ax_] >= min_axis_n]
    plot_axes.sort(key=lambda a: -axis_n[a])

    # Single shared model order across all panels: by overall accuracy desc
    # (so the strongest models appear at the top of every panel — easier to scan)
    model_order = sorted(models, key=lambda n: -(overall[n]["correct"] / overall[n]["n"]))

    n_panels = len(plot_axes)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(TWO_COL_W + 1.5, 0.24 * len(model_order) + 1.0),
        sharey=False, gridspec_kw={"wspace": 0.10},
    )
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.21, right=0.99)

    for panel_idx, ax_name in enumerate(plot_axes):
        ax_plot = axes[panel_idx]
        # Per-panel rand baseline
        rand_total = sum(axis_stats[name].get(ax_name, {"exp_rand": 0})["exp_rand"]
                         for name in model_order)
        rand_n = sum(axis_stats[name].get(ax_name, {"n": 0})["n"]
                     for name in model_order)
        rand = rand_total / max(1, rand_n)
        ax_plot.axvline(100*rand, color="#c33", ls="--", lw=0.7, alpha=0.7)

        for i, name in enumerate(model_order):
            d = axis_stats[name].get(ax_name)
            if not d or d["n"] == 0:
                continue
            m = metrics(d)
            lo, hi = wilson(m["correct"], m["n"])
            is_ours = "Ours" in name
            color = "#c63" if is_ours else "#0050b3"
            y = len(model_order) - 1 - i  # invert so best is at top
            ax_plot.hlines(y, 100*lo, 100*hi, color=color, lw=2,
                           alpha=0.75 if is_ours else 0.45)
            if is_ours:
                ax_plot.plot(100*m["raw"], y, "*", color="#c63", ms=8,
                             mec="#7a3a00", mew=0.8, zorder=4)
            else:
                ax_plot.plot(100*m["raw"], y, "o", color=color, ms=4,
                             mec="white", mew=0.8)

        ax_plot.set_xlim(15, 85)
        ax_plot.set_ylim(-0.5, len(model_order) - 0.5)
        ax_plot.set_xlabel("Acc (%)", fontsize=8)
        ax_plot.set_title(f"{ax_name}\n(n={axis_n[ax_name]})",
                          fontsize=8, loc="center")
        ax_plot.set_xticks([20, 35, 50, 65, 80])
        ax_plot.set_xticklabels(["20", "35", "50", "65", "80"], fontsize=7)
        ax_plot.tick_params(axis="x", labelsize=7)
        ax_plot.spines["left"].set_visible(False)
        ax_plot.tick_params(axis="y", left=False)

    # Y-axis labels only on the leftmost panel; identical limits across all
    yticks = list(range(len(model_order)))
    ytick_labels = list(reversed(model_order))
    for k, ax_p in enumerate(axes):
        ax_p.set_ylim(-0.5, len(model_order) - 0.5)
        ax_p.set_yticks(yticks)
        if k == 0:
            ax_p.set_yticklabels(ytick_labels, fontsize=7)
            for tick_label in ax_p.get_yticklabels():
                if "Ours" in tick_label.get_text():
                    tick_label.set_fontweight("bold")
                    tick_label.set_color("#7a3a00")
        else:
            ax_p.set_yticklabels([""] * len(yticks))

    fig.suptitle(f"Per-axis accuracy with Wilson 95% CI (axes with n ≥ {min_axis_n}; "
                 f"models sorted by overall accuracy)",
                 fontsize=9.5, x=0.5, ha="center", y=0.995)
    fig.savefig(f"{out_stem}.pdf", bbox_inches=None)
    fig.savefig(f"{out_stem}.png", bbox_inches=None)
    plt.close(fig)
    print(f"  wrote {out_stem}.{{pdf,png}}")


# --------------------------------------------------------------------------- #
# Figure 5 — McNemar matrix
# --------------------------------------------------------------------------- #

def fig_mcnemar(models: dict, out_stem: Path, top_k: int = 8) -> None:
    overall = {n: overall_metrics(rs) for n, rs in models.items()}
    top_models_all = sorted(models, key=lambda n: -(overall[n]["correct"]/overall[n]["n"]))
    top_models = [n for n in top_models_all if AGGREGATED_MARKER not in n][:top_k]

    correctness = {name: {r["id"]: bool(r["correct"]) for r in models[name]
                          if r.get("status") == "ok" and r.get("pred_letter")}
                   for name in top_models}
    common_ids = set.intersection(*(set(d) for d in correctness.values()))
    n_common = len(common_ids)

    K = len(top_models)
    chi_mat = np.full((K, K), np.nan)
    sign_mat = np.zeros((K, K), dtype=int)
    for i, ri in enumerate(top_models):
        for j, ci in enumerate(top_models):
            if i == j: continue
            b = sum(1 for x in common_ids if correctness[ri][x] and not correctness[ci][x])
            c = sum(1 for x in common_ids if not correctness[ri][x] and correctness[ci][x])
            chi_mat[i, j] = mcnemar_chi2(b, c)
            sign_mat[i, j] = 1 if b > c else (-1 if c > b else 0)

    sig_mask = chi_mat > 3.84
    cmap = LinearSegmentedColormap.from_list(
        "ng", ["#dc5050", "#eaeaea", "#329646"], N=256
    )
    norm = Normalize(vmin=-1, vmax=1)
    visual = np.where(sig_mask, sign_mat, 0).astype(float)

    fig, ax = plt.subplots(figsize=(TWO_COL_W, TWO_COL_W * 0.85))
    im = ax.imshow(visual, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels(top_models, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(top_models, fontsize=7)
    for i in range(K):
        for j in range(K):
            if i == j or np.isnan(chi_mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#888")
            else:
                p = chi2_to_p_one_df(chi_mat[i, j])
                txt = f"χ²={chi_mat[i,j]:.1f}"
                if sig_mask[i, j]:
                    txt += f"\np={p:.3g}"
                tcolor = "white" if abs(visual[i, j]) > 0.5 else "#222"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, color=tcolor)
    ax.set_title(f"McNemar χ² matrix — top-{K} models (paired test, n_common={n_common})",
                 loc="left")
    ax.grid(False)
    fig.text(0.01, 0.01,
             "Green: row model significantly better; red: row significantly worse; "
             "gray: not significant (χ² ≤ 3.84). †-marked aggregated models excluded.",
             fontsize=6.5, color="#444")
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png")
    plt.close(fig)
    print(f"  wrote {out_stem}.{{pdf,png}}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--no-mcnemar", action="store_true",
                    help="Skip the McNemar appendix figure")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions ...")
    models, _, _ = load_models()
    print(f"  {len(models)} models loaded")

    print("Rendering figures ...")
    fig_lollipop(models, args.out_dir / "fig_accuracy_lollipop")
    fig_heatmap(models, args.out_dir / "fig_axis_heatmap")
    fig_scaling(models, args.out_dir / "fig_scaling")
    fig_forest(models, args.out_dir / "fig_axis_forest")
    if not args.no_mcnemar:
        fig_mcnemar(models, args.out_dir / "fig_mcnemar")
    print(f"\nAll figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
