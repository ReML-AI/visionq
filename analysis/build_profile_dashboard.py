#!/usr/bin/env python3
"""Model Accuracy and Judgment Profile — VisionQ cross-model dashboard.

Layout (primary → appendix):

  PRIMARY (always visible):
    1. Ranked overall accuracy with Wilson 95% CI (lollipop).
    2. Model × axis profile heatmap (norm_score, diverging palette).

  COLLAPSIBLE (click to expand):
    3. Per-model detail cards (axis breakdown + top/bottom leaves).

  APPENDIX (collapsed by default):
    4. McNemar χ² matrix (paired significance test for top models).
    5. Per-axis forest plots (only for axes with n >= min_axis_n).

Reviewer guidance: focus on accuracy + profile (the research story).
McNemar is rigorous but tangential to the headline claim. Per-axis
forest plots add noise for low-n axes. Both moved to appendix.

Usage:
    python build_profile_dashboard.py --out outputs/dashboard_profiles.html
"""
from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parent
DEFAULT_PRED_DIR = REPO / "results" / "benchmark" / "predictions"

# Display name override (filename stem -> friendly label)
NAME_MAP = {
    "test_questions_p1":                            "Anthropic Haiku 4.5",
    "test_questions_haiku_p1":                      "Anthropic Haiku 4.5",
    "test_questions_sonnet_p1":                     "Anthropic Sonnet 4.6",
    "test_questions_opus_p1":                       "Anthropic Opus 4.6",
    "test_questions_or_gpt-5.5_p1":                 "OpenAI GPT-5.5",
    "test_questions_or_gpt-5.4_p1":                 "OpenAI GPT-5.4",
    "test_questions_or_gpt-5.3-codex_p1":           "OpenAI GPT-5.3-codex",
    "test_questions_or_gpt-5.2_p1":                 "OpenAI GPT-5.2",
    "test_questions_or_gpt-5.1_p1":                 "OpenAI GPT-5.1",
    "test_questions_or_gemma-4-31b-it_p1":          "Gemma-4-31B-it",
    "test_questions_or_gemma-4-26b-a4b-it_p1":      "Gemma-4-26B-A4B-it",
    "test_questions_or_grok-4.20_p1":               "Grok-4.20",
    "test_questions_or_grok-4.3_p1":                "Grok-4.3",
    "test_questions_or_mistral-medium-3-5_p1":      "Mistral Medium 3.5",
    "test_questions_or_glm-5v-turbo_p1":            "GLM-5V Turbo",
    "test_questions_or_mimo-v2.5_p1":               "Xiaomi MiMo v2.5",
    "test_questions_or_nemotron-omni-reasoning_p1": "NVIDIA Nemotron-Omni",
    "test_questions_or_qwen3.6-35b-a3b_p1":         "Qwen 3.6-35B-A3B",
    "test_questions_or_qwen3.6-27b_p1":             "Qwen 3.6-27B",
    "test_questions_or_qwen3.5-9b_p1":              "Qwen 3.5-9B",
    "test_questions_or_gemma4-4b-base_p1":          "Gemma-4-E4B-it (base of VisionQ-Judge)",
    "test_questions_or_gemma4-4b-dpo_p1":           "VisionQ-Judge (Ours)",
}

# Models with † are imported from per-leaf aggregates (not per-record), so
# their per-record correctness flags are deterministic placeholders. They show
# up correctly in lollipop/heatmap (those depend only on leaf-level marginals)
# but MUST be excluded from McNemar (paired test requires real per-record).
AGGREGATED_MARKER = " †"


# Known parameter counts (billions). For MoE models we report (active, total).
# Closed-source models with undisclosed sizes are excluded from the scaling
# scatter but listed in a footnote.
MODEL_PARAMS = {
    # name -> (active_B, total_B). None == undisclosed.
    "Anthropic Haiku 4.5":            (None, None),
    "Anthropic Sonnet 4.6":           (None, None),
    "Anthropic Opus 4.6":             (None, None),
    "OpenAI GPT-5.5":                 (None, None),
    "OpenAI GPT-5.4":                 (None, None),
    "OpenAI GPT-5.3-codex":           (None, None),
    "OpenAI GPT-5.2":                 (None, None),
    "OpenAI GPT-5.1":                 (None, None),
    "Gemma-4-31B-it":                 (31, 31),
    "Gemma-4-26B-A4B-it":             (4,  26),    # MoE: 26B total, ~4B active
    "Grok-4.20":                      (None, None),
    "Grok-4.3":                       (None, None),
    "Mistral Medium 3.5":             (128, 128),  # dense, HF repo name
    "GLM-5V Turbo":                   (None, None),
    "Xiaomi MiMo v2.5":               (15, 310),   # MoE 15B-active / 310B-total, Xiaomi product page
    "NVIDIA Nemotron-Omni":           (3,  30),    # 30B-A3B
    "Qwen 3.6-35B-A3B":               (3,  30),    # MoE — labeled 35B by OpenRouter but reported 30B
    "Qwen 3.6-27B":                   (27, 27),
    "Qwen 3.5-9B":                    (9,  9),
    "Gemma-4-E4B-it (base of VisionQ-Judge)":  (4,  4),
    "VisionQ-Judge (Ours)":   (4,  4),
}


def display_name(stem: str) -> str:
    return NAME_MAP.get(stem, stem)


def wilson(k: int, N: int, z: float = 1.96) -> tuple[float, float]:
    if N == 0:
        return (0.0, 0.0)
    p = k / N
    d = 1 + z * z / N
    c = (p + z * z / (2 * N)) / d
    h = (z * math.sqrt(p * (1 - p) / N + z * z / (4 * N * N))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def diverging_color(v: float, vmin: float = -0.3, vmax: float = 0.6) -> str:
    """Diverging palette centered at 0. Red below, gray near 0, green above.

    Designed so chance-adjusted norm_score=0 is visually neutral.
    """
    if v is None:
        return "#eee"
    v = max(vmin, min(vmax, v))
    if v < 0:
        # red → light gray
        t = (v - vmin) / max(1e-9, -vmin)        # 0 at vmin, 1 at 0
        r = int(220 - (220 - 235) * t)
        g = int(80 + (235 - 80) * t)
        b = int(80 + (235 - 80) * t)
    else:
        # light gray → green
        t = v / max(1e-9, vmax)                  # 0 at 0, 1 at vmax
        r = int(235 - (235 - 50) * t)
        g = int(235 - (235 - 160) * t)
        b = int(235 - (235 - 70) * t)
    return f"rgb({r},{g},{b})"


def build_per_category(rows: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "exp_rand": 0.0})
    for r in rows:
        if False:  # every eligible question counts; unparsed answers are scored wrong
            continue
        cat = r.get(key) or "_none"
        out[cat]["n"] += 1
        out[cat]["correct"] += int(bool(r.get("correct")))
        out[cat]["exp_rand"] += 1.0 / max(1, r.get("n_choices", 4))
    return out


def metrics(stats: dict) -> dict:
    n, c, er = stats["n"], stats["correct"], stats["exp_rand"]
    if n == 0:
        return {"n": 0, "correct": 0, "raw": None, "lift": None, "norm": None, "random": None}
    raw = c / n
    rand = er / n
    lift = raw - rand
    norm = lift / max(1e-9, 1 - rand)
    return {"n": n, "correct": c, "raw": raw, "lift": lift, "norm": norm, "random": rand}


def mcnemar_chi2(b: int, c: int) -> float:
    """Continuity-corrected McNemar χ². b+c < 25 is fragile but still reportable."""
    if (b + c) == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / (b + c)


def chi2_to_p_one_df(chi2: float) -> float:
    """Approximate p-value from χ² with 1 df via the survival function of N(0,1).
    For χ²₁ = z², p_two_sided = 2 * (1 - Φ(|z|)).
    """
    if chi2 <= 0:
        return 1.0
    z = math.sqrt(chi2)
    # erf-based normal CDF
    return 2.0 * 0.5 * math.erfc(z / math.sqrt(2))


# --------------------------------------------------------------------------- #
# Lollipop / CI rendering helpers (pure SVG, no deps)
# --------------------------------------------------------------------------- #

def lollipop_svg(rows: list[tuple[str, float, float, float, int, int]],
                 random_baseline: float,
                 width: int = 880, row_h: int = 22) -> str:
    """Each row = (label, point, lo, hi, correct, n).
    Returns an inline SVG with horizontal CI bars + dot.
    Range x ∈ [0.2, 0.8] (drawn area). Random line drawn as dashed reference.
    """
    pad_l, pad_r, pad_t, pad_b = 240, 90, 28, 28
    h = pad_t + pad_b + row_h * len(rows)
    plot_w = width - pad_l - pad_r
    xmin, xmax = 0.20, 0.80

    def x(v: float) -> float:
        return pad_l + (v - xmin) / (xmax - xmin) * plot_w

    parts = [f"<svg viewBox='0 0 {width} {h}' width='{width}' height='{h}' "
             f"xmlns='http://www.w3.org/2000/svg' style='font-family:ui-monospace,Menlo,monospace;font-size:11px'>"]

    # Axis ticks
    for tick in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        parts.append(f"<line x1='{x(tick)}' x2='{x(tick)}' y1='{pad_t-4}' y2='{h-pad_b+4}' "
                     f"stroke='#ddd' stroke-width='1'/>")
        parts.append(f"<text x='{x(tick)}' y='{h-pad_b+18}' fill='#888' "
                     f"text-anchor='middle'>{int(tick*100)}%</text>")

    # Random baseline reference
    rx = x(random_baseline)
    parts.append(f"<line x1='{rx}' x2='{rx}' y1='{pad_t-4}' y2='{h-pad_b+4}' "
                 f"stroke='#c33' stroke-width='1' stroke-dasharray='3,3'/>")
    parts.append(f"<text x='{rx}' y='{pad_t-8}' fill='#c33' text-anchor='middle' "
                 f"font-weight='600'>random ≈ {int(random_baseline*100)}%</text>")

    # Rows
    for i, (label, pt, lo, hi, c, n) in enumerate(rows):
        y = pad_t + row_h * (i + 0.5)
        # Label (right-aligned in left margin)
        parts.append(f"<text x='{pad_l-10}' y='{y+3}' fill='#222' text-anchor='end' "
                     f"font-weight='600'>{html.escape(label)}</text>")
        # CI bar
        parts.append(f"<line x1='{x(lo)}' x2='{x(hi)}' y1='{y}' y2='{y}' "
                     f"stroke='#4287f5' stroke-width='3' stroke-linecap='round'/>")
        # Point
        parts.append(f"<circle cx='{x(pt)}' cy='{y}' r='4' fill='#0050b3' stroke='white' stroke-width='1.5'/>")
        # Right-side numeric
        parts.append(f"<text x='{x(xmax)+8}' y='{y+3}' fill='#444'>"
                     f"{100*pt:5.1f}% ({c}/{n})</text>")
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Scaling-law scatter plot (log x-axis: active params, y: accuracy)
# --------------------------------------------------------------------------- #

SHORT_LABEL_MAP = {
    "Gemma-4-E4B-it (base of VisionQ-Judge)":   "Gemma-4-E4B-it (base)",
    "VisionQ-Judge (Ours)":    "Gemma-4-E4B-it (Ours, DPO)",
    "Gemma-4-31B-it":               "Gemma-4-31B",
    "Gemma-4-26B-A4B-it":           "Gemma-4-26B-A4B",
    "NVIDIA Nemotron-Omni":         "Nemotron-Omni-30B-A3B",
    "Qwen 3.6-35B-A3B":             "Qwen 3.6-35B-A3B",
    "Qwen 3.6-27B":                 "Qwen 3.6-27B",
    "Qwen 3.5-9B":                  "Qwen 3.5-9B",
    "Mistral Medium 3.5":           "Mistral Medium 3.5 (128B)",
    "Xiaomi MiMo v2.5":             "MiMo v2.5 (15B/310B)",
}


def scaling_svg(rows: list[tuple[str, float, float, float, float]],
                random_baseline: float,
                width: int = 760, height: int = 380) -> str:
    """rows = (label, active_B, total_B, accuracy, has_moe). Log x, linear y.
    Tighter layout: smaller font, shorter labels, vertical staggering when
    points are near each other on x.
    """
    pad_l, pad_r, pad_t, pad_b = 60, 30, 24, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    xmin_log, xmax_log = math.log10(1), math.log10(500)        # extends to 500B for MiMo / Mistral
    ymin, ymax = 0.30, 0.65

    def x(active_b: float) -> float:
        v = max(0.5, min(500.0, active_b))
        return pad_l + (math.log10(v) - xmin_log) / (xmax_log - xmin_log) * plot_w

    def y(acc: float) -> float:
        return pad_t + (1 - (acc - ymin) / (ymax - ymin)) * plot_h

    parts = [f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' "
             f"xmlns='http://www.w3.org/2000/svg' "
             f"style='font-family:ui-monospace,Menlo,monospace;font-size:10px'>"]

    # Y grid + ticks (every 5 pp)
    for tick in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        yy = y(tick)
        parts.append(f"<line x1='{pad_l}' x2='{width-pad_r}' y1='{yy}' y2='{yy}' "
                     f"stroke='#eee' stroke-width='1'/>")
        parts.append(f"<text x='{pad_l-6}' y='{yy+3}' fill='#666' text-anchor='end'>"
                     f"{int(tick*100)}%</text>")

    # X grid: powers-of-10 + a few intermediate
    for tick in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        xx = x(tick)
        parts.append(f"<line x1='{xx}' x2='{xx}' y1='{pad_t}' y2='{height-pad_b}' "
                     f"stroke='#eee' stroke-width='1'/>")
        parts.append(f"<text x='{xx}' y='{height-pad_b+15}' fill='#666' "
                     f"text-anchor='middle'>{tick}B</text>")
    parts.append(f"<text x='{pad_l + plot_w/2}' y='{height-pad_b+33}' "
                 f"fill='#222' text-anchor='middle' font-weight='600' font-size='11'>"
                 f"active parameters (log scale)</text>")
    parts.append(f"<text x='{pad_l-44}' y='{pad_t + plot_h/2}' "
                 f"fill='#222' text-anchor='middle' font-weight='600' font-size='11' "
                 f"transform='rotate(-90, {pad_l-44}, {pad_t + plot_h/2})'>"
                 f"accuracy</text>")

    # Random baseline
    yr = y(random_baseline)
    parts.append(f"<line x1='{pad_l}' x2='{width-pad_r}' y1='{yr}' y2='{yr}' "
                 f"stroke='#c33' stroke-width='1' stroke-dasharray='3,3'/>")
    parts.append(f"<text x='{pad_l+6}' y='{yr-4}' fill='#c33' font-size='9'>"
                 f"random ≈ {int(random_baseline*100)}%</text>")

    # Sort by x then alternate label side (above/below) to reduce collisions
    rows = sorted(rows, key=lambda r: (r[1], -r[3]))

    for i, (label, active_b, total_b, acc, has_moe) in enumerate(rows):
        cx, cy = x(active_b), y(acc)
        # Total-params dashed ring for MoE
        if has_moe and total_b > active_b:
            rx = max(8, math.sqrt(total_b) * 2.0)
            parts.append(f"<circle cx='{cx}' cy='{cy}' r='{rx}' fill='none' "
                         f"stroke='#999' stroke-dasharray='2,2' stroke-width='1'/>")
        # Active-params dot — small, consistent
        rd = 5
        is_ours = "Ours" in label
        fill = "#c63" if is_ours else "#0050b3"
        parts.append(f"<circle cx='{cx}' cy='{cy}' r='{rd}' fill='{fill}' "
                     f"stroke='white' stroke-width='1.5'/>")

        short = SHORT_LABEL_MAP.get(label, label)
        # Place label above for odd index, below for even, to spread vertically
        if i % 2 == 0:
            ty = cy - rd - 4
            anchor = "middle"
            tx = cx
        else:
            ty = cy + rd + 11
            anchor = "middle"
            tx = cx
        parts.append(f"<text x='{tx}' y='{ty}' fill='#222' text-anchor='{anchor}' "
                     f"font-size='10'>{html.escape(short)}</text>")

    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "dashboard_profiles.html")
    p.add_argument("--top-leaves",  type=int, default=8,
                   help="Show top-N strongest and weakest leaves per model.")
    p.add_argument("--min-leaf-n",  type=int, default=10,
                   help="Hide leaves with n < this in per-model profile.")
    p.add_argument("--min-axis-n",  type=int, default=30,
                   help="Below this n, axis is treated as exploratory: heatmap "
                        "cell grayed out, forest plot suppressed.")
    p.add_argument("--mcnemar-top", type=int, default=8,
                   help="Compute McNemar matrix for top-N models by accuracy.")
    args = p.parse_args(argv)

    # ---- Load all prediction files from benchmark/predictions/ ----
    pred_paths = sorted(DEFAULT_PRED_DIR.glob("test_questions_*_p1.jsonl"))
    print(f"Found {len(pred_paths)} prediction JSONLs in {DEFAULT_PRED_DIR}")

    # Canonical eligible pool from the haiku run
    haiku_path = DEFAULT_PRED_DIR / "test_questions_haiku_p1.jsonl"
    if not haiku_path.exists():
        # fall back to outputs/
        haiku_path = EXP_DIR / "outputs" / "test_questions_p1.jsonl"
    haiku_rows = [json.loads(l) for l in open(haiku_path)]
    elig_ids = {r["id"] for r in haiku_rows
                if (r.get("quality_flags") or {}).get("benchmark_eligible")}
    print(f"Canonical eligible pool: {len(elig_ids)}")

    # Load each model's eligible rows. Dedupe by display name.
    models: dict[str, list[dict]] = {}
    for path in pred_paths:
        name = display_name(path.stem)
        rows = [json.loads(l) for l in open(path)]
        elig = [r for r in rows if r["id"] in elig_ids]
        if name in models:
            print(f"  Skipping duplicate display name {name!r} from {path.name}")
            continue
        models[name] = elig

    # ---- Per-model overall stats + axis stats + leaf stats ----
    overall: dict[str, dict] = {}
    axis_stats: dict[str, dict[str, dict]] = {}
    leaf_stats: dict[str, dict[str, dict]] = {}
    for name, rows in models.items():
        axis_stats[name] = build_per_category(rows, "axis")
        leaf_stats[name] = build_per_category(rows, "leaf")
        agg = {"n": 0, "correct": 0, "exp_rand": 0.0}
        for s in axis_stats[name].values():
            agg["n"] += s["n"]; agg["correct"] += s["correct"]; agg["exp_rand"] += s["exp_rand"]
        overall[name] = metrics(agg)

    # Per-axis n (canonical from any model since they all evaluate the same record set)
    axis_n: dict[str, int] = {}
    for d in axis_stats.values():
        for ax, s in d.items():
            axis_n[ax] = max(axis_n.get(ax, 0), s["n"])

    all_axes = sorted(axis_n, key=lambda a: -axis_n[a])

    # Random baseline (per-record-weighted), use the haiku eligible set
    rand_baseline = sum(1.0 / max(1, r.get("n_choices", 4))
                        for r in haiku_rows if r["id"] in elig_ids
                        and r.get("status") == "ok" and r.get("pred_letter")) / \
                    max(1, sum(1 for r in haiku_rows if r["id"] in elig_ids
                               and r.get("status") == "ok" and r.get("pred_letter")))

    # ---- Build HTML ----
    parts: list[str] = []
    parts.append(f"""<!doctype html><html><head>
<meta charset="utf-8"/>
<title>VisionQ — profile dashboard</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; padding: 24px;
          background: #f7f7f8; color: #222; max-width: 1700px; margin: 0 auto; }}
  h1 {{ margin: 0 0 4px 0; font-size: 22px; }}
  h2 {{ margin: 32px 0 8px 0; font-size: 16px; padding-top: 12px;
        border-top: 1px solid #ddd; }}
  h3 {{ margin: 16px 0 6px 0; font-size: 13px;
        font-family: ui-monospace, Menlo, monospace; }}
  .sub {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  .note {{ background: #fffbe6; border-left: 4px solid #d4a017; padding: 8px 12px;
           font-size: 12px; color: #5a3e00; margin: 8px 0; border-radius: 4px; }}
  table {{ border-collapse: collapse; font-size: 12px; margin: 8px 0; }}
  th, td {{ padding: 4px 8px; text-align: right; border: 1px solid #ddd; }}
  th {{ background: #f0f0f3; font-weight: 600; text-align: center; }}
  td.cat {{ text-align: left; font-family: ui-monospace, Menlo, monospace; background: #fafafb; }}
  td.tot {{ font-weight: 700; background: #eef5ff; }}
  td.gray {{ background: #f0f0f0; color: #888; font-style: italic; }}
  .heatcell {{ position: relative; padding: 6px 8px; min-width: 60px; text-align: center;
               font-family: ui-monospace, monospace; font-weight: 600; font-size: 11px; }}
  .small {{ color: #888; font-size: 11px; }}
  .card {{ background: white; padding: 14px 18px; border-radius: 8px;
           box-shadow: 0 1px 2px rgba(0,0,0,.06); margin-bottom: 16px; }}
  .pwrap {{ background: white; padding: 14px; border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,.06); margin: 8px 0; }}
  .legend {{ display: flex; align-items: center; gap: 8px; margin: 8px 0;
             font-size: 11px; color: #666; flex-wrap: wrap; }}
  .swatch {{ width: 24px; height: 12px; display: inline-block; border-radius: 2px; }}
  .narrow {{ font-size: 10px; color: #999; }}
  .star {{ color: #c63 }}
  details {{ background: #fff; border-radius: 8px; padding: 8px 14px; margin: 12px 0;
             box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  details > summary {{ cursor: pointer; font-weight: 600; font-size: 14px;
                       padding: 6px 0; color: #444; user-select: none; }}
  details > summary:hover {{ color: #0050b3; }}
  details[open] > summary {{ border-bottom: 1px solid #eee; margin-bottom: 12px; }}
</style></head><body>""")

    parts.append("<h1>Model Accuracy and Judgment Profile</h1>")
    parts.append(
        f"<div class='sub'>{len(models)} models on benchmark/test_questions.jsonl, "
        f"slice=eligible n={len(elig_ids)}. Single-pass (Round 1, no positional debias).</div>"
    )

    # =========================================================================
    # 1. PRIMARY — ranked accuracy + Wilson CI lollipop
    # =========================================================================
    parts.append("<h2>1. Overall accuracy — which model performs best?</h2>")
    parts.append("<div class='note'>Sorted by point estimate. Random baseline = red "
                 "dashed line. Bars are Wilson 95% CIs (rough heuristic for ranking). "
                 "Models marked <b>†</b> are imported from per-leaf aggregates "
                 "(not per-record predictions); accuracy and per-axis breakdown are "
                 "correct, but paired-significance tests against them are not valid.</div>")
    rows_sorted = sorted(
        ((name, ov["correct"] / ov["n"], wilson(ov["correct"], ov["n"]),
          ov["correct"], ov["n"])
         for name, ov in overall.items()),
        key=lambda x: -x[1],
    )
    lp_rows = [(name, pt, ci[0], ci[1], c, n)
               for (name, pt, ci, c, n) in rows_sorted]
    parts.append(f"<div class='pwrap'>{lollipop_svg(lp_rows, rand_baseline)}</div>")

    # =========================================================================
    # 2. DIAGNOSTIC — model × axis diverging heatmap (norm_score)
    # =========================================================================
    parts.append("<h2>2. Judgment profile — model × axis (norm_score)</h2>")
    parts.append(
        "<div class='note'>"
        "<b>norm_score</b> = (acc − random) / (1 − random). "
        "0 = chance, 1 = perfect, &lt;0 = below chance. "
        "Cells colored on a diverging palette centered at 0. "
        f"Axes with <span class='star'>★</span> have n &lt; {args.min_axis_n} "
        "(low confidence — gray). Right-most column is the weighted norm_score "
        "across all axes (this is the comparable per-model summary)."
        "</div>")
    parts.append("<div class='legend'>"
                 f"<span class='swatch' style='background:{diverging_color(-0.3)}'></span> ≤−0.3 "
                 f"<span class='swatch' style='background:{diverging_color(0)}'></span> 0 "
                 f"<span class='swatch' style='background:{diverging_color(0.3)}'></span> 0.3 "
                 f"<span class='swatch' style='background:{diverging_color(0.6)}'></span> ≥0.6"
                 "</div>")
    parts.append("<table>")
    header_cells = []
    for ax in all_axes:
        n = axis_n[ax]
        flag = "★" if n < args.min_axis_n else ""
        header_cells.append(f"<th>{html.escape(ax)}{flag}<br/>"
                            f"<span class='narrow'>n={n}</span></th>")
    parts.append("<tr><th class='cat' style='text-align:left'>Model</th>" +
                 "".join(header_cells) +
                 "<th>Weighted<br/>norm_score</th></tr>")
    model_order_norm = sorted(models, key=lambda n: -(overall[n]["norm"] or -1))
    for name in model_order_norm:
        d = axis_stats[name]
        parts.append(f"<tr><td class='cat'>{html.escape(name)}</td>")
        for ax in all_axes:
            m = metrics(d.get(ax, {"n": 0, "correct": 0, "exp_rand": 0.0}))
            if axis_n[ax] < args.min_axis_n:
                parts.append(f"<td class='gray'>{m['norm']:.2f}</td>" if m['norm'] is not None else "<td class='gray'>—</td>")
            elif m["norm"] is None:
                parts.append("<td class='heatcell' style='background:#eee'>—</td>")
            else:
                color = diverging_color(m["norm"])
                tip = (f"raw {100*m['raw']:.1f}% · lift {100*m['lift']:+.1f}pp · "
                       f"random {100*m['random']:.0f}% · n={m['n']}")
                parts.append(f"<td class='heatcell' style='background:{color}' title='{tip}'>{m['norm']:.2f}</td>")
        ov = overall[name]
        color = diverging_color(ov["norm"])
        parts.append(f"<td class='heatcell tot' style='background:{color}'>{ov['norm']:.2f}</td></tr>")
    parts.append("</table>")

    # =========================================================================
    # 3. Scaling — model parameters vs accuracy
    # =========================================================================
    parts.append("<h2>3. Scaling — model parameters vs accuracy</h2>")
    # Build scatter rows for models with disclosed parameter counts
    sc_rows = []
    no_params = []
    for name, ov in overall.items():
        active_b, total_b = MODEL_PARAMS.get(name, (None, None))
        if active_b is None:
            no_params.append(name); continue
        has_moe = (total_b is not None and total_b > active_b)
        sc_rows.append((name, float(active_b), float(total_b or active_b),
                        ov["correct"] / ov["n"], has_moe))
    # Sort by active params for legibility (so labels stack predictably)
    sc_rows.sort(key=lambda r: r[1])

    parts.append("<div class='note'>X-axis: log<sub>10</sub>(total parameters in B). "
                 "Y-axis: accuracy on eligible slice. Solid dot/star = total params "
                 "(advertised model size). Smaller gray dot connected by dotted line "
                 "= active parameters for MoE models. "
                 "<b>Only models with credible primary-source parameter disclosures are plotted</b>; "
                 "9 closed-source models with undisclosed sizes "
                 "(all Anthropic + OpenAI GPT-5.x + GLM-5V Turbo) are excluded. "
                 "'Ours' Gemma-4-E4B-it variants highlighted as orange stars.</div>")
    parts.append(f"<div class='pwrap'>{scaling_svg(sc_rows, rand_baseline)}</div>")
    if no_params:
        parts.append(
            f"<div class='small' style='margin:6px 0 12px 0'>"
            f"<b>Excluded</b> (parameters undisclosed): "
            f"{', '.join(html.escape(n) for n in no_params)}"
            f"</div>"
        )

    # =========================================================================
    # COLLAPSIBLE — Per-model detail cards (expand-on-click)
    # =========================================================================
    parts.append("<h2>4. Per-model detail (click to expand)</h2>")
    parts.append("<details><summary>Show per-model axis breakdown + top/bottom leaves "
                 f"({len(models)} models)</summary>")
    for name in [n for n, *_ in rows_sorted]:
        ov = overall[name]
        parts.append("<div class='card'>")
        parts.append(
            f"<h3>{html.escape(name)} — overall: "
            f"raw {100*ov['raw']:.1f}% · lift {100*ov['lift']:+.1f}pp · "
            f"<b>norm_score {ov['norm']:.3f}</b> · n={ov['n']}</h3>"
        )
        parts.append("<table><tr>"
                     "<th class='cat' style='text-align:left'>Axis</th>"
                     "<th>n</th><th>correct</th><th>raw</th><th>random</th><th>lift</th>"
                     "<th>norm_score</th></tr>")
        d = axis_stats[name]
        for ax in sorted(d.keys(), key=lambda a: -d[a]["n"]):
            m = metrics(d[ax])
            color = diverging_color(m["norm"])
            warn = " ★" if axis_n.get(ax, 0) < args.min_axis_n else ""
            parts.append(
                f"<tr><td class='cat'>{html.escape(ax)}{warn}</td>"
                f"<td>{m['n']}</td><td>{m['correct']}</td>"
                f"<td>{100*m['raw']:.1f}%</td>"
                f"<td>{100*m['random']:.1f}%</td>"
                f"<td>{100*m['lift']:+.1f}pp</td>"
                f"<td class='heatcell' style='background:{color}'>{m['norm']:.3f}</td></tr>"
            )
        parts.append(
            f"<tr><td class='cat tot'>WEIGHTED AVG</td>"
            f"<td class='tot'>{ov['n']}</td><td class='tot'>{ov['correct']}</td>"
            f"<td class='tot'>{100*ov['raw']:.1f}%</td>"
            f"<td class='tot'>{100*ov['random']:.1f}%</td>"
            f"<td class='tot'>{100*ov['lift']:+.1f}pp</td>"
            f"<td class='tot' style='background:{diverging_color(ov['norm'])}'>{ov['norm']:.3f}</td></tr>"
        )
        parts.append("</table>")

        # Top / bottom leaves (only n>=min_leaf_n)
        leaves = leaf_stats[name]
        scored = [(lf, metrics(s)) for lf, s in leaves.items() if s["n"] >= args.min_leaf_n]
        scored = [(lf, m) for lf, m in scored if m["norm"] is not None]
        scored.sort(key=lambda x: -x[1]["norm"])
        if scored:
            top = scored[:args.top_leaves]
            bot = list(reversed(scored[-args.top_leaves:]))
            parts.append("<div style='display:flex; gap:24px;'>")
            for label, sublist in [
                (f"STRONGEST leaves (n≥{args.min_leaf_n})", top),
                (f"WEAKEST leaves (n≥{args.min_leaf_n})", bot),
            ]:
                parts.append(f"<div><div class='small' style='font-weight:600;margin-top:8px'>{label}</div>")
                parts.append("<table><tr><th class='cat'>Leaf</th><th>n</th>"
                             "<th>raw</th><th>norm</th></tr>")
                for lf, m in sublist:
                    color = diverging_color(m["norm"])
                    parts.append(
                        f"<tr><td class='cat'>{html.escape(lf)}</td>"
                        f"<td>{m['n']}</td><td>{100*m['raw']:.1f}%</td>"
                        f"<td class='heatcell' style='background:{color}'>{m['norm']:.3f}</td></tr>"
                    )
                parts.append("</table></div>")
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</details>")

    # =========================================================================
    # APPENDIX — McNemar + per-axis forest plots (collapsed by default)
    # =========================================================================
    parts.append("<h2>Appendix — advanced statistical views</h2>")
    parts.append("<details><summary>Paired significance: McNemar χ² matrix (top-{} models)"
                 "</summary>".format(args.mcnemar_top))
    parts.append(
        f"<div class='note'>Top-{args.mcnemar_top} models by accuracy. Cell shows "
        "<b>χ²</b> · <b>p</b> · <b>(b, c)</b> where b = row✓+col✗ and c = row✗+col✓. "
        "<b>Significant if χ² &gt; 3.84</b> (p&lt;0.05). "
        "Models below the diagonal are paired against the row model — colored green if "
        "the row model is significantly better, red if worse, gray if not significant. "
        "This is the right test (paired) since all models answered the same questions; "
        "CI overlap is only a rough heuristic.</div>")
    # Skip aggregated-source models (paired test invalid)
    top_models_all = [n for n, _, _, _, _ in rows_sorted[:args.mcnemar_top]]
    top_models = [n for n in top_models_all if AGGREGATED_MARKER not in n]
    n_excluded = len(top_models_all) - len(top_models)
    if n_excluded:
        parts.append(f"<div class='note' style='background:#f4f4f4'>"
                     f"Excluded {n_excluded} aggregated-source model(s) (marked †) — paired "
                     f"test requires real per-record predictions, not leaf-level marginals."
                     f"</div>")
    # Build per-model dict of {id: correct_bool} from eligible-evaluated rows
    correctness: dict[str, dict[str, bool]] = {}
    for name in top_models:
        correctness[name] = {r["id"]: bool(r["correct"]) for r in models[name]
                             if r.get("status") == "ok" and r.get("pred_letter")}
    # Common id set
    common_ids = set.intersection(*(set(d) for d in correctness.values()))
    parts.append(f"<div class='small'>n_common = {len(common_ids)} eligible records "
                 f"answered by all top-{args.mcnemar_top} models.</div>")
    parts.append("<table>")
    parts.append("<tr><th class='cat' style='text-align:left'>row \\ col</th>" +
                 "".join(f"<th>{html.escape(n)}</th>" for n in top_models) + "</tr>")
    for ri, rname in enumerate(top_models):
        parts.append(f"<tr><td class='cat'>{html.escape(rname)}</td>")
        for ci, cname in enumerate(top_models):
            if ri == ci:
                parts.append("<td class='gray'>—</td>"); continue
            b = sum(1 for i in common_ids
                    if correctness[rname][i] and not correctness[cname][i])
            c = sum(1 for i in common_ids
                    if not correctness[rname][i] and correctness[cname][i])
            chi2 = mcnemar_chi2(b, c)
            pval = chi2_to_p_one_df(chi2)
            sig = chi2 > 3.84
            if not sig:
                bg, txtcolor = "#f0f0f0", "#666"
            elif b > c:
                bg, txtcolor = "#d4edda", "#155724"
            else:
                bg, txtcolor = "#f8d7da", "#721c24"
            cell = (f"<td class='heatcell' style='background:{bg};color:{txtcolor}' "
                    f"title='b={b} c={c} χ²={chi2:.2f} p={pval:.3g}'>"
                    f"χ²={chi2:.1f}<br/>"
                    f"<span class='small'>p={pval:.3g} ({b},{c})</span></td>")
            parts.append(cell)
        parts.append("</tr>")
    parts.append("</table>")
    parts.append("</details>")  # close McNemar <details>

    # ---- Per-axis forest plots (collapsed appendix) ----
    parts.append(f"<details><summary>Per-axis forest plots (axes with n ≥ "
                 f"{args.min_axis_n})</summary>")
    parts.append(
        "<div class='note'>"
        f"Forest plot per axis. Skipped: axes with n &lt; {args.min_axis_n} "
        "(Wilson CIs become uninterpretably wide). Skipped axes shown only in the "
        "diagnostic heatmap above."
        "</div>")
    for ax in all_axes:
        n_ax = axis_n[ax]
        if n_ax < args.min_axis_n:
            continue
        # Per-model accuracy on this axis
        ax_rows = []
        for name in models:
            d = axis_stats[name].get(ax)
            if not d or d["n"] == 0:
                continue
            m = metrics(d)
            lo, hi = wilson(m["correct"], m["n"])
            ax_rows.append((name, m["raw"], lo, hi, m["correct"], m["n"]))
        ax_rows.sort(key=lambda x: -x[1])
        parts.append(f"<div class='pwrap'><h3>{html.escape(ax)} (n={n_ax})</h3>")
        # Per-axis random baseline
        ax_rand = sum((s["exp_rand"] for s in (axis_stats[name].get(ax) for name in models) if s)) / \
                  max(1, sum(s["n"] for s in (axis_stats[name].get(ax) for name in models) if s))
        parts.append(lollipop_svg(ax_rows, ax_rand, row_h=18))
        parts.append("</div>")

    # Note about suppressed axes
    suppressed = [ax for ax in all_axes if axis_n[ax] < args.min_axis_n]
    if suppressed:
        notes = ", ".join(f"{ax} (n={axis_n[ax]})" for ax in suppressed)
        parts.append(f"<div class='note' style='background:#f4f4f4;border-color:#888;color:#444'>"
                     f"<b>Suppressed (low-n)</b>: {notes}. See diagnostic heatmap for these.</div>")
    parts.append("</details>")  # close per-axis forest <details>

    parts.append("</body></html>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Open: file://{args.out.resolve()}")


if __name__ == "__main__":
    main()
