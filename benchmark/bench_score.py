#!/usr/bin/env python3
"""Score a bench_run.py output JSONL.

Computes:
  - overall accuracy (and 95% Wilson CI)
  - per-MCQ-size (2-way / 3-way / 4-way) accuracy
  - per-axis accuracy
  - per-leaf accuracy (top 12 by support)
  - error / unparseable rate
  - token / latency totals (for cost estimation)

Usage:
    python bench_score.py outputs/run_haiku_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — better than normal approx for small n / extreme p."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fmt_acc(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson_ci(k, n)
    return f"{k:4d}/{n:<4d} = {100*k/n:5.2f}%  [{100*lo:5.2f}, {100*hi:5.2f}]"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="JSONL output from bench_run.py")
    p.add_argument("--summary-out", type=Path, default=None,
                   help="Optional JSON file for machine-readable summary.")
    p.add_argument("--top-leaves",  type=int, default=12)
    p.add_argument("--slice",
                   choices=["eligible", "eligible-with-ref", "eligible-no-ref",
                            "eligible-clean-repr-crop", "ineligible", "all"],
                   default="eligible", dest="slice_name",
                   help="Filter records by quality_flags before scoring. "
                        "'eligible' (default) = HEADLINE, drops records with fatal "
                        "structural failures. 'eligible-with-ref' = subset where "
                        "P1-ref is meaningful. 'eligible-no-ref' = sanity check "
                        "subset where P1 ≡ P1-ref. 'eligible-clean-repr-crop' = "
                        "strictest (no representative-crop ambiguity). "
                        "'ineligible' / 'all' = analytical slices.")
    args = p.parse_args(argv)

    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    if not rows:
        print("No rows to score.", file=sys.stderr)
        sys.exit(1)

    # Apply slice filter (requires quality_flags from --tag-quality at run time).
    n_before = len(rows)
    if args.slice_name != "all":
        def keep(r):
            q = r.get("quality_flags") or {}
            elig = bool(q.get("benchmark_eligible"))
            has_ref = bool(q.get("has_reference_context"))
            clean_crop = (not q.get("repeated_main_crop")
                          and not q.get("distinct_row_viz_ambiguity"))
            if args.slice_name == "eligible":               return elig
            if args.slice_name == "eligible-with-ref":      return elig and has_ref
            if args.slice_name == "eligible-no-ref":        return elig and not has_ref
            if args.slice_name == "eligible-clean-repr-crop": return elig and clean_crop
            if args.slice_name == "ineligible":             return not elig
            return True
        kept = [r for r in rows if keep(r)]
        if not kept:
            print(f"No rows after --slice {args.slice_name} "
                  f"(records may lack quality_flags — run with --tag-quality).",
                  file=sys.stderr)
            sys.exit(1)
        rows = kept
        print(f"slice={args.slice_name}: kept {len(rows)}/{n_before} records",
              file=sys.stderr)

    n_total       = len(rows)
    n_error       = sum(1 for r in rows if r["status"] != "ok")
    n_unparseable = sum(1 for r in rows if r["status"] == "ok" and r["pred_letter"] is None)
    n_answered    = sum(1 for r in rows if r["status"] == "ok" and r["pred_letter"] is not None)
    n_correct     = sum(1 for r in rows if r["correct"])

    # Every question counts; an unparseable answer or a failed request is scored wrong.
    # Per-axis
    by_axis: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in rows:
        ax = r["axis"] or "_none"
        by_axis[ax][1] += 1
        if r["correct"]:
            by_axis[ax][0] += 1

    # Per-leaf (top by support)
    by_leaf: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        lf = r["leaf"] or "_none"
        by_leaf[lf][1] += 1
        if r["correct"]:
            by_leaf[lf][0] += 1

    # Per MCQ size
    by_size: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        n = r["n_choices"]
        by_size[n][1] += 1
        if r["correct"]:
            by_size[n][0] += 1

    # Tokens / latency
    in_tok  = sum(r.get("input_tokens", 0) for r in rows)
    out_tok = sum(r.get("output_tokens", 0) for r in rows)
    lat     = sum(r.get("latency_ms", 0) for r in rows)

    # Random baseline (per record uniform random over n_choices)
    random_baseline = sum(1 / r["n_choices"] for r in rows)

    # ---- Print ----
    print(f"=== {args.input.name} ===")
    print(f"Total records   : {n_total}")
    print(f"Errors          : {n_error}")
    print(f"Unparseable     : {n_unparseable}")
    print(f"Answered        : {n_answered}")
    print()
    print(f"Overall accuracy: {fmt_acc(n_correct, n_total)}")
    print(f"  (random baseline ~{100 * random_baseline / max(1, n_total):.2f}%)")
    print()
    print("By MCQ size:")
    for n_ch in sorted(by_size):
        c, t = by_size[n_ch]
        print(f"  {n_ch}-way: {fmt_acc(c, t)}")
    print()
    print("By axis:")
    for ax, (c, t) in sorted(by_axis.items(), key=lambda kv: -kv[1][1]):
        print(f"  {ax:20s}: {fmt_acc(c, t)}")
    print()
    print(f"By leaf (top {args.top_leaves} by support):")
    for lf, (c, t) in sorted(by_leaf.items(), key=lambda kv: -kv[1][1])[: args.top_leaves]:
        print(f"  {lf:30s}: {fmt_acc(c, t)}")
    print()
    print(f"Tokens          : in={in_tok:,}  out={out_tok:,}")
    print(f"Total latency   : {lat/1000:.1f}s")
    if n_total:
        print(f"Avg per call    : {lat/n_total:.0f}ms, in={in_tok//n_total}, out={out_tok//n_total}")

    if args.summary_out:
        summary = {
            "input": str(args.input),
            "n_total": n_total, "n_error": n_error, "n_unparseable": n_unparseable,
            "n_answered": n_answered, "n_correct": n_correct,
            "accuracy": n_correct / n_answered if n_answered else None,
            "wilson_ci_95": wilson_ci(n_correct, n_answered),
            "by_size": {str(k): {"correct": v[0], "total": v[1]} for k, v in by_size.items()},
            "by_axis": {k: {"correct": v[0], "total": v[1]} for k, v in by_axis.items()},
            "by_leaf": {k: {"correct": v[0], "total": v[1]} for k, v in by_leaf.items()},
            "tokens_in": in_tok, "tokens_out": out_tok,
            "latency_ms_total": lat,
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
