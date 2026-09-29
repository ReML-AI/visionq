#!/usr/bin/env python3
"""Comprehensive validator for dpo_scratch*.jsonl benchmark datasets.

Runs the full pre-deadline checklist (structural, leakage, correctness,
sampling, metadata, P1-ref-specific) and prints a per-category pass/fail
summary. Usage:

    python validate_dataset.py dpo_scratch.jsonl
    python validate_dataset.py dpo_scratch_v2.jsonl

Each finding is one of:
    [PASS]   check passed for all records
    [WARN]   non-fatal — diagnostic flag, record kept
    [FAIL]   fatal — record should be excluded from benchmark

Counts are reported globally. The JSON returned (with --json-out) lists
per-record fail reasons for spot-inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the canonical helpers — same logic as bench_run.py and
# synthesize_dpo_scratch.py to keep the eligibility verdict consistent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_run import (
    is_extended_reference, method_family_stem, _y_band_of,
    SKIP_BASELINE, _compute_record_quality_flags, _load_data_points,
    DATA_POINTS_DEFAULT, DATASET_ROOT_DEFAULT,
)


REQUIRED_FIELDS = [
    "id", "question", "choices", "answer", "methods", "image_paths",
    "v5_data_point_key", "v5_visual_attr", "v5_claim_raw", "v5_claim_norm",
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input",         type=Path)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    p.add_argument("--data-points",  type=Path, default=DATA_POINTS_DEFAULT)
    p.add_argument("--json-out",     type=Path, default=None,
                   help="Write per-record finding details to JSON")
    p.add_argument("--max-show",     type=int, default=3,
                   help="How many example record IDs to print per fail-reason")
    args = p.parse_args(argv)

    # ---- Load ----
    rows = []
    line_errors = []
    for n, line in enumerate(args.input.open(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            line_errors.append((n, str(e)))
    n_total = len(rows)
    print(f"Loaded {n_total} records from {args.input.name}\n")

    # ---- Tally helpers ----
    fatal: dict[str, list[str]] = defaultdict(list)
    warn:  dict[str, list[str]] = defaultdict(list)

    def add(bucket: dict, key: str, rid: str):
        bucket[key].append(rid)

    # ---- Structural checks ----
    seen_ids: set[str] = set()
    for r in rows:
        rid = r.get("id", "<missing-id>")
        if rid in seen_ids:
            add(fatal, "duplicate_id", rid)
        seen_ids.add(rid)
        for f in REQUIRED_FIELDS:
            if f not in r:
                add(fatal, f"missing_field:{f}", rid)
        if "choices" in r and "methods" in r and "image_paths" in r:
            if not (len(r["choices"]) == len(r["methods"]) == len(r["image_paths"])):
                add(fatal, "length_mismatch_choices_methods_images", rid)
        if r.get("answer") and r.get("choices") and r["answer"] not in r["choices"]:
            add(fatal, "invalid_answer_not_in_choices", rid)
        if not r.get("answer_known", True):
            add(fatal, "answer_known_false", rid)

    # ---- File existence ----
    for r in rows:
        rid = r.get("id", "?")
        for ip in r.get("image_paths", []):
            if not (args.dataset_root / ip).exists():
                add(fatal, "missing_image_file", rid); break

    # ---- Sampling validity (uses bench_run's canonical quality_flags) ----
    print("Loading data_points for sampling-validity checks...", file=sys.stderr)
    dps = _load_data_points(args.data_points)

    cross_subrow_count = 0
    for r in rows:
        rid = r["id"]
        dp = dps.get(r.get("v5_data_point_key", ""))
        if dp is None:
            add(warn, "data_point_missing_in_source", rid)
            continue
        try:
            qf = _compute_record_quality_flags(
                r["image_paths"], dp, r["methods"], r.get("v5_claim_norm"),
            )
        except Exception as e:
            add(fatal, f"quality_flags_compute_error:{type(e).__name__}", rid)
            continue
        for reason in qf["ineligible_reasons"]:
            add(fatal, reason, rid)
        if qf.get("cross_subrow_mixing"):
            add(fatal, "cross_subrow_mixing", rid)
            cross_subrow_count += 1
        # Diagnostic flags → warn
        for flag in ["has_reference_context", "repeated_main_crop",
                     "repeated_main_rgb_crop", "distinct_row_viz_ambiguity",
                     "weak_evidence_grounding"]:
            if qf.get(flag):
                add(warn, flag, rid)

        # Same-figure check (figure id parsed from data_point_key)
        dpk = r.get("v5_data_point_key", "")
        # paper_XXXX/fig_N_pP::group → expect all chosen image paths to start
        # with paper_XXXX/figures/fig_N_pP/
        if "/" in dpk and "::" in dpk:
            paper, rest = dpk.split("/", 1)
            fig_id = rest.split("::", 1)[0]
            prefix = f"{paper}/figures/{fig_id}/"
            if any(not p.startswith(prefix) for p in r["image_paths"]):
                add(fatal, "candidates_not_same_figure", rid)

    # ---- Answer correctness (claim_norm vs answer letter) ----
    for r in rows:
        rid = r["id"]
        cn = r.get("v5_claim_norm") or ""
        if cn:
            # Find any "image X" in claim_norm and verify it matches answer letter
            import re as _re
            m = _re.search(r"\bimage\s+([A-D])\b", cn, _re.IGNORECASE)
            if m:
                claim_letter = m.group(1).upper()
                ans_letter = r.get("answer", "").strip("()")
                if claim_letter != ans_letter:
                    add(fatal, "answer_letter_mismatch_with_claim_norm", rid)

    # ---- "Ours" winner resolution sanity ----
    for r in rows:
        rid = r["id"]
        ans_idx = ord(r["answer"].strip("()")) - ord("A")
        if 0 <= ans_idx < len(r["methods"]):
            winner_label = r["methods"][ans_idx]
            if not winner_label:
                add(warn, "winner_method_label_empty", rid)
            elif winner_label.strip().lower() in {"ours", "our method", "proposed"}:
                add(warn, "self_reference_unresolved", rid)

    # ---- Generic-method-label diagnostic only (eligibility checks come from
    # _compute_record_quality_flags above; avoid double-counting here) ----
    for r in rows:
        rid = r["id"]
        for m in r["methods"]:
            if _is_generic_label(m):
                add(warn, "generic_method_label", rid); break

    # ---- Print summary ----
    print(f"=== STRUCTURAL / FATAL CHECKS ({args.input.name}) ===")
    if line_errors:
        print(f"  invalid JSON lines: {len(line_errors)}")
    if not fatal:
        print(f"  ALL PASS — 0 fatal findings across {n_total} records")
    else:
        print(f"  Fatal findings on {len({rid for rids in fatal.values() for rid in rids})} unique records")
        for reason, rids in sorted(fatal.items(), key=lambda kv: -len(kv[1])):
            print(f"  [FAIL] {reason:50s}: {len(rids):4d} records  "
                  f"(e.g., {', '.join(rids[:args.max_show])})")

    print()
    print(f"=== DIAGNOSTIC / WARN FLAGS ===")
    if not warn:
        print("  No diagnostic flags raised")
    else:
        for reason, rids in sorted(warn.items(), key=lambda kv: -len(kv[1])):
            pct = 100*len(rids)/n_total
            print(f"  [WARN] {reason:50s}: {len(rids):4d}/{n_total} ({pct:.1f}%)")

    # ---- Net eligible count ----
    fatal_rids = {rid for rids in fatal.values() for rid in rids}
    eligible = n_total - len(fatal_rids)
    print()
    print(f"=== ELIGIBILITY VERDICT ===")
    print(f"  Total records      : {n_total}")
    print(f"  Fatal-flagged      : {len(fatal_rids)} ({100*len(fatal_rids)/n_total:.1f}%)")
    print(f"  Eligible (passes all fatal checks): {eligible} ({100*eligible/n_total:.1f}%)")

    # ---- Optional JSON dump ----
    if args.json_out:
        out = {
            "input": str(args.input),
            "n_total": n_total,
            "n_eligible": eligible,
            "n_invalid_json_lines": len(line_errors),
            "fatal":   {k: v for k, v in fatal.items()},
            "warn":    {k: v for k, v in warn.items()},
        }
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\nWrote details -> {args.json_out}")


def _is_generic_label(m: str) -> bool:
    import re as _re
    s = (m or "").strip().lower()
    if _re.match(r"^\(?[a-z]\)?\.?$", s): return True
    if _re.match(r"^method\s*\d+$", s):    return True
    if _re.match(r"^column\s*\d+$", s):    return True
    return False


if __name__ == "__main__":
    main()
