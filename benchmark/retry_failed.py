#!/usr/bin/env python3
"""Retry only the failed/unparseable rows in an existing bench_run.py output.

Reads an existing predictions JSONL, identifies rows where status != "ok" or
pred_letter is None, re-runs ONLY those records via bench_run's normal
machinery, and merges results back in place (writes a .bak first).

Useful when an OpenRouter run partially failed due to rate-limits, credit
exhaustion, or transient errors — rather than re-running the whole 326
records.

Usage:
    python retry_failed.py outputs/test_questions_or_grok-4.3_p1.jsonl \
        --provider openrouter --model "x-ai/grok-4.3" \
        --questions benchmark/test_questions.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

# Reuse bench_run machinery directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_run import (
    Record, run_one, load_records,
    DATASET_ROOT_DEFAULT, QUESTIONS_DEFAULT, DATA_POINTS_DEFAULT,
    BedrockVLMClient, OpenRouterVLMClient,
    DEFAULT_HAIKU_MODEL_ID, DEFAULT_SONNET_MODEL_ID, DEFAULT_OPUS_MODEL_ID,
)


def needs_retry(row: dict) -> bool:
    if row.get("status") != "ok":
        return True
    if row.get("pred_letter") is None:
        return True
    return False


async def amain(args: argparse.Namespace) -> None:
    # 1. Load existing predictions
    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    existing_ids = {r["id"] for r in rows}
    failed_ids = {r["id"] for r in rows if needs_retry(r)}
    print(f"Loaded {len(rows)} predictions; {len(failed_ids)} need retry "
          f"({100*len(failed_ids)/max(1,len(rows)):.1f}%)", file=sys.stderr)

    # 2. Backup
    if not args.no_backup:
        backup = args.input.with_suffix(args.input.suffix + ".bak")
        shutil.copyfile(args.input, backup)
        print(f"Backed up -> {backup}", file=sys.stderr)

    # 3. Load source records (only the ones we need to re-run)
    src = load_records(args.questions, args.dataset_root,
                       drop_dup_methods=True, rotate=0,
                       include_context=False,
                       tag_quality=True,
                       data_points_path=args.data_points)
    src_by_id = {r.id: r for r in src}
    # Records to (re-)run = failed-in-existing PLUS missing-from-existing
    missing_from_existing = set(src_by_id) - existing_ids
    if missing_from_existing:
        print(f"Plus {len(missing_from_existing)} records missing entirely from "
              f"the JSONL (partial run). Adding to retry queue.", file=sys.stderr)
    todo_ids = failed_ids | missing_from_existing
    todo = [src_by_id[i] for i in todo_ids if i in src_by_id]
    if not todo:
        print("Nothing to retry. Exiting.", file=sys.stderr)
        return
    print(f"To run: {len(todo)} records ({len(failed_ids)} retries + "
          f"{len(missing_from_existing)} missing)", file=sys.stderr)

    # 4. Provider client
    if args.provider == "bedrock":
        if args.model == "haiku":   model_id = DEFAULT_HAIKU_MODEL_ID
        elif args.model == "sonnet": model_id = DEFAULT_SONNET_MODEL_ID
        elif args.model == "opus":   model_id = DEFAULT_OPUS_MODEL_ID
        else:                        model_id = args.model
        client = BedrockVLMClient(model_id=model_id)
    else:
        client = OpenRouterVLMClient(model_id=args.model)

    sem = asyncio.Semaphore(args.concurrency)

    # 5. Re-run failed records
    new_results: dict[str, dict] = {}
    done = 0; n_ok = 0; n_err = 0

    async def run_and_capture(rec: Record) -> None:
        nonlocal done, n_ok, n_err
        result = await run_one(client, rec, sem,
                               max_tokens=args.max_tokens,
                               composite=args.composite,
                               retries=args.retries)
        new_results[rec.id] = result
        done += 1
        if result["status"] == "ok" and result.get("pred_letter") is not None:
            n_ok += 1
        else:
            n_err += 1
        mark = "✓" if result.get("correct") else (
               "✗" if result["status"] == "ok" else "E")
        print(f"  [{done:4d}/{len(todo)}] {mark} {rec.id[:60]:60s} "
              f"pred={result.get('pred_letter')} gold={rec.answer_letter} "
              f"({result.get('latency_ms',0)}ms, in={result.get('input_tokens',0)}, "
              f"out={result.get('output_tokens',0)}, attempts={result.get('n_attempts',1)})",
              file=sys.stderr)

    tasks = [asyncio.create_task(run_and_capture(r)) for r in todo]
    await asyncio.gather(*tasks)

    # 6. Merge: replace failed rows with new results, keep successful rows,
    # and APPEND any rows that were missing from the original JSONL.
    merged = []
    for r in rows:
        if r["id"] in new_results:
            merged.append(new_results[r["id"]])
        else:
            merged.append(r)
    appended = 0
    for rid, result in new_results.items():
        if rid not in existing_ids:
            merged.append(result)
            appended += 1
    if appended:
        print(f"Appended {appended} new records that were missing from the "
              f"original JSONL.", file=sys.stderr)

    with args.input.open("w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nRetry complete. {n_ok}/{len(todo)} now ok; {n_err} still failed.",
          file=sys.stderr)
    print(f"Merged -> {args.input}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input",         type=Path, help="Existing predictions JSONL to fix in place")
    p.add_argument("--provider",    default="openrouter", choices=["bedrock", "openrouter"])
    p.add_argument("--model",       required=True, help="haiku/sonnet/opus shortcut, or raw slug")
    p.add_argument("--questions",   type=Path,
                   default=QUESTIONS_DEFAULT,
                   help="Source questions JSONL — must match what bench_run was "
                        "originally called with. Default points at benchmark/"
                        "test_questions.jsonl (the canonical eval split).")
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    p.add_argument("--data-points", type=Path, default=DATA_POINTS_DEFAULT)
    p.add_argument("--composite",   action="store_true", default=True,
                   help="Default True (matches the canonical P1 protocol)")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--max-tokens",  type=int, default=None)
    p.add_argument("--retries",     type=int, default=2)
    p.add_argument("--no-backup",   action="store_true")
    args = p.parse_args(argv)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
