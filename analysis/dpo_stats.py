"""Reproduce the VisionQ-Judge numbers in Section 5 from saved predictions.

Accuracy (overall and by number of choices), paper-clustered 95% bootstrap
intervals, last-letter bias, accuracy split by the position of the correct
answer, and per-leaf changes with the number of source papers per leaf.

    python analysis/dpo_stats.py                       # checkpoint-900 (reported)
    python analysis/dpo_stats.py --tuned final         # final checkpoint
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "judge"


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.open())}


def n_choices(r: dict) -> int:
    c = r["choices"]
    return len(json.loads(c.replace("'", '"')) if isinstance(c, str) else c)


def correct(r: dict) -> bool:
    # An unparseable answer (no pred_letter) counts as wrong.
    return bool(r.get("pred_letter")) and f"({r['pred_letter']})" == r["answer"]


def last_letter(r: dict) -> str:
    return chr(64 + n_choices(r))


def bootstrap(items: list[tuple[str, float, float]], B: int = 10_000, seed: int = 0):
    """Paired bootstrap over source papers. items = (paper_id, base_value, tuned_value)."""
    by_paper = defaultdict(list)
    for pid, b, t in items:
        by_paper[pid].append((b, t))
    papers = list(by_paper)
    rng = random.Random(seed)
    deltas = []
    for _ in range(B):
        rows = [x for p in (rng.choice(papers) for _ in papers) for x in by_paper[p]]
        deltas.append(sum(t - b for b, t in rows) / len(rows))
    deltas.sort()
    return deltas[int(0.025 * B)], deltas[int(0.975 * B)]


def report(name: str, items: list[tuple[str, float, float]]) -> None:
    n = len(items)
    b = sum(x[1] for x in items) / n
    t = sum(x[2] for x in items) / n
    lo, hi = bootstrap(items)
    print(f"{name:<38} n={n:<4} {b:.3f} -> {t:.3f}  delta {100*(t-b):+5.1f} pp  "
          f"95% CI [{100*lo:+5.1f}, {100*hi:+5.1f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", choices=["ckpt900", "final"], default="ckpt900")
    args = ap.parse_args()
    base = load(RESULTS / "base_predictions.jsonl")
    tuned = load(RESULTS / f"{args.tuned}_predictions.jsonl")
    ids = [i for i in base if i in tuned]

    print(f"== Accuracy (base vs {args.tuned}) ==")
    acc = [(base[i]["paper_id"], float(correct(base[i])), float(correct(tuned[i]))) for i in ids]
    report("overall", acc)
    for k in (2, 3, 4):
        report(f"  {k}-choice", [a for i, a in zip(ids, acc) if n_choices(base[i]) == k])

    print("\n== Positional bias ==")
    last = [(base[i]["paper_id"], float(base[i].get("pred_letter") == last_letter(base[i])),
             float(tuned[i].get("pred_letter") == last_letter(base[i]))) for i in ids]
    gold_last = sum(base[i]["answer"] == f"({last_letter(base[i])})" for i in ids) / len(ids)
    report("answers on the last letter", last)
    print(f"{'  (gold share on the last letter)':<38} {gold_last:.3f}")
    report("accuracy, answer is not the last letter",
           [a for i, a in zip(ids, acc) if base[i]["answer"] != f"({last_letter(base[i])})"])
    report("accuracy, answer is the last letter",
           [a for i, a in zip(ids, acc) if base[i]["answer"] == f"({last_letter(base[i])})"])

    print("\n== Per leaf (n >= 6), with number of source papers ==")
    leaves = defaultdict(list)
    for i, a in zip(ids, acc):
        leaves[base[i]["leaf_name"]].append(a)
    rows = [(sum(t - b for _, b, t in v) / len(v), leaf, v) for leaf, v in leaves.items() if len(v) >= 6]
    for d, leaf, v in sorted(rows, reverse=True):
        print(f"  {leaf:<26} n={len(v):<3} papers={len({p for p, _, _ in v}):<3} delta {100*d:+6.1f} pp")
    single = sum(len({p for p, _, _ in v}) == 1 for v in leaves.values())
    print(f"\n{single} of {len(leaves)} leaves in the test split come from a single paper.")


if __name__ == "__main__":
    main()
