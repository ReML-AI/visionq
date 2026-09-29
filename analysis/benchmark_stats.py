"""Reproduce the VisionQ-Bench numbers from saved predictions.

Overall and per-axis accuracy on the 309 eligible questions for all 21 rows
(19 external judges, VisionQ-Judge, and its base model), with 95% intervals
from a bootstrap over source papers (Table 5), plus the 2023-vs-2024 check.
Unparseable answers and API failures count as wrong.

    python analysis/benchmark_stats.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRED = REPO / "results" / "benchmark" / "predictions"
AXES = ["Object Form", "Reference Fidelity", "Image Appearance", "Relation", "Prompt Match", "Scene Layout"]


def paper(r: dict) -> str:
    return r["id"].split("__")[0]


def interval(rows: list[dict], B: int = 4000, seed: int = 0) -> tuple[float, float]:
    by_paper = defaultdict(list)
    for r in rows:
        by_paper[paper(r)].append(bool(r.get("correct")))
    papers, rng, accs = list(by_paper), random.Random(seed), []
    for _ in range(B):
        xs = [x for p in (rng.choice(papers) for _ in papers) for x in by_paper[p]]
        accs.append(sum(xs) / len(xs))
    accs.sort()
    return accs[int(0.025 * B)], accs[int(0.975 * B)]


def main() -> None:
    models = {}
    for f in sorted(PRED.glob("test_questions_*_p1.jsonl")):
        rows = [json.loads(line) for line in f.open()]
        models[f.stem[len("test_questions_"):-len("_p1")]] = [
            r for r in rows if (r.get("quality_flags") or {}).get("benchmark_eligible")]
    any_rows = next(iter(models.values()))
    print(f"{len(models)} models, {len(any_rows)} eligible questions, "
          f"chance {sum(1 / r['n_choices'] for r in any_rows) / len(any_rows):.3f}")
    print("questions / papers per axis:",
          {a: (sum(r['axis'] == a for r in any_rows), len({paper(r) for r in any_rows if r['axis'] == a})) for a in AXES})

    table = []
    for name, rows in models.items():
        cells = {"Overall": rows, **{a: [r for r in rows if r["axis"] == a] for a in AXES}}
        table.append((sum(bool(r.get("correct")) for r in rows) / len(rows), name,
                      {k: (sum(bool(r.get("correct")) for r in v) / len(v),) + interval(v) for k, v in cells.items()}))
    print(f"\n{'model':<32}" + "".join(f"{k[:12]:>18}" for k in ["Overall"] + AXES))
    for _, name, cells in sorted(table, reverse=True):
        print(f"{name:<32}" + "".join(f"{100*a:5.1f} [{100*lo:3.0f},{100*hi:3.0f}]".rjust(18) for a, lo, hi in cells.values()))

    year = {r["paper_id"]: r["year"] for r in csv.DictReader((REPO / "data" / "paper_list.csv").open())}
    print("\nquestions by publication year:", dict(Counter(year.get(paper(r)) for r in any_rows)))
    diffs = []
    for name, rows in models.items():
        acc = {y: [bool(r.get("correct")) for r in rows if year.get(paper(r)) == y] for y in ("2023", "2024")}
        diffs.append(sum(acc["2023"]) / len(acc["2023"]) - sum(acc["2024"]) / len(acc["2024"]))
    print(f"accuracy(2023) - accuracy(2024): mean {100*sum(diffs)/len(diffs):+.1f} pp; "
          f"higher on 2023 for {sum(d > 0 for d in diffs)} of {len(diffs)} models")


if __name__ == "__main__":
    main()
