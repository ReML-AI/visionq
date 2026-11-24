"""Leaf-stratified, paper-disjoint train/val/test split for the VisionQ-Judge DPO runs (frozen before training).

Rule, fixed in advance:
  * Pool: data/mcq/pool.jsonl (4,524 questions), Trap-4 filter, each leaf capped at 300 questions (seed 42), as in every DPO run.
  * Every paper goes to exactly one side (no paper in two sides, across all leaves).
  * Coverage: every leaf with >= 2 papers gets at least one paper in test, every leaf with >= 3 papers
    at least one in val, and every leaf keeps at least one paper in train. Rarest leaves are placed first;
    among a leaf's candidate papers the smallest is taken (ties broken by the seeded shuffle).
  * Fill: test, then val, are topped up to 15% / 10% of the questions, each time adding the paper that
    brings the side's axis mix closest (L1) to the pool's axis mix.
Usage: python judge/make_split.py data/mcq/pool.jsonl split.json [--seed 42]
"""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path

import train_dpo as T


def make_split(rows, seed=42, test_frac=0.15, val_frac=0.10):
    rng = random.Random(seed)
    prow, pleaf, lpap = defaultdict(list), defaultdict(set), defaultdict(set)
    for r in rows:
        prow[r["paper_id"]].append(r); pleaf[r["paper_id"]].add(r["leaf_name"]); lpap[r["leaf_name"]].add(r["paper_id"])
    order = sorted(prow); rng.shuffle(order)
    rank = {p: i for i, p in enumerate(order)}
    side = {}                                                   # paper -> 'test' | 'val'; unassigned = train
    pool_ax = Counter(r["leaf_path"][0] for r in rows)
    N = len(rows)

    def train_ok(p):                                            # each of p's leaves keeps a train paper
        return all(any(q != p and q not in side for q in lpap[l]) for l in pleaf[p])

    for s, min_p in (("test", 2), ("val", 3)):
        for leaf in sorted(lpap, key=lambda l: (len(lpap[l]), l)):
            if len(lpap[leaf]) < min_p or any(side.get(q) == s for q in lpap[leaf]):
                continue
            cands = sorted((p for p in lpap[leaf] if p not in side and train_ok(p)), key=lambda p: (len(prow[p]), rank[p]))
            if cands:
                side[cands[0]] = s

    def l1(ax, n):
        return sum(abs(ax[a] / n - pool_ax[a] / N) for a in pool_ax)

    for s, frac in (("test", test_frac), ("val", val_frac)):
        target = round(N * frac)
        while True:
            ax = Counter(r["leaf_path"][0] for p, v in side.items() if v == s for r in prow[p])
            n = sum(ax.values())
            if n >= target:
                break
            best = min((p for p in order if p not in side and train_ok(p)),
                       key=lambda p: (l1(ax + Counter(r["leaf_path"][0] for r in prow[p]), n + len(prow[p])), rank[p]),
                       default=None)
            if best is None:
                break
            side[best] = s
    out = {"train": sorted(p for p in prow if p not in side),
           "val": sorted(p for p, v in side.items() if v == "val"),
           "test": sorted(p for p, v in side.items() if v == "test")}
    return out


def report(rows, split):
    by = {s: [r for r in rows if r["paper_id"] in set(ps)] for s, ps in split.items()}
    leaves = defaultdict(set)
    for r in rows:
        leaves[r["leaf_name"]].add(r["paper_id"])
    pool_ax = Counter(r["leaf_path"][0] for r in rows)
    print(f"{'axis':20s} {'pool':>6s} " + " ".join(f"{s:>6s}" for s in by))
    for a in pool_ax:
        print(f"{a:20s} {pool_ax[a]/len(rows):6.1%} " + " ".join(f"{Counter(r['leaf_path'][0] for r in v)[a]/len(v):6.1%}" for v in by.values()))
    for s, v in by.items():
        print(f"{s:5s}: {len(v):4d} questions ({len(v)/len(rows):.1%}), {len(split[s]):3d} papers, {len({r['leaf_name'] for r in v})} leaves")
    for s, k in (("test", 2), ("val", 3)):
        need = [l for l, ps in leaves.items() if len(ps) >= k]
        got = {r["leaf_name"] for r in by[s]}
        print(f"  {s}: covers {sum(l in got for l in need)}/{len(need)} leaves with >= {k} papers")
    tr = {r["leaf_name"] for r in by["train"]}
    print(f"  train covers {len(tr)}/{len(leaves)} leaves")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("questions"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows = T.filter_records(T.load_questions(Path(a.questions)), drop_trap4=True, max_per_leaf=300, seed=42)
    split = make_split(rows, a.seed)
    s = [set(v) for v in split.values()]
    assert not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2]), "a paper is on two sides"
    assert set().union(*s) == {r["paper_id"] for r in rows}
    report(rows, split)
    Path(a.out).write_text(json.dumps({"rule": __doc__.split("Usage")[0].strip(), "seed": a.seed, **split}, indent=1))
    print("written", a.out)
