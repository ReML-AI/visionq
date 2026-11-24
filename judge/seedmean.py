# Paired paper bootstrap between seed-averaged groups. Usage: python3 seedmean.py "A1,A2,A3:B1,B2" ...
import json, sys, random
from collections import defaultdict
def load(m):
    f = f'{m}/base_predictions.jsonl' if m == 'base' else f'{m}/tuned_predictions.jsonl'
    return {r['id']: (r['paper_id'], r['is_correct']) for r in map(json.loads, open(f))}
def avg(ms):
    L = [load(m) for m in ms]; return {i: (L[0][i][0], sum(l[i][1] for l in L) / len(L)) for i in L[0]}
for arg in sys.argv[1:]:
    a, b = (avg(x.split(',')) for x in arg.split(':')); assert a.keys() == b.keys() and len(a) == 538
    P = defaultdict(list)
    for i, (p, v) in a.items(): P[p].append(v - b[i][1])
    ps = list(P); d = lambda s: sum(sum(P[p]) for p in s) / sum(len(P[p]) for p in s)
    rng = random.Random(0); bs = sorted(d(rng.choices(ps, k=len(ps))) for _ in range(10000))
    print(f'{arg}: A {100*sum(v for _,v in a.values())/538:.1f}  B {100*sum(v for _,v in b.values())/538:.1f}  '
          f'diff {100*d(ps):+.1f} [{100*bs[250]:+.1f}, {100*bs[9749]:+.1f}]')
