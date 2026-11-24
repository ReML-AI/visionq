# Paired paper-clustered bootstrap: accuracy of model A minus model B on the 538 test questions.
import json, random, sys
from collections import defaultdict
def load(m):
    f = f'{m}/base_predictions.jsonl' if m == 'base' else f'{m}/tuned_predictions.jsonl'
    return {r['id']: (r['paper_id'], r['is_correct']) for r in map(json.loads, open(f))}
def ci(a, b, n=4000):
    A, B = load(a), load(b); assert A.keys() == B.keys()
    P = defaultdict(list)
    for k in A: P[A[k][0]].append(A[k][1] - B[k][1])
    ps = list(P); rnd = random.Random(0); d = []
    for _ in range(n):
        s = [x for p in rnd.choices(ps, k=len(ps)) for x in P[p]]; d.append(sum(s) / len(s))
    d.sort(); m = sum(x for v in P.values() for x in v) / len(A)
    return f'{a} - {b}: {100*m:+.1f} [{100*d[int(.025*n)]:+.1f}, {100*d[int(.975*n)]:+.1f}]'
for p in sys.argv[1:]: print(ci(*p.split(':')))
