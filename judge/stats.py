# Seed-mean DPO stats vs base for the paper (test or val split): accuracy, last-option share, accuracy by gold position.
# Usage: python3 stats.py DIR   (DIR holds base/ and model dirs; paired paper bootstrap, 10000 resamples, seed 0)
import json, sys, random, os
from collections import defaultdict
D = sys.argv[1]
G = {'B': ['Bfix_s42', 'Bfix_s43', 'Bfix_s44'], 'C_allord': ['Cfix_s42', 'Cfix_s43', 'Cfix_s44'], 'B_noswap': ['B_noswap']}  # one reordered copy / all orders / none
def load(m):
    f = f'{D}/{m}/' + ('base_predictions.jsonl' if m == 'base' else 'tuned_predictions.jsonl')
    return {r['id']: r for r in map(json.loads, open(f))}
base = load('base'); ids = sorted(base)
last = lambda r: r['pred_letter'] == r['choices'][-1].strip('()')
goldlast = {i: base[i]['answer'] == base[i]['choices'][-1] for i in ids}
def feats(ms):
    L = [load(m) for m in ms]; assert all(set(l) == set(ids) for l in L)
    return {i: (sum(l[i]['is_correct'] for l in L) / len(L), sum(last(l[i]) for l in L) / len(L)) for i in ids}
F = {'base': feats(['base'])} | {g: feats(ms) for g, ms in G.items() if all(os.path.exists(f'{D}/{m}') for m in ms)}
def boot(a, b, k, sub):
    P = defaultdict(list)
    for i in ids:
        if sub(i): P[base[i]['paper_id']].append(F[a][i][k] - F[b][i][k])
    ps = list(P); d = lambda s: sum(sum(P[p]) for p in s) / sum(len(P[p]) for p in s)
    rng = random.Random(0); bs = sorted(d(rng.choices(ps, k=len(ps))) for _ in range(10000))
    return f'{100*d(ps):+.1f} [{100*bs[250]:+.1f}, {100*bs[9749]:+.1f}]'
subs = {'all': lambda i: True, 'gold last': lambda i: goldlast[i], 'gold not last': lambda i: not goldlast[i]}
print(f'{D}: n={len(ids)}, gold last {sum(goldlast.values())}, chance {100*sum(1/len(base[i]["choices"]) for i in ids)/len(ids):.1f}')
for g in F:
    n = lambda s: [i for i in ids if subs[s](i)]
    acc = {s: 100 * sum(F[g][i][0] for i in n(s)) / len(n(s)) for s in subs}
    print(f"{g:9} acc {acc['all']:.1f}  last-share {100*sum(F[g][i][1] for i in ids)/len(ids):.1f}  gold-last {acc['gold last']:.1f}  not-last {acc['gold not last']:.1f}")
for g in [g for g in F if g != 'base']:
    print(f"{g} - base: acc {boot(g,'base',0,subs['all'])} | last-share {boot(g,'base',1,subs['all'])} | not-last acc {boot(g,'base',0,subs['gold not last'])} | last acc {boot(g,'base',0,subs['gold last'])}")
if 'C_allord' in F and 'B' in F:
    print(f"C_allord - B: acc {boot('C_allord','B',0,subs['all'])} | last-share {boot('C_allord','B',1,subs['all'])}")
