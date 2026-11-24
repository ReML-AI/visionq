"""PriDe and probability averaging over cyclic orders vs VisionQ-Judge, all from Blackwell letter log-probs."""
import json, math, random
from collections import defaultdict
E2 = '../e2'
def load(f):
    by = defaultdict(dict)
    for l in open(f):
        r = json.loads(l); by[r['base_id']][r['shift']] = r
    return by
am = lambda v: max(range(len(v)), key=v.__getitem__)
def prior(by):   # PriDe: per-question position prior from all cyclic orders, averaged per option count
    acc = defaultdict(list)
    for q in by.values():
        n = q[0]['n']; m = [sum(q[k]['logp'][i] for k in range(n)) / n for i in range(n)]
        z = math.log(sum(math.exp(x) for x in m)); acc[n].append([x - z for x in m])
    return {n: [math.log(sum(math.exp(v[i]) for v in vs) / len(vs)) for i in range(n)] for n, vs in acc.items()}
def permavg(q):  # shift k puts original option (j+k)%n at position j
    n = q[0]['n']; s = [0.0] * n
    for k in range(n):
        for j in range(n): s[(j + k) % n] += q[k]['logp'][j]
    return s
def methods(m):
    val, test = load(f'{m}_val.jsonl'), load(f'{m}_test.jsonl'); P = prior(val)
    out = {}
    for b, q in test.items():
        r = q[0]; n = r['n']; g = ord(r['answer']) - 65
        out[b] = {'paper': r['paper_id'], 'n': n, 'gold': g,
                  'single': am(r['logp']),
                  'pride': am([x - p for x, p in zip(r['logp'], P[n])]),
                  'permavg': (g - 0) if False else am(permavg(q))}
    return out
def boot(pairs, B=10000):
    by = defaultdict(list)
    for p, v in pairs: by[p].append(v)
    ps = list(by); rng = random.Random(0); vs = []
    for _ in range(B):
        xs = [x for p in (rng.choice(ps) for _ in ps) for x in by[p]]; vs.append(sum(xs) / len(xs))
    vs.sort(); return 100 * sum(v for _, v in pairs) / len(pairs), 100 * vs[int(.025 * B)], 100 * vs[int(.975 * B)]
R = {m: methods(m) for m in ['base', 'Cfix_s42', 'Cfix_s43', 'Cfix_s44']}
ids = sorted(R['base'])
acc = lambda m, k: [int(R[m][i][k] == R[m][i]['gold']) for i in ids]
last = lambda m, k: 100 * sum(R[m][i][k] == R[m][i]['n'] - 1 for i in ids) / len(ids)
seeds = ['Cfix_s42', 'Cfix_s43', 'Cfix_s44']
for m in R:
    print(m, ' '.join(f"{k} {100*sum(acc(m,k))/len(ids):.1f} (last {last(m,k):.1f})" for k in ('single', 'pride', 'permavg')))
for k in ('single', 'pride', 'permavg'):
    print(f'Judge mean {k}: {100*sum(sum(acc(s,k)) for s in seeds)/3/len(ids):.1f}, last {sum(last(s,k) for s in seeds)/3:.1f}')
judge = [sum(x) / 3 for x in zip(*[acc(s, 'single') for s in seeds])]
pap = [R['base'][i]['paper'] for i in ids]
for k in ('single', 'pride', 'permavg'):
    print(f'Judge single (seed mean) minus base {k}: %+.1f [%+.1f, %+.1f]' % boot(list(zip(pap, [j - b for j, b in zip(judge, acc('base', k))]))))
# E2 answer changes with both conditions on Blackwell (standard = argmax letter after "(", E2 = greedy generation)
L = lambda f: {(r := json.loads(l))['id']: r for l in open(f)}
pl = lambda r: (r.get('pred_letter') or '').strip('()')
for m, f in [('base', 'base/base_predictions.jsonl')] + [(s, f'{s}/tuned_predictions.jsonl') for s in seeds]:
    e = L(f'{E2}/{f}'); ch = gain = loss = 0
    for i in ids:
        st = chr(65 + R[m][i]['single']); g = chr(65 + R[m][i]['gold']); p = pl(e[i])
        if st != p:
            ch += 1; gain += st == g; loss += p == g
    print(f'E2 same-machine {m}: changes {ch} ({100*ch/len(ids):.1f}%), standard right/E2 wrong {gain}, reverse {loss}, E2-minus-standard {(loss-gain)/len(ids)*100:+.1f}')
# Same-machine E2 paired differences with paper bootstrap (seed mean for the judge)
def e2acc(f): e = L(f'{E2}/{f}'); return [int(pl(e[i]) == chr(65 + R['base'][i]['gold'])) for i in ids]
eb = e2acc('base/base_predictions.jsonl'); ej = [sum(x) / 3 for x in zip(*[e2acc(f'{s}/tuned_predictions.jsonl') for s in seeds])]
print('E2 minus standard, base: %+.1f [%+.1f, %+.1f]' % boot(list(zip(pap, [a - b for a, b in zip(eb, acc('base', 'single'))]))))
print('E2 minus standard, judge: %+.1f [%+.1f, %+.1f]' % boot(list(zip(pap, [a - b for a, b in zip(ej, judge)]))))
jp = [sum(x) / 3 for x in zip(*[acc(s, 'permavg') for s in seeds])]
print('judge permavg minus judge single: %+.1f [%+.1f, %+.1f]' % boot(list(zip(pap, [a - b for a, b in zip(jp, judge)]))))
for k in ('pride', 'permavg'):   # Tab. 5 block (c): each alternative minus the base model's single call
    print(f'base {k} minus base single: %+.1f [%+.1f, %+.1f]' % boot(list(zip(pap, [a - b for a, b in zip(acc('base', k), acc('base', 'single'))]))))
print('judge permavg minus base single: %+.1f [%+.1f, %+.1f]' % boot(list(zip(pap, [a - b for a, b in zip(jp, acc('base', 'single'))]))))
print('calls per question', sum(R['base'][i]['n'] for i in ids) / len(ids))
