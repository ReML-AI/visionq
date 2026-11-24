# No-reference IQA baselines (Sec. 5): for each scored question, pick the crop with the highest predicted quality.
# Also a sharpness heuristic (variance of a 3x3 Laplacian). Reports accuracy with a 95% paper-bootstrap interval.
#   python baselines/iqa_baseline.py musiq clipiqa+ topiq_nr          # score crops (GPU, needs pyiqa)
#   python baselines/iqa_baseline.py --from-scores results/iqa/iqa_scores.json   # re-score saved values only
import json, sys, random
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data'
scored = set(open(D / 'bench/scored_ids.txt').read().split())
items = [{'id': q['id'], 'paper': q['paper_id'], 'files': q['image_paths'], 'gold': q['choices'].index(q['answer'])}
         for q in map(json.loads, open(D / 'bench/test_questions.jsonl')) if q['id'] in scored]


def lap_var(path):
    g = np.asarray(Image.open(path).convert('L'), dtype=np.float32)
    l = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(l.var())


def boot(rows, B=4000):
    by = defaultdict(list)
    for p, v in rows: by[p].append(v)
    ps, rng, acc = list(by), random.Random(0), []
    for _ in range(B):
        xs = [x for p in (rng.choice(ps) for _ in ps) for x in by[p]]; acc.append(sum(xs) / len(xs))
    acc.sort(); return acc[int(.025 * B)], acc[int(.975 * B)]


if sys.argv[1:2] == ['--from-scores']:
    scores = json.load(open(sys.argv[2]))
else:
    import torch, pyiqa
    scores = defaultdict(dict)
    for it in items:
        scores['sharpness'][it['id']] = [lap_var(D / 'images' / f) for f in it['files']]
    for name in sys.argv[1:]:
        m = pyiqa.create_metric(name, device='cuda'); sign = -1 if m.lower_better else 1
        for it in items:
            with torch.no_grad(): scores[name][it['id']] = [sign * float(m(str(D / 'images' / f))) for f in it['files']]
        del m; torch.cuda.empty_cache()
    json.dump(scores, open('iqa_scores.json', 'w'))
for name, S in scores.items():
    rows = [(it['paper'], float(int(np.argmax(S[it['id']])) == it['gold'])) for it in items]
    a = sum(v for _, v in rows) / len(rows); lo, hi = boot(rows)
    print(f'{name:12s} acc {100*a:.1f} [{100*lo:.0f},{100*hi:.0f}]  n={len(rows)}')
