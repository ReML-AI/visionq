#!/usr/bin/env python3
"""Recognition check: does a judge's expert alignment survive mirrored crops?

Each of the 309 eligible VisionQ-Bench questions is asked twice in one run, with the benchmark's
renderer, prompt and parser: standard crops, and the same crops flipped left-right. Mirroring keeps
the relative quality of the outputs but breaks an exact match with the published figure, so a large
drop would point to recognition of published figures rather than visual comparison.

No request is made without --run. Usage:
  python recognition.py --run --model openai/gpt-5.3-codex --out-dir results/recog_gpt-5.3-codex
  python recognition.py --selftest
"""
import argparse, asyncio, json, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import controls as C            # benchmark renderer, records, parser and run_one come through controls.br
br = C.br


def mirror_crops(records, root, outdir):
    """Write a mirrored copy of every candidate crop; return {original path: mirrored path}."""
    from PIL import Image, ImageOps
    m = {}
    for r in records:
        for p in r.image_paths:
            q = outdir / Path(p).relative_to(root)
            if not q.exists():
                q.parent.mkdir(parents=True, exist_ok=True)
                ImageOps.mirror(Image.open(p)).save(q)
            m[p] = q
    return m


def summarize(rows, records):
    by = {(r['id'], r['condition']): r for r in rows}
    out = {}
    for cond in ('standard', 'mirrored'):
        xs = [by[(r.id, cond)] for r in records if (r.id, cond) in by]
        out[cond] = {'n': len(xs), 'accuracy': sum(x['correct'] for x in xs) / len(xs) if xs else None,
                     'errors': sum(str(x.get('status', '')).startswith('error') for x in xs)}
    pairs = [(C.paper_id(by[(r.id, 'standard')]), int(by[(r.id, 'mirrored')]['correct']) - int(by[(r.id, 'standard')]['correct']))
             for r in records if (r.id, 'standard') in by and (r.id, 'mirrored') in by]
    out['mirrored_minus_standard'] = C.paper_bootstrap(pairs)
    out['same_answer'] = sum(by[(r.id, 'standard')].get('pred_letter') == by[(r.id, 'mirrored')].get('pred_letter')
                             for r in records if (r.id, 'standard') in by and (r.id, 'mirrored') in by) / max(1, len(pairs))
    return out


async def run(a):
    recs = br.load_records(a.questions, a.dataset_root, True, 0, include_context=True, tag_quality=True, data_points_path=a.data_points)
    recs = [r for r in recs if r.quality_flags and r.quality_flags['benchmark_eligible']]
    if len(recs) != 309: raise SystemExit(f'preflight: {len(recs)} eligible records, expected 309; no calls made')
    recs = recs[:a.limit] if a.limit else recs
    mir = mirror_crops(recs, a.dataset_root, a.out_dir / 'mirrored_crops')
    from lib.openrouter_vlm import OpenRouterVLMClient
    client = OpenRouterVLMClient(model_id=a.model)
    sem = asyncio.Semaphore(a.concurrency)
    out = a.out_dir / 'raw.jsonl'
    done = set()
    if out.exists():
        for line in out.open():
            if line.strip():
                x = json.loads(line)
                if not str(x.get('status', '')).startswith('error'): done.add((x['id'], x['condition']))

    async def once(rec, cond):
        rr = replace(rec, context_tiles=[], image_paths=[mir[p] for p in rec.image_paths] if cond == 'mirrored' else list(rec.image_paths))
        row = await br.run_one(client, rr, sem, a.max_tokens, True, a.retries)
        row['condition'] = cond
        return row

    for i, rec in enumerate(recs, 1):
        todo = [c for c in ('standard', 'mirrored') if (rec.id, c) not in done]
        res = await asyncio.gather(*[once(rec, c) for c in todo], return_exceptions=True)
        with out.open('a') as f:
            for c, r in zip(todo, res):
                if isinstance(r, Exception):
                    r = {'id': rec.id, 'condition': c, 'pred_letter': None, 'correct': False, 'status': 'error:' + type(r).__name__, 'error_msg': str(r)}
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        if i % 25 == 0: print(f'{i}/{len(recs)}', file=sys.stderr, flush=True)
    latest = {}
    for line in out.open():
        if line.strip():
            x = json.loads(line); latest[(x['id'], x['condition'])] = x
    s = summarize(list(latest.values()), recs)
    (a.out_dir / 'summary.json').write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=1))


def selftest():
    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / 'a').mkdir(); im = Image.new('RGB', (4, 2), 'white'); im.putpixel((0, 0), (255, 0, 0)); im.save(d / 'a/x.png')
        R = type('R', (), {})(); R.image_paths = [d / 'a/x.png']
        m = mirror_crops([R], d, d / 'out')
        assert Image.open(m[d / 'a/x.png']).getpixel((3, 0)) == (255, 0, 0)   # red pixel moved to the right edge
    rows = [{'id': 'p1__f', 'condition': 'standard', 'correct': True, 'pred_letter': 'A'}, {'id': 'p1__f', 'condition': 'mirrored', 'correct': False, 'pred_letter': 'B'}]
    R = type('R', (), {})(); R.id = 'p1__f'
    s = summarize(rows, [R])
    assert s['standard']['accuracy'] == 1 and s['mirrored']['accuracy'] == 0 and s['mirrored_minus_standard']['mean_difference'] == -1 and s['same_answer'] == 0
    print('selftest ok')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='store_true'); p.add_argument('--selftest', action='store_true')
    p.add_argument('--model', default=''); p.add_argument('--out-dir', type=Path)
    p.add_argument('--questions', type=Path, default=C.ROOT / 'data/bench/test_questions.jsonl')
    p.add_argument('--data-points', type=Path, default=C.ROOT / 'data/bench/data_points.jsonl')
    p.add_argument('--dataset-root', type=Path, default=C.ROOT / 'data/images')
    p.add_argument('--concurrency', type=int, default=4); p.add_argument('--retries', type=int, default=2)
    p.add_argument('--max-tokens', type=int); p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if a.selftest: selftest()
    elif a.run:
        if not a.model or not a.out_dir: raise SystemExit('--model and --out-dir are required')
        asyncio.run(run(a))
    else: p.print_help()
