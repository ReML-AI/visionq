#!/usr/bin/env python3
"""Deterministic question sanitiser shared by the human study and E4 (controls.py).

sanitize_question(record) takes the full repaired criterion question and removes
method names: (1) comparison clauses ("compared to/with ...", "than ...",
"vs ...", "over ...") that contain a method name, (2) every remaining candidate
label of the item and the source paper's method name (whole word, case-insensitive,
optionally followed by a bracketed citation such as "[27]"). Punctuation is
collapsed; if fewer than 3 words remain the leaf name is used. Nothing is added.
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPERS = REPO/'data/papers.csv'
HERE = Path(__file__).resolve().parent
TEMPLATE = 'Which method demonstrates better '
GENERIC = {w.lower() for w in (
    'Image Depth Error View Shape Style Camera Plane Edit Identity Visual Distribution Disparity NeRF').split()}
# Ordinary English words that occur inside multi-word labels (e.g. "Average Feature
# Matching"); they are not method names, so neither removed nor counted as leaks.
# Plain words / context labels that appear as candidate labels but are not method names.
GENERIC |= {w.lower() for w in (
    'Baseline Condition Distribution Events Forged Front Merged Original Partial Query Seen Source Support Top '
    'Vanilla Voxel Conv Recon Gaussians Panorama Ground-Truth GT Input Reference Target Low-Resolution').split()}
_titles = None

def _paper_title(pid):
    global _titles
    if _titles is None:
        with PAPERS.open(newline='') as f:
            _titles = {r['paper_id']: r['title'] for r in csv.DictReader(f)}
    return _titles.get(pid, '')

def _strip_ours(label):
    return re.sub(r'\(\s*ours[^)]*\)|\bours\b', ' ', label, flags=re.I).strip(' -+,:')

def paper_method_name(title):
    """Leading method name of titles shaped 'Name: subtitle' (<=3 words, no spaces needed)."""
    if ':' in title:
        head = title.split(':', 1)[0].strip()
        if 0 < len(head.split()) <= 2 and len(head) >= 3: return head
    return ''

def label_tokens(label):
    """Tokens (len>=4, non-generic) a blinding check should look for."""
    toks = re.findall(r'[A-Za-z0-9][A-Za-z0-9.\-]*', _strip_ours(label))
    toks = [t for t in toks if len(t) >= 4 and t.lower().strip('.-') not in GENERIC]
    if len(toks) > 1 or (toks and len(re.findall(r'\w+', _strip_ours(label))) > 1):
        # words inside multi-word labels count only when name-like (digit, inner capital, hyphen, all caps)
        toks = [t for t in toks if re.search(r'\d|-|[A-Z].*[A-Z]|.[A-Z]', t)]
    return toks

def names_for(record):
    """Phrases to remove: full labels and non-generic label tokens, plus the paper's method name."""
    names = set()
    for m in record['methods']:
        base = _strip_ours(m)
        if len(base) >= 2 and base.lower() not in GENERIC: names.add(base)
        names.update(label_tokens(m))
    pm = paper_method_name(_paper_title(record.get('paper_id', '')))
    if pm and pm.lower() not in GENERIC: names.add(pm)
    return sorted(names, key=len, reverse=True)

def _name_re(names):
    if not names: return None
    alt = '|'.join(re.escape(n) for n in names)
    return re.compile(r'(?<![A-Za-z0-9])(?:' + alt + r')(?![A-Za-z0-9])(?:\s*\[\d+(?:\s*[,–-]\s*\d+)*\])?', re.I)

_CLAUSE = re.compile(r'\b(?:compared\s+(?:to|with)|as\s+compared\s+(?:to|with)|than|vs\.?|versus|over)(?=\s)(?!\s+ground[\s-]truth)[^;]*', re.I)  # keyword must be followed by a space (not 'over-segmentation'); keep 'compared to ground truth' (describes the reference, not a method)

def _tidy(s):
    s = re.sub(r'\(\s*\)|\[\s*\]', ' ', s)
    s = re.sub(r'\s+([,;:.])', r'\1', s)
    s = re.sub(r'([,;:])(\s*[,;:])+', r'\1', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,;:')
    # drop connectors left dangling at either end
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'(?:\s+|^)(?:and|or|with|to|of|the|a|an|by|in|than|vs|over|between|from|for|against|compared)\s*$', '', s, flags=re.I).strip(' ,;:')
        s = re.sub(r'^(?:and|or|,|;)\s+', '', s, flags=re.I).strip(' ,;:')
    return s

def sanitize_criterion(criterion, record):
    rx = _name_re(names_for(record))
    if rx is None: return criterion
    s = criterion
    s = _CLAUSE.sub(' ', s)
    s = rx.sub(' ', s)
    return _tidy(s)

def sanitize_question(record):
    """record: dict with 'repaired_question', 'methods', 'paper_id', 'leaf_name'."""
    rq = record['repaired_question']
    crit = rq[len(TEMPLATE):-1] if rq.startswith(TEMPLATE) and rq.endswith('?') else rq
    out = sanitize_criterion(crit, record)
    if out == crit: return rq
    if len(out.split()) < 3:
        out = (record.get('leaf_name') or '').strip().lower()
    return f'{TEMPLATE}{out}?'

def make_record(q, dp, full_question_fn):
    return {'id': q['id'], 'paper_id': q['paper_id'], 'methods': q['methods'], 'leaf_name': q.get('leaf_name', ''),
            'published_question': q['question'], 'repaired_question': full_question_fn(q, dp or {})}

def removed_terms(record):
    rq = record['repaired_question']; sq = sanitize_question(record)
    a = re.findall(r'[A-Za-z0-9][A-Za-z0-9.\-]*', rq); b = re.findall(r'[A-Za-z0-9][A-Za-z0-9.\-]*', sq)
    bl = [x.lower() for x in b]; out = []
    for t in a:
        if t.lower() in bl: bl.remove(t.lower())
        else: out.append(t)
    return out

def build_all():
    sys.path.insert(0, str(HERE)); import controls
    qs, dps = controls.load_question_sources(REPO/'data/bench/test_questions.jsonl', REPO/'data/bench/data_points.jsonl')
    ids = sorted(r['id'] for r in controls.eligible_predictions(controls.PRED/'test_questions_or_gpt-5.4_p1.jsonl'))
    rows = []
    for i in ids:
        q = qs[i]; rec = make_record(q, dps.get(q.get('v5_data_point_key')), controls.full_question)
        sq = sanitize_question(rec)
        rows.append({'id': i, 'published_question': rec['published_question'], 'repaired_question': rec['repaired_question'],
                     'sanitized_question': sq, 'changed': sq != rec['repaired_question'], 'removed_terms': removed_terms(rec)})
        rec['sanitized'] = sq
    return rows, qs, dps

def leaked_tokens(question, methods):
    low = question[len(TEMPLATE):].lower()
    return [t for m in methods for t in label_tokens(m) if re.search(r'(?<![a-z0-9])' + re.escape(t.lower()) + r'(?![a-z0-9])', low)]

def selftest():
    r = {'paper_id': 'x', 'methods': ['MobRecon', 'METRO', 'MeshGraphormer', 'Ours'], 'leaf_name': 'Surface Quality',
         'repaired_question': TEMPLATE + 'Hand mesh reconstruction accuracy and completeness compared to MobRecon, METRO, and MeshGraphormer?'}
    assert sanitize_question(r) == TEMPLATE + 'Hand mesh reconstruction accuracy and completeness?', sanitize_question(r)
    r2 = {'paper_id': 'x', 'methods': ['ALTO', 'Foo'], 'leaf_name': 'Surface Quality',
          'repaired_question': TEMPLATE + 'ALTO achieves sharper surfaces with fewer holes[27]?'}
    assert 'ALTO' not in sanitize_question(r2) and 'sharper surfaces' in sanitize_question(r2), sanitize_question(r2)
    r3 = {'paper_id': 'x', 'methods': ['A1', 'B1'], 'leaf_name': 'Blur', 'repaired_question': TEMPLATE + 'Sharper edges and fewer artifacts?'}
    assert sanitize_question(r3) == r3['repaired_question']
    r4 = {'paper_id': 'x', 'methods': ['Foo'], 'leaf_name': 'Surface Quality', 'repaired_question': TEMPLATE + 'Foo wins?'}
    assert sanitize_question(r4) == TEMPLATE + 'surface quality?'
    probe={'paper_id':'x','methods':['Ours','Voxel','PVD','DPC'],'leaf_name':'Quality','repaired_question':TEMPLATE+'Surface quality compared to voxel-based and point cloud baselines?'}
    assert 'compared' not in sanitize_question(probe).lower() and 'baselines' not in sanitize_question(probe).lower()
    rows, qs, _ = build_all()
    assert len(rows) == 309
    for row in rows:
        bad = leaked_tokens(row['sanitized_question'], qs[row['id']]['methods'])
        assert not bad, (row['id'], bad, row['sanitized_question'])
        assert row['sanitized_question'].startswith(TEMPLATE) and row['sanitized_question'].endswith('?')
    ch = [r for r in rows if r['changed']]
    print(f'selftest OK: {len(ch)}/309 questions changed')
    for r in ch[:10]: print('BEFORE:', r['repaired_question'], '\nAFTER: ', r['sanitized_question'], '\n')
    return rows

if __name__ == '__main__':
    if '--selftest' in sys.argv:
        rows = selftest()
    else:
        rows, _, _ = build_all()
    with (HERE/'sanitized_questions.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
