# Builds Tab. 4 (input controls E1-E4 per judge) from the saved control runs (no API calls).
# Rows are measures; each judge has an accuracy column and a change column (paired change from the standard
# condition, green = up, red = down, darker = larger; * = paper-bootstrap 95% CI excludes zero).
# Usage: python make_tab_controls.py OUT.tex [--audited]   (--audited: same subset as Tab. 3)
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'controls'))
import controls as C

RES = ROOT / 'results/controls'
MODELS = [('gemma_base', 'G-4B'), ('gemma-4-26b-a4b-it', 'G-26B'), ('gemma-4-31b-it', 'G-31B'),
          ('mimo-v2.5', 'MiMo'), ('gpt-5.3-codex', 'Codex'), ('gemini-3.1-pro-preview', 'Gemini')]
audited = '--audited' in sys.argv
recs = C.br.load_records(ROOT / 'data/bench/test_questions.jsonl',
                         ROOT / 'data/images', True, 0, include_context=True,
                         tag_quality=True, data_points_path=ROOT / 'data/bench/data_points.jsonl')
recs = [r for r in recs if r.quality_flags and r.quality_flags['benchmark_eligible']]
assert len(recs) == 309
if audited:
    AUD = set(open(ROOT / 'data/bench/audited_ids.txt').read().split())
    recs = [r for r in recs if r.id in AUD]
ids = {r.id for r in recs}

MODELS = [(k, n) for (k, _), n in zip(MODELS, ['Gemma-4-E4B', 'Gemma-4-26B-A4B', 'Gemma-4-31B', 'MiMo v2.5', 'GPT-5.3-codex', 'Gemini 3.1 Pro'])]
NA = None   # measure without a paired change

def delta(d):
    m, (lo, hi) = 100 * d['mean_difference'], d['ci95']
    txt = f"{m:+.1f}".replace('-', '$-$') + ('*' if lo > 0 or hi < 0 else '')
    return f"\\cellcolor{{{'up' if m > 0 else 'down'}!{min(60, 6 + 2 * abs(m)):.0f}}}{txt}"

acc = lambda rs: 100 * sum(r['correct'] for r in rs) / len(rs)
cols, done = [], []
for key, name in MODELS:
    p = RES / key / 'raw.jsonl'
    rows = [json.loads(l) for l in open(p)] if p.exists() else []
    bad = sum(r.get('status') not in ('ok', 'unparseable') for r in rows)   # unparseable = answered without a letter
    if len(rows) != 2574 or bad:
        print(f'skip {name}: {len(rows)}/2574 rows, {bad} failed calls'); continue
    rows = [r for r in rows if r['id'] in ids]
    s = C.summarize(rows, recs); pd = s['paired_differences']
    cond = lambda c: [r for r in rows if r['condition'] == c]
    ref = [r for r in cond('E4') if r.get('n_context_tiles')]
    cols.append([(100 * s['per_order_accuracy']['0'], NA),
                 (100 * s['e1_item_mean_accuracy'], delta(pd['e1_item_mean_vs_single_order'])),
                 (100 * s['majority_strict_accuracy'], delta(pd['majority_strict_vs_single_order'])),
                 (100 * s['within_item_consistency'], NA),
                 (100 * s['criterion_removed_accuracy'], delta(pd['criterion_removed'])),
                 (100 * s['blank_accuracy'], delta(pd['blank'])),
                 (100 * s['mismatched_accuracy'], delta(pd['mismatched'])),
                 (acc(cond('E4-text-only')), delta(pd['E4-text-only'])),
                 (acc(ref), delta(s['e4_split']['E4_minus_standard_with_reference']))])
    done.append(name)
n_ref = s['e4_split']['n_with_reference']
# (label, group heading before it or None, bold the best accuracy in the row?)
ROWS = [('Standard', None, True),
        ('Mean over orders', 'E1: option order', True), ('Plurality vote', None, True),
        ('Same crop in all orders', None, True),
        ('Criterion removed', 'E2: criterion', True),
        ('Blank tiles', 'E3: images', False), ('Mismatched composite', None, False),
        ('Full question', 'E4: restored inputs', True), (f'Full question + ref.$^\\dagger$', None, True)]
J = len(done)
sub = 'audited subset' if audited else 'all'
out = [r"\begin{table*}[t]", r"\centering",
       r"\caption{Input controls on VisionQ-Bench (" + ("audited subset, " if audited else "") + f"{len(ids)} questions)." + f" $\\Delta$: paired change from the standard run (\\colorbox{{up!35}}{{green}} up, \\colorbox{{down!35}}{{red}} down; * 95\\% paper-bootstrap interval excludes zero; --: n/a). Bold: best in the row. \\emph{{Same crop}}: same crop in every option order (random 13.8\\%). $^\\dagger$On the {n_ref} questions with a reference image." + r"}",
       r"\label{tab:controls" + ('-audited' if audited else '') + "}", r"\footnotesize", r"\setlength{\tabcolsep}{1pt}",
       r"\begin{tabularx}{\textwidth}{@{}l" + r">{\raggedleft\arraybackslash}X>{\centering\arraybackslash}X" * J + r"@{}}", r"\toprule",
       " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\scriptsize\\textbf{{{n}}}}}" for n in done) + r" \\",
       "".join(f"\\cmidrule(lr){{{2 + 2 * j}-{3 + 2 * j}}}" for j in range(J)),
       "Condition & " + " & ".join(["Acc." , "$\\Delta$"] * J) + r" \\", r"\midrule"]
for i, (lab, group, bold) in enumerate(ROWS):
    if group:
        out.append(r"\midrule" if i else "")
        out.append(f"\\multicolumn{{{1 + 2 * J}}}{{@{{}}l}}{{\\textbf{{{group}}}}} \\\\")
    best = max(round(c[i][0], 1) for c in cols)
    cells = []
    for c in cols:
        a, d = c[i]
        t = f"{a:.1f}"
        cells += [f"\\textbf{{{t}}}" if bold and round(a, 1) == best else t, d if d is not NA else "--"]
    out.append(lab + " & " + " & ".join(cells) + r" \\")
out += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
open(sys.argv[1], 'w').write("\n".join(l for l in out if l) + "\n")
print('wrote', sys.argv[1], 'judges:', done)
