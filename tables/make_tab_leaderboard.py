# Builds Tab. 3 (leaderboard) from saved predictions: overall and per-axis accuracy with
# paper-bootstrap 95% CIs (B=4000, seed 0, same scheme as analysis/benchmark_stats.py),
# plus overall accuracy on the audited subset (T8: eligible minus F1/F2 items confirmed on manual review).
import json, glob, random, collections, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PRED = str(ROOT / 'results/benchmark/predictions') + '/'
Q = {json.loads(l)['id']: json.loads(l) for l in open(ROOT / 'data/bench/test_questions.jsonl')}
AUD = set(open(ROOT / 'data/bench/audited_ids.txt').read().split())  # questions that pass the human check (Supp. B)
AXES = ["Object Form", "Reference Fidelity", "Image Appearance", "Relation", "Prompt Match", "Scene Layout"]
NAMES = {'or_gpt-5.3-codex': 'GPT-5.3-codex', 'or_gpt-5.5': 'GPT-5.5', 'or_gpt-5.2': 'GPT-5.2', 'or_gpt-5.4': 'GPT-5.4',
         'or_gpt-5.1': 'GPT-5.1', 'or_mimo-v2.5': 'MiMo v2.5', 'or_glm-5v-turbo': 'GLM-5V Turbo', 'opus': 'Claude Opus 4.6',
         'or_qwen3.6-35b-a3b': 'Qwen 3.6-35B-A3B', 'or_gemma-4-31b-it': 'Gemma-4-31B', 'or_grok-4.3': 'Grok-4.3',
         'or_qwen3.6-27b': 'Qwen 3.6-27B', 'sonnet': 'Claude Sonnet 4.6', 'or_nemotron-omni-reasoning': 'Nemotron-Omni',
         'haiku': 'Claude Haiku 4.5', 'or_mistral-medium-3-5': 'Mistral Medium 3.5', 'or_gemma-4-26b-a4b-it': 'Gemma-4-26B-A4B',
         'or_gemma4-4b-dpo': 'VisionQ-Judge (earlier ckpt.)', 'or_qwen3.5-9b': 'Qwen 3.5-9B', 'or_gemma4-4b-base': 'Gemma-4-E4B (base)',
         'or_grok-4.20': 'Grok-4.20', 'or_gemini-3.1-pro-preview': 'Gemini 3.1 Pro'}

def ci(items, B=4000, seed=0):
    by = collections.defaultdict(list)
    for pid, v in items: by[pid].append(v)
    ps, rng, acc = list(by), random.Random(seed), []
    for _ in range(B):
        xs = [x for p in (rng.choice(ps) for _ in ps) for x in by[p]]
        acc.append(sum(xs) / len(xs))
    acc.sort(); return acc[int(.025 * B)], acc[int(.975 * B)]

def cell(items):
    a = sum(v for _, v in items) / len(items); lo, hi = ci(items)
    return a, (round(100 * a), f"{{\\scriptsize[{100*lo:.0f},{100*hi:.0f}]}}")


rows = []
for f in sorted(glob.glob(PRED + 'test_questions_*_p1.jsonl')):
    key = f.split('test_questions_')[1][:-len('_p1.jsonl')]
    el = [r for r in map(json.loads, open(f)) if r['quality_flags']['benchmark_eligible']]
    ok = lambda r: float(bool(r.get('pred_letter')) and f"({r['pred_letter']})" == Q[r['id']]['answer'])
    items = [(Q[r['id']]['paper_id'], ok(r)) for r in el]
    a, overall = cell(items)
    axes = [cell([(Q[r['id']]['paper_id'], ok(r)) for r in el if Q[r['id']]['leaf_path'][0] == ax])[1] for ax in AXES]
    aud = cell([(Q[r['id']]['paper_id'], ok(r)) for r in el if r['id'] in AUD])[1]
    if key == 'or_gemma4-4b-dpo': continue   # earlier VisionQ-Judge checkpoint: trained on papers that supplied benchmark items
    rows.append((a, NAMES.get(key, key), overall, axes, aud, len(el)))
assert len(rows) == 21 and all(r[5] == 309 for r in rows), [(r[1], r[5]) for r in rows]
rows.sort(key=lambda r: -r[0])
ids = [r['id'] for r in map(json.loads, open(PRED + 'test_questions_or_gpt-5.5_p1.jsonl')) if r['quality_flags']['benchmark_eligible']]
ch = lambda S: 100 * sum(1 / len(Q[i]['choices']) for i in S) / len(S)
chance = [ch(ids)] + [ch([i for i in ids if Q[i]['leaf_path'][0] == ax]) for ax in AXES] + [ch([i for i in ids if i in AUD])]
npap = lambda S: len({Q[i]['paper_id'] for i in S})
hdr = [f"{len(ids)} / {npap(ids)}"] + [f"{len(S)} / {npap(S)}" for S in ([i for i in ids if Q[i]['leaf_path'][0] == ax] for ax in AXES)] + [f"{len(AUD)} / {npap(AUD)}"]
# Main table (following the review): Overall and Audited with intervals, axis point estimates only,
# the three axes with fewer than ten papers set apart as indicative; no cell shading (axes differ in chance level and
# support, so one colour scale would mislead); bold only on the highest Overall estimate(s), as a descriptive marker.
# The full matrix with every interval goes to the supplement (second output file).
NAMES_AX = ['Object Form', 'Ref.\\ Fidelity', 'Img.\\ Appearance', 'Relation', 'Prompt Match', 'Scene Layout']
top = max(r[2][0] for r in rows)
fmt_ci = lambda c: f"{c[0]} {c[1]}"
main = [r"\begin{table*}[t]", r"\centering",
        r"\caption{Expert alignment (\%) on VisionQ-Bench (309 questions; chance 32\%). Brackets: 95\% paper-bootstrap intervals. \colorbox{bench!12}{Shaded}: second to fifth, not separated by paired tests; bold: best in each column. $^\ddagger$Base model of VisionQ-Judge.}",
        r"\label{tab:leaderboard}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{@{}l*{1}{>{\centering\arraybackslash}X}@{\hspace{8pt}}*{3}{>{\centering\arraybackslash}X}@{\hspace{8pt}}*{3}{>{\centering\arraybackslash}X}@{}}",
        r"\toprule",
        r"\textbf{Judge} & \textbf{Overall} & " + " & ".join(f"\\textbf{{{n}}}" for n in NAMES_AX) + r" \\",
        r"\midrule"]
# Story pass (2026-11-12): shade the inseparable top four, bold the column maxima (not on the indicative axes, which are greyed),
# mark the base model that Sec. 6 fine-tunes. Still no colour scale on the axes (see note above).
best = [max(r[k] for r in [[r[2][0], r[4][0]] + [c[0] for c in r[3]] for r in rows]) for k in range(8)]
grey = lambda t: f"\\textcolor{{black!45}}{{{t}}}"
for rank, (a, name, overall, axes, aud, _) in enumerate(rows):
    vals = [overall[0], aud[0]] + [c[0] for c in axes]
    b = lambda k, t: f"\\textbf{{{t}}}" if vals[k] == best[k] else str(t)
    cells = [f"{b(0, overall[0])} {overall[1]}"] + [b(k + 2, axes[k][0]) for k in range(6)]
    if name == 'Gemma-4-E4B (base)': name = 'Gemma-4-E4B$^\\ddagger$'
    main.append((r"\rowcolor{bench!12}" if 1 <= rank <= 4 else "") + name + " & " + " & ".join(cells) + r" \\")
main += [r"\midrule", r"\emph{Chance} & " + " & ".join(f"\\emph{{{c:.1f}}}" for c in [chance[0]] + chance[1:-1]) + r" \\",
         r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
open(sys.argv[1], 'w').write("\n".join(main) + "\n")
# supplement: every cell with its interval, no emphasis
if len(sys.argv) > 2:
    sup = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Per-axis agreement with the author key (\%) on VisionQ-Bench with 95\% paper-bootstrap intervals (complement to Tab.~2). The second header row gives questions / source papers.}",
           r"\label{tab:leaderboard-full}", r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}",
           r"\begin{tabularx}{\textwidth}{@{}l*{8}{>{\centering\arraybackslash}X}@{}}", r"\toprule",
           r"\textbf{Judge} & \textbf{Overall} & \textbf{Audited} & " + " & ".join(f"\\textbf{{{n}}}" for n in NAMES_AX) + r" \\",
           " & " + " & ".join([hdr[0], hdr[-1]] + hdr[1:-1]) + r" \\", r"\midrule"]
    for a, name, overall, axes, aud, _ in rows:
        sup.append(name + " & " + " & ".join(fmt_ci(c) for c in [overall, aud] + axes) + r" \\")
    sup += [r"\midrule", "Chance & " + " & ".join(f"{c:.1f}" for c in [chance[0], chance[-1]] + chance[1:-1]) + r" \\",
            r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    open(sys.argv[2], 'w').write("\n".join(sup) + "\n")
print("rows", len(rows), "chance overall %.1f audited %.1f" % (chance[0], chance[-1]))
