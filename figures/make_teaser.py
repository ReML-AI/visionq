# Fig. 1 teaser (full width): one VisionQ-Bench question as the judge sees it, the hidden key, and who picks what
# under the standard condition, criterion removed (E2) and reference restored (E4). Item: paper_0452__fig_5_p6__row_1.
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
from paperstyle import plt, PAL, INK, MUTED
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = str(ROOT); OUT = str(ROOT / 'figures')
CROPS = str(ROOT / 'data/images/paper_0452/figures/fig_5_p6/crops') + '/'
ITEM = 'paper_0452__fig_5_p6__row_1'
FONT = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf', 34)

def tile(path, label, bar='black', s=260):
    im = ImageOps.pad(Image.open(path).convert('RGB'), (s, s), color='white')
    t = Image.new('RGB', (s, s + 46), 'white'); d = ImageDraw.Draw(t)
    d.rectangle([0, 0, s, 44], fill=bar); d.text((s / 2, 22), label, fill='white', font=FONT, anchor='mm'); t.paste(im, (0, 46)); return t

def grid(tiles, cols, gap=8):
    w, h = tiles[0].size; rows = (len(tiles) + cols - 1) // cols
    g = Image.new('RGB', (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), 'white')
    for k, t in enumerate(tiles): g.paste(t, ((k % cols) * (w + gap), (k // cols) * (h + gap)))
    return g

comp = grid([tile(CROPS + f, f'({l})') for l, f in zip('ABCD', ['bbox_23_ours.png', 'bbox_16_metro.png', 'bbox_21_mobrecon.png', 'bbox_22_meshgraphormer.png'])], 2)
ref = Image.open(f'{R}/figures/assets/{ITEM}_reference.png').convert('RGB'); ref = ref.crop((0, 0, ref.width, 575))

J = [('gemma_base', 'G-4B'), ('gemma-4-26b-a4b-it', 'G-26B'), ('gemma-4-31b-it', 'G-31B'), ('mimo-v2.5', 'MiMo'), ('gpt-5.3-codex', 'Codex'), ('gemini-3.1-pro-preview', 'Gemini')]
P = {}
for k, _ in J:
    for l in open(f'{R}/results/controls/{k}/raw.jsonl'):
        r = json.loads(l)
        if r['id'] == ITEM and (r['condition'] != 'perm' or r['order_shift'] == 0): P[k, r['condition']] = r['pred_letter'] or '–'

f = plt.figure(figsize=(6.875, 2.0))
def ax(x, y, w, h): a = f.add_axes([x, y, w, h]); a.axis('off'); return a
def T(a, x, y, s, **k):
    return a.text(x, y, s, fontsize=k.pop('fs', 6.6), color=k.pop('color', INK), va=k.pop('va', 'top'), ha=k.pop('ha', 'left'), transform=a.transAxes, linespacing=1.15, **k)
# column 1: composite
a = ax(0, 0, .215, .88); a.imshow(comp)
T(ax(0, .88, .215, .12), 0, .8, 'What the judge sees', fs=7.2, fontweight='bold')
# column 2: question, hidden key, reference row
b = ax(.235, 0, .30, 1)
T(b, 0, .98, 'Question', fs=7.2, fontweight='bold')
T(b, 0, .885, '“Which method demonstrates better hand mesh\nreconstruction accuracy and completeness?”', style='italic')
T(b, 0, .70, 'Hidden from the judge', fs=7.2, fontweight='bold')
T(b, 0, .605, '(A) the paper’s method (key)   (B) METRO\n(C) MobRecon   (D) MeshGraphormer\nCriterion leaf: Relation / Correspondence')
T(b, 0, .355, 'Reference row, added in E4', fs=7.2, fontweight='bold')
r = ax(.235, -.01, .19, .245); r.imshow(ref)
# column 3: answer table
c = ax(.555, 0, .445, 1)
T(c, 0, .98, 'Choices (correct = A)', fs=7.2, fontweight='bold')
heads = ['Annot.'] + [n for _, n in J]; X0, CW = .37, .09
for k, h in enumerate(heads): T(c, X0 + k * CW, .85, h, fs=5.7, fontweight='bold', ha='center')
rows = [('Standard', ['A A'] + [P[k, 'perm'] for k, _ in J]),
        ('Criterion removed (E2)', [''] + [P[k, 'criterion_removed'] for k, _ in J]),
        ('Full question only (E4)', [''] + [P[k, 'E4-text-only'] for k, _ in J]),
        ('+ reference row (E4)', ['A A'] + [P[k, 'E4'] for k, _ in J])]
for i, (lab, vals) in enumerate(rows):
    y = .68 - i * .14
    T(c, 0, y, lab, fs=6.4, va='center', color=MUTED)
    for k, v in enumerate(vals):
        if not v: continue
        ok = set(v.replace(' ', '')) == {'A'}; col = PAL['green' if ok else 'red']
        c.add_patch(plt.Rectangle((X0 + k * CW - .042, y - .052), .084, .104, color=col[0], lw=0, transform=c.transAxes))
        T(c, X0 + k * CW, y, v, fs=6.8, va='center', ha='center', fontweight='bold', color=col[1])
T(c, 0, .17, 'Annotators answer without, then with the reference. No judge picks\nthe key in the standard run; with the reference row, five of six do.', fs=6.2, color=MUTED)
f.savefig(f'{OUT}/teaser.pdf', dpi=300); f.savefig(f'{OUT}/teaser.png', dpi=200); print('wrote teaser')
