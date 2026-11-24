#!/usr/bin/env python3
"""Letter probabilities for every cyclic order of each MCQ question, for PriDe-style debiasing.

Uses evaluate.py's prompt and tiling exactly, then reads the next-token distribution after the
model's answer prefix "(" and keeps the probabilities of the option letters.
Usage: python pride_logits.py --questions X_perm.jsonl --out out.jsonl [--adapter output/Cfix_s42/final]
"""
import argparse, json, sys
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev

p = argparse.ArgumentParser()
p.add_argument('--questions', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
p.add_argument('--adapter'); p.add_argument('--dataset-root', type=Path, default=Path('dataset'))
p.add_argument('--model-id', default='google/gemma-4-E4B-it')
a = p.parse_args()

proc = AutoProcessor.from_pretrained(a.model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(a.model_id, torch_dtype=torch.float16, device_map='auto',
                                                    trust_remote_code=True, attn_implementation='eager').eval()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).eval()

tok = proc.tokenizer
open_id = tok.encode('(', add_special_tokens=False)
letter_ids = []
for L in 'ABCD':
    ids = tok.encode(f'({L})', add_special_tokens=False)
    assert ids[:len(open_id)] == open_id and len(ids) == len(open_id) + 2, (L, ids)   # "(" + letter + ")"
    letter_ids.append(ids[len(open_id)])

done = set()
if a.out.exists():
    done = {json.loads(l)['id'] for l in a.out.open() if l.strip()}
qs = [json.loads(l) for l in a.questions.open()]
with a.out.open('a') as f, torch.no_grad():
    for i, q in enumerate(qs):
        if q['id'] in done: continue
        imgs = [Image.open(ev.resolve_image(x, a.dataset_root)).convert('RGB') for x in q['image_paths']]
        imgs = ev._tile_for_eval(imgs, 512) if len(imgs) > 1 else imgs
        text = ev.build_prompt_text(q, prepend_context=True, anti_bias=True)
        text += f"\n\nAnswer with one of {', '.join(q['choices'])} on the first line, then optionally explain."
        msgs = [{'role': 'user', 'content': [{'type': 'image'} for _ in imgs] + [{'type': 'text', 'text': text}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + '('
        inp = proc(text=prompt, images=imgs, return_tensors='pt').to(model.device)
        logits = model(**inp).logits[0, -1].float()
        n = len(q['choices'])
        lp = torch.log_softmax(logits, -1)[letter_ids[:n]].tolist()
        f.write(json.dumps({'id': q['id'], 'base_id': q.get('base_id', q['id']), 'shift': q.get('shift', 0),
                            'paper_id': q['paper_id'], 'n': n, 'answer': q['answer'].strip('()'), 'logp': lp}) + '\n')
        f.flush()
        if (i + 1) % 200 == 0: print(f'{i+1}/{len(qs)}', flush=True)
print('done', a.out)
