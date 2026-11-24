#!/usr/bin/env python3
"""Offline analysis and (when explicitly invoked) VisionQ control runner.

Imports renderer, records, parser, prompt and run_one from the existing benchmark
runner. No request is made unless --run is supplied.
"""
from __future__ import annotations
import argparse, asyncio, importlib.util, json, os, random, sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT/'results/benchmark/predictions'
RUNNER_PATH = ROOT/'benchmark/bench_run.py'
sys.path.insert(0, str(ROOT/'benchmark'))
spec = importlib.util.spec_from_file_location('visionq_bench_run', RUNNER_PATH)
br = importlib.util.module_from_spec(spec); sys.modules[spec.name] = br; spec.loader.exec_module(br)

def eligible_predictions(path):
    rows=[json.loads(x) for x in path.open() if x.strip()]
    return [r for r in rows if (r.get('quality_flags') or {}).get('benchmark_eligible')]

def paper_id(row): return row['id'].split('__',1)[0]

def paper_bootstrap(diffs, B=10000, seed=0):
    """Paired mean difference CI, resampling source papers as clusters."""
    if not diffs:
        return {'mean_difference':None,'ci95':None,'B':B,'seed':seed,'n_items':0,'n_papers':0}
    by=defaultdict(list)
    for pid,d in diffs: by[pid].append(float(d))
    papers=list(by); rng=random.Random(seed); vals=[]
    for _ in range(B):
        sampled=[rng.choice(papers) for _ in papers]
        x=[d for p in sampled for d in by[p]]
        vals.append(sum(x)/len(x))
    vals.sort(); mean=sum(d for _,d in diffs)/len(diffs)
    return {'mean_difference':mean,'ci95':[vals[int(.025*B)], vals[min(B-1,int(.975*B))]],'B':B,'seed':seed,'n_items':len(diffs),'n_papers':len(by)}

def map_slot_to_crop(slot, crop_ids, shift=0):
    """Right-rotate crop list by shift; return crop occupying predicted slot."""
    n=len(crop_ids); return crop_ids[(slot-shift)%n]

def choose_vote(crop_votes, gold_crop):
    """Fixed: correct iff gold is unique winner. Also report permissive tie accuracy."""
    if not crop_votes: return {'winner':None,'strict_correct':False,'tie_includes_gold':False,'tie':False}
    counts={x:crop_votes.count(x) for x in set(crop_votes)}; top=max(counts.values())
    leaders=[x for x,n in counts.items() if n==top]
    return {'winner':leaders[0] if len(leaders)==1 else None,
            'strict_correct':len(leaders)==1 and leaders[0]==gold_crop,
            'tie_includes_gold':len(leaders)>1 and gold_crop in leaders,'tie':len(leaders)>1}

def accuracy(rows): return sum(bool(r.get('correct')) for r in rows)/len(rows) if rows else 0.0

def saved_check():
    paths=[PRED/'test_questions_or_gemma4-4b-base_p1.jsonl',PRED/'test_questions_or_gpt-5.5_p1.jsonl']
    out={}
    for p in paths:
        rows=eligible_predictions(p); out[p.stem]={"n":len(rows),"accuracy":accuracy(rows),"percent":100*accuracy(rows)}
    return out

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def load_question_sources(questions_path, data_points_path):
    """Load canonical questions and their full visual-attribute sources."""
    questions={}
    for line in questions_path.open():
        if line.strip():
            row=json.loads(line); questions[row['id']]=row
    dps={}
    for line in data_points_path.open():
        if line.strip():
            row=json.loads(line); dps[row['data_point_key']]=row
    return questions,dps

def full_question(q, dp):
    """Restore a clipped visual-attribute criterion without adding claim text."""
    original=q.get('question','')
    if '...' not in original:
        return original
    attr=q.get('v5_visual_attr') or ''
    full=(dp.get('taxonomy') or {}).get('visual_attribute') or (dp.get('claim') or {}).get('visual_attribute_text') or ''
    # The data-point field may contain alternatives; the first is the canonical
    # source phrase when it extends the clipped v5_visual_attr prefix.
    criterion=full.split(';',1)[0].strip()
    prefix=attr[:-3] if attr.endswith('...') else ''
    if not prefix or not criterion.startswith(prefix):
        raise ValueError(f"Cannot recover visual criterion for {q.get('id')}")
    return f"Which method demonstrates better {criterion}?"

def context_candidate_invariant(rec):
    allowed={'GT','INPUT','REF','TARGET','REAL','SCAN'}
    return (len(rec.image_paths)==rec.n_choices and
            all(t.label in allowed for t in (rec.context_tiles or [])) and
            not any(t.path in rec.image_paths for t in (rec.context_tiles or [])))

async def run(args):
    records=br.load_records(args.questions,args.dataset_root,True,0,include_context=True,tag_quality=True,data_points_path=args.data_points)
    records=[r for r in records if r.quality_flags and r.quality_flags['benchmark_eligible']]
    if len(records)!=309:
        raise SystemExit(f'Preflight failed: loaded {len(records)} eligible records; expected 309; no calls made')
    missing=[r.id for r in records if any(not p.is_file() for p in r.image_paths)]
    if missing: raise SystemExit(f'Preflight failed: {len(missing)} eligible records lack candidate crops; no calls made')
    ctx=sum(bool(r.context_tiles) for r in records)
    print(f'Preflight: loaded {len(records)} items; E4 context rows {ctx} (of 211 reference-context items)')
    if ctx < args.min_context_rows: raise SystemExit(f'Preflight failed: E4 would receive {ctx} context rows, fewer than --min-context-rows {args.min_context_rows}; no calls made')
    if args.dry_run:
        class OfflineClient:
            async def generate(self, **kwargs): return {'text':'(A)','usage':{'input_tokens':0,'output_tokens':0},'latency_ms':0}
        client=OfflineClient()
    else:
        if args.provider=='openrouter':
            from lib.openrouter_vlm import OpenRouterVLMClient; client=OpenRouterVLMClient(model_id=args.model)
        elif args.provider=='bedrock':
            from lib.bedrock_vlm import BedrockVLMClient; client=BedrockVLMClient(model_id=args.model)
        else: client=LocalGemmaClient(args.model,args.adapter,args.load_in_4bit,args.fp16)
    qsrc,dps=load_question_sources(args.questions,args.data_points)
    sys.path.insert(0,str(Path(__file__).resolve().parent)); import sanitize
    repaired={r.id:sanitize.sanitize_question(sanitize.make_record(qsrc[r.id],dps.get(r.data_point_key),full_question)) for r in records}
    if not all(context_candidate_invariant(r) for r in records):
        raise AssertionError('context/candidate separation invariant failed')
    by_n=defaultdict(list)
    for r in records: by_n[r.n_choices].append(r)
    # Fixed seed and same-choice-count mismatch, excluding self where possible.
    rng=random.Random(args.seed); mismatches={}
    for n,rs in by_n.items():
        order=rs[:]; rng.shuffle(order)
        for i,r in enumerate(order):
            candidates=[x for x in order if x.id!=r.id and x.id.split('__')[0:2]!=r.id.split('__')[0:2]]
            if not candidates: raise ValueError(f'No different-figure mismatch partner for {r.id}')
            mismatches[r.id]=candidates[i%len(candidates)]
    # create same-size white tiles once. Renderer is still the benchmark renderer.
    from PIL import Image
    blankdir=args.out_dir/'.blank_tiles'; blankdir.mkdir(parents=True,exist_ok=True)
    blankpath=blankdir/'white.png'; Image.new('RGB',(512,512),'white').save(blankpath)
    sem=asyncio.Semaphore(args.concurrency)
    async def once(rec, name, shift=0, question=None, render_rec=None):
        source=render_rec or rec
        imgs=list(source.image_paths)
        gold=rec.answer_letter
        rr=rec
        if shift:
            n=len(imgs); imgs=[imgs[(i-shift)%n] for i in range(n)]
            gi=(ord(gold)-65+shift)%n; gold=chr(65+gi)
        if name=='blank': imgs=[blankpath]*rec.n_choices
        rr=replace(rec,image_paths=imgs,answer_letter=gold,
                   question=question if question is not None else rec.question,
                   context_tiles=(rec.context_tiles or []) if name=='E4' else [])
        row=await br.run_one(client,rr,sem,args.max_tokens,True,args.retries)
        # crop identity mapping is calculated from original ordering/shift.
        row.update({'condition':name,'order_shift':shift,'gold_crop_index':ord(rec.answer_letter)-65})
        return row
    outpath=args.out_dir/'raw.jsonl'; args.out_dir.mkdir(parents=True,exist_ok=True)
    done=set()
    if outpath.exists() and outpath.stat().st_size and not args.resume:
        raise SystemExit(f'{outpath} already has rows; pass --resume to continue it or choose a new --out-dir (refusing to append duplicates)')
    if args.resume and outpath.exists():
        for line in outpath.open():
            if not line.strip(): continue
            try: x=json.loads(line)
            except json.JSONDecodeError:
                print('warning: skipping a torn line in raw.jsonl', file=sys.stderr); continue
            if str(x.get('status','')).startswith('error'): continue   # failed calls are retried on resume
            done.add((x.get('id'),x.get('condition'),x.get('order_shift',0)))
    allrows=[]
    run_recs=records[:args.limit] if args.limit else records   # --limit: smoke test only
    for i,rec in enumerate(run_recs,1):
        requests=[('perm',s,None,None) for s in range(rec.n_choices)]+[
            ('criterion_removed',0,'Which output is better overall?',None),('blank',0,None,None),
            ('mismatched',0,None,mismatches[rec.id]),('E4',0,repaired[rec.id],None),('E4-text-only',0,repaired[rec.id],None)]
        tasks=[once(rec,nm,sh,question=q,render_rec=rr) for nm,sh,q,rr in requests if (rec.id,nm,sh) not in done]
        fresh=await asyncio.gather(*tasks,return_exceptions=True)
        with outpath.open('a') as f:
            for request,result in zip([x for x in requests if (rec.id,x[0],x[1]) not in done],fresh):
                if isinstance(result,Exception): result={'id':rec.id,'condition':request[0],'order_shift':request[1],'pred_letter':None,'correct':False,'status':'error:'+type(result).__name__,'error_msg':str(result)}
                allrows.append(result); f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush()
        if i%20==0: print(f'{i}/{len(records)} items',file=sys.stderr)
    latest={}
    for line in outpath.open():
        if not line.strip(): continue
        try: x=json.loads(line)
        except json.JSONDecodeError: continue
        latest[(x.get('id'),x.get('condition'),x.get('order_shift',0))]=x   # a retried call supersedes its error row
    allrows=list(latest.values())
    summary=summarize(allrows,run_recs)
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2))

class LocalGemmaClient:
    """Thin generate()-interface adapter for Gemma-4 using HF Transformers."""
    def __init__(self, model_id, adapter=None, load_in_4bit=False, fp16=False):
        import torch
        dt=torch.float16 if fp16 else torch.bfloat16   # Turing GPUs (RTX 8000) have no native bf16
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        self.torch=torch; self.processor=AutoProcessor.from_pretrained(model_id,trust_remote_code=True)
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=dt,bnb_4bit_use_double_quant=True) if load_in_4bit else None
        self.model=AutoModelForImageTextToText.from_pretrained(model_id,dtype=dt,device_map='auto',quantization_config=q,trust_remote_code=True,attn_implementation='eager')
        if adapter:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,str(adapter))
        self.model.eval()
    async def generate(self,system,user_text,images,image_format='png',temperature=0.0,max_tokens=None,label_images=False):
        from PIL import Image
        import io
        pil=[Image.open(io.BytesIO(b)).convert('RGB') for b in images]
        msgs=[{'role':'user','content':[{'type':'image'}, {'type':'text','text':system+'\n\n'+user_text}]}]
        prompt=self.processor.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
        inputs=self.processor(text=prompt,images=pil,return_tensors='pt').to(self.model.device)
        ids=self.model.generate(**inputs,max_new_tokens=max_tokens or 128,do_sample=False)
        tail=ids[0][inputs['input_ids'].shape[-1]:]
        return {'text':self.processor.decode(tail,skip_special_tokens=True),'reasoning':'','reasoning_details':[],
                'raw_message':{},'usage':{'input_tokens':int(inputs['input_ids'].shape[-1]),'output_tokens':int(tail.shape[-1])},'latency_ms':0}

def summarize(rows,records):
    groups=defaultdict(list)
    for r in rows: groups[r['condition']].append(r)
    byid=defaultdict(list)
    for r in groups['perm']: byid[r['id']].append(r)
    # Map through the exact right rotation: slot chosen -> original crop index.
    per_order=[{'id':r['id'],'shift':r['order_shift'],'correct':r['correct']} for r in groups['perm']]
    votes=[]; same=[]; last=0; first=0; total=0; unparsed=0; vote_den=0
    gold={r.id:ord(r.answer_letter)-65 for r in records}
    for rid,rs in byid.items():
        crops=[(ord(r['pred_letter'])-65-r['order_shift'])%r['n_choices'] for r in rs if r['pred_letter']]
        if len(crops)==len(rs): same.append(len(set(crops))==1)
        unparsed += len(rs)-len(crops); vote_den += len(rs)
        vote=choose_vote(crops,gold[rid])
        if len(crops)!=len(rs): vote={'winner':None,'strict_correct':False,'tie_includes_gold':False,'tie':False}
        votes.append(vote)
        if crops:
            last+=sum((ord(r['pred_letter'])-65)==r['n_choices']-1 for r in rs if r['pred_letter'])
            first+=sum((ord(r['pred_letter'])-65)==0 for r in rs if r['pred_letter']); total+=len(crops)
    base={r['id']:r for r in groups['perm'] if r['order_shift']==0}
    by_shift=defaultdict(list)
    for r in groups['perm']: by_shift[r['order_shift']].append(r)
    out={'n_items':len(records),'n_orders':len(per_order),'per_order_accuracy':{str(s):accuracy(rs) for s,rs in sorted(by_shift.items())},
         'majority_strict_accuracy':sum(v['strict_correct'] for v in votes)/len(votes),
         'majority_tie_as_correct_accuracy':sum(v['strict_correct'] or v['tie_includes_gold'] for v in votes)/len(votes),
         'within_item_consistency':sum(same)/len(same) if same else None,'within_item_consistency_n':len(same),
         'last_slot_share':last/total if total else None,'last_slot_denominator':total,
         'first_slot_share':first/total if total else None,
         'baseline_always_first_accuracy_order0':sum(ord(r.answer_letter)==65 for r in records)/len(records),
         'baseline_always_last_accuracy_order0':sum(ord(r.answer_letter)-65==r.n_choices-1 for r in records)/len(records),
         'baseline_fixed_slot_accuracy_all_orders':sum(1 for r in records)/sum(r.n_choices for r in records),
         'unparsed_orders':unparsed,'vote_order_denominator':vote_den,
         'criterion_removed_accuracy':accuracy(groups['criterion_removed']),
         'blank_accuracy':accuracy(groups['blank']),'mismatched_accuracy':accuracy(groups['mismatched']),
         'per_order':per_order}
    out['usage']={k:sum(sum(int(a.get(k,0) or 0) for a in r.get('all_attempts',[])) if r.get('all_attempts') else int(r.get(k,0) or 0) for r in rows) for k in ('input_tokens','output_tokens')}
    out['usage']['logical_calls']=len(rows); out['usage']['image_blocks']=len(rows)
    out['error_rows_by_condition']={c:sum(str(r.get('status','')).startswith('error') for r in rs) for c,rs in groups.items()}
    out['usage']['billable_attempts']=sum(sum(a.get('status') in ('ok','unparseable') for a in r.get('all_attempts',[])) for r in rows)
    out['paired_differences']={}
    # For permutation summaries, average correctness over each item's cyclic orders,
    # then compare with that item's identity-order result using paper clusters.
    item_mean={rid:sum(int(x['correct']) for x in rs)/len(rs) for rid,rs in byid.items()}
    out['paired_differences']['permutation_mean_vs_single_order']=paper_bootstrap(
        [(paper_id(base[rid]),item_mean[rid]-int(base[rid]['correct'])) for rid in base])
    out['paired_differences']['majority_strict_vs_single_order']=paper_bootstrap(
        [(paper_id(base[rid]),int(votes[j]['strict_correct'])-int(base[rid]['correct'])) for j,(rid,_) in enumerate(byid.items())])
    out['e1_item_mean_accuracy']=sum(item_mean.values())/len(item_mean) if item_mean else None
    out['e1_item_mean_denominator']=len(item_mean)
    out['e1_item_mean_chance']=sum(1/r.n_choices for r in records)/len(records) if records else None
    for shift,rs in sorted(by_shift.items()):
        chosen={r['id']:r for r in rs}
        out['paired_differences'][f'order_{shift}_vs_single_order']=paper_bootstrap(
            [(paper_id(base[rid]),int(chosen[rid]['correct'])-int(base[rid]['correct'])) for rid in base if rid in chosen])
    out['paired_differences']['e1_item_mean_vs_single_order']=paper_bootstrap(
        [(paper_id(base[rid]),item_mean[rid]-int(base[rid]['correct'])) for rid in base if rid in item_mean])
    for cond in ('criterion_removed','blank','mismatched','E4','E4-text-only'):
        c={r['id']:r for r in groups[cond]}
        out['paired_differences'][cond]=paper_bootstrap([(paper_id(r),int(c[r['id']]['correct'])-int(r['correct'])) for r in base.values() if r['id'] in c])
    # E4 split: items whose E4 input carried a reference row vs items where E4 equals E4-text-only (N7)
    e4={r['id']:r for r in groups['E4']}; e4t={r['id']:r for r in groups['E4-text-only']}
    ctx_ids=[i for i,r in e4.items() if r.get('n_context_tiles')]; noctx_ids=[i for i in e4 if i not in set(ctx_ids)]
    out['e4_split']={'n_with_reference':len(ctx_ids),'n_without_reference':len(noctx_ids),
        'E4_minus_standard_with_reference':paper_bootstrap([(paper_id(base[i]),int(e4[i]['correct'])-int(base[i]['correct'])) for i in ctx_ids if i in base]),
        'E4_minus_text_only_with_reference':paper_bootstrap([(paper_id(base[i]),int(e4[i]['correct'])-int(e4t[i]['correct'])) for i in ctx_ids if i in base and i in e4t])}
    rec_by_id={r.id:r for r in records}
    out['per_axis']={}
    for axis in sorted({r.axis or '_none' for r in records}):
        ids=[r.id for r in records if (r.axis or '_none')==axis]
        cells={}
        for cond in ('criterion_removed','blank','mismatched','E4','E4-text-only'):
            c={r['id']:r for r in groups[cond]}
            rs=[c[i] for i in ids]
            cells[cond]={'accuracy':accuracy(rs),'n':len(rs)}
        cells['baseline']={'accuracy':accuracy([base[i] for i in ids]),'n':len(ids)}
        for cond in ('criterion_removed','E4','E4-text-only'):
            c={r['id']:r for r in groups[cond]}
            cells[cond+'_difference']=paper_bootstrap([(paper_id(base[i]),int(c[i]['correct'])-int(base[i]['correct'])) for i in ids])
        out['per_axis'][axis]=cells
    return out

def selftest():
    assert [map_slot_to_crop(s,['x','y','z'],1) for s in range(3)]==['z','x','y']
    assert choose_vote(['g','x'],'g')['strict_correct'] is False
    assert choose_vote(['g','x'],'g')['tie_includes_gold'] is True
    # Two-choice items have only orders 0 and 1; mixed-order summaries must not
    # assume that every item appears in every global shift bucket.
    mixed=[br.Record(f'p{i}__fig_1__x','q',['A','B'],'A',[Path('/a'),Path('/b')],['m1','m2'],'ax','leaf',2) for i in range(2)]
    fake=[]
    for r in mixed:
        for sh in range(r.n_choices):
            fake.append({'id':r.id,'condition':'perm','order_shift':sh,'correct':True,'pred_letter':chr(65+sh),'n_choices':2})
        for condition in ('criterion_removed','blank','mismatched','E4','E4-text-only'):
            fake.append({'id':r.id,'condition':condition,'correct':True,'n_choices':2})
    assert summarize(fake,mixed)['per_order_accuracy']=={'0':1.0,'1':1.0}
    d=paper_bootstrap([('p1',1),('p1',1),('p2',-1)],B=1000,seed=0)
    assert d['mean_difference']==1/3 and d['ci95'][0] <= 1/3 <= d['ci95'][1]
    # Context paths are separate from candidate slots and labels are generic.
    dummy=br.Record('x','q',['(A)','(B)'],'A',[Path('/a'),Path('/b')],
                    ['Method One','Method Two'],'axis','leaf',2,
                    context_tiles=[br.ContextTile(Path('/context'), 'GT')])
    assert context_candidate_invariant(dummy)
    assert not any(t.label in dummy.methods for t in dummy.context_tiles)
    # Render a synthetic composite and verify the context image is above the
    # candidate images, with separate gray and black banner rows.
    from PIL import Image
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        td=Path(td); candidates=[]
        for name,color in [('a',(0,180,0)),('b',(0,0,200)),('ctx',(220,0,0))]:
            pth=td/f'{name}.png'; Image.new('RGB',(80,80),color).save(pth)
            if name!='ctx': candidates.append(pth)
            else: ctx_path=pth
        composite=Image.open(__import__('io').BytesIO(br.build_composite_grid(
            candidates,[br.ContextTile(ctx_path,'GT')]))).convert('RGB')
        pixels=composite.load(); w,h=composite.size
        ys={color:[] for color in [(0,180,0),(0,0,200),(220,0,0)]}
        colors=set(ys)
        for y in range(h):
            for x in range(w):
                if pixels[x,y] in colors: ys[pixels[x,y]].append(y)
        assert all(ys.values()) and max(ys[(220,0,0)]) < min(ys[(0,180,0)])
        assert max(ys[(220,0,0)]) < min(ys[(0,0,200)])
        assert (102,102,102) in set(pixels[x,0] for x in range(w)) or any(
            pixels[x,y]==(102,102,102) for y in range(h) for x in range(w))
        assert any(pixels[x,y]==(0,0,0) for y in range(h) for x in range(w))
    qsrc,dps=load_question_sources(ROOT/'data/bench/test_questions.jsonl',ROOT/'data/bench/data_points.jsonl')
    eligible={r['id'] for r in eligible_predictions(PRED/'test_questions_or_gpt-5.4_p1.jsonl')}
    unresolved=[]
    for rid in eligible:
        q=qsrc[rid]
        try: full_question(q,dps.get(q.get('v5_data_point_key'),{}))
        except ValueError: unresolved.append(rid)
    assert len([rid for rid in eligible if '...' in qsrc[rid].get('question','')])==126
    assert not unresolved, unresolved
    sys.path.insert(0,str(Path(__file__).resolve().parent)); import sanitize
    assert sanitize.sanitize_question(sanitize.make_record(qsrc[next(iter(eligible))],dps.get(qsrc[next(iter(eligible))].get('v5_data_point_key')),full_question)).startswith('Which method demonstrates better ')
    return saved_check()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--selftest',action='store_true'); p.add_argument('--run',action='store_true'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--resume',action='store_true'); p.add_argument('--min-context-rows',type=int,default=205)
    p.add_argument('--provider',choices=['openrouter','bedrock','local'],default='openrouter'); p.add_argument('--model',default='')
    p.add_argument('--adapter',type=Path); p.add_argument('--load-in-4bit',action='store_true'); p.add_argument('--fp16',action='store_true'); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--questions',type=Path,default=ROOT/'data/bench/test_questions.jsonl')
    p.add_argument('--dataset-root',type=Path,default=ROOT/'data/images')
    p.add_argument('--data-points',type=Path,default=ROOT/'data/bench/data_points.jsonl')
    p.add_argument('--out-dir',type=Path,default=Path('outputs/control_run')); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--concurrency',type=int,default=4); p.add_argument('--retries',type=int,default=2); p.add_argument('--max-tokens',type=int)
    a=p.parse_args()
    if a.selftest: print(json.dumps(selftest(),indent=2)); return
    if a.run:
        if not a.model and not a.dry_run: p.error('--model required with --run')
        asyncio.run(run(a)); return
    p.error('choose --selftest or --run')
if __name__=='__main__': main()
