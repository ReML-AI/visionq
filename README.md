# VisionQ

Paper: [arXiv:2610.00666](https://arxiv.org/abs/2610.00666) · Data and weights: [huggingface.co/visionq-review](https://huggingface.co/visionq-review)

Code for *VisionQ: Benchmarking VLM Judges on Criterion-Specific Comparisons of Computer Vision Method Outputs*. It runs VLM judges on VisionQ-Bench, runs the input controls, trains and evaluates VisionQ-Judge, and rebuilds the paper's tables from saved predictions.

## Setup

```bash
pip install -r requirements.txt
hf download visionq-review/VisionQ --repo-type dataset --local-dir data   # questions, splits, images (pip install huggingface_hub)
cp .env.example .env                                                      # API keys, only for new judge runs
```

The dataset goes in `data/` (`data/bench`, `data/mcq`, `data/images`, ...). Every script finds it there. Nothing below needs an API key or a GPU unless marked.

## Paper to code

| Paper | What it shows | Command | Inputs |
|---|---|---|---|
| Sec. 3.2, App. B | rule-based filter (332 to 309 questions) | `python benchmark/bench_run.py ... --tag-quality` | `data/bench/` |
| Tab. 2, Tab. S6 | expert alignment of all 21 judges, per axis, 95% paper-bootstrap intervals (Tab. 2 shows nine of them) | `python tables/make_tab_leaderboard.py tab2.tex supp_full.tex` | `results/benchmark/predictions/` |
| Sec. 5 (IQA) | no-reference IQA baselines (CLIP-IQA+ 41.7, TOPIQ 40.8, MUSIQ 40.1, sharpness 36.2) | `python baselines/iqa_baseline.py --from-scores results/iqa/iqa_scores.json` | `results/iqa/` |
| Tab. 3, App. C | input controls E1 to E4 for six judges | `python tables/make_tab_controls.py tab3.tex` | `results/controls/` |
| App. D (audited controls) | the same on the 254 audited questions | `python tables/make_tab_controls.py tab3_aud.tex --audited` | same |
| App. D (mirror test) | accuracy with original and mirrored crops for three judges | `python -c "import json,glob; [print(f, json.load(open(f))['mirrored_minus_standard']) for f in glob.glob('results/recognition/*/summary.json')]"`; new run: `python controls/recognition.py --run --model <slug> --out-dir <dir>` (API key) | `results/recognition/` |
| Tab. 5 (a) | VisionQ-Judge vs base, reordering ablation, last-option share | `cd results/judge/test && python ../../../judge/stats.py .` | `results/judge/test/` |
| Tab. 5 (b) | VisionQ-Judge minus each larger judge | `cd results/judge/test && python ../../../judge/seedmean.py "Cfix_s42,Cfix_s43,Cfix_s44:api_gpt-5.5" ...` | same |
| Sec. 6 | VisionQ-Judge vs larger judges (paired) | `cd results/judge/test && python ../../../judge/seedmean.py "Cfix_s42,Cfix_s43,Cfix_s44:api_gpt-5.3-codex"` | same |
| Tab. 5 (c) | PriDe and log-prob averaging over option orders | `cd results/judge/logprobs && python ../../../judge/pride_analyze.py` | `results/judge/logprobs/`, `results/judge/e2/` |
| Fig. 3 | the six controlled judges' choices on the Fig. 1 question (`paper_0226__fig_1_p0__row_2`) under each condition | read `pred_letter` for this id in the control predictions | `results/controls/` |

Figs. 1, 2 and 4 are drawn diagrams with no script, and Fig. 3 is drawn from the predictions listed above. Fig. 5 plots VisionQ-Judge on the VisionQ-MCQ test split: panels (a) and (b) use `results/judge/test/`, and panel (c) uses predictions on every cyclic option order, which are not in this release. Tab. 4 (human study) and the raters' column of Fig. 3 are not reproducible from this release, because the raters' responses are withheld to protect their privacy.

## Run a judge on VisionQ-Bench (API key)

```bash
python benchmark/bench_run.py --provider openrouter --model "<openrouter/model-slug>" \
    --composite --tag-quality --concurrency 4 --out outputs/my_model.jsonl
python benchmark/bench_score.py outputs/my_model.jsonl --slice eligible
```

Each question is sent as one composite image with (A) to (D) labels and the question text only (no caption, claim or method name). `--tag-quality` marks the 309 scored questions. `benchmark/retry_failed.py` retries failed calls. Name the output `results/benchmark/predictions/test_questions_<name>_p1.jsonl` to add the judge to Tab. 2.

## Input controls (API key or GPU)

```bash
python controls/controls.py --selftest                     # offline check against saved predictions
python controls/controls.py --run --provider openrouter --model "<slug>" --out-dir outputs/controls_<name>
```

One run makes 2,574 calls over the 309 questions:
- E1: every cyclic option order.
- E2: the criterion removed, with questions rewritten by `controls/sanitize.py`.
- E3: blank and mismatched images.
- E4: full question plus the reference row.

`--provider local --model google/gemma-4-E4B-it [--adapter ...]` runs a local model on a GPU.

## VisionQ-Judge (GPU)

`judge/split_strat_v1.json` is the frozen paper-disjoint split (train 2,687 / val 359 / test 538), rebuilt exactly by `python judge/make_split.py data/mcq/pool.jsonl split.json` from the 4,524-question pool. Training is DPO with LoRA on Gemma-4-E4B, using symmetric evidence pairs in every cyclic option order (13,394 pairs, one epoch):

```bash
judge/train.sh Cfix_s42 42 all     # VisionQ-Judge; seeds 42, 43, 44 in Tab. 5
judge/train.sh Bfix_s42 42 one     # one reordered copy (ablation)
judge/train.sh B_noswap 42 none    # no reordering (ablation)
judge/train.sh base 0 base         # base model, no training
```

Trained adapters for the three reported seeds are on Hugging Face ([`visionq-review/VisionQ-Judge`](https://huggingface.co/visionq-review/VisionQ-Judge), folders `seed42`, `seed43`, `seed44`); `judge/evaluate.py --adapter <path>/seed42` scores one. Each training run also scores the val and test splits with `judge/evaluate.py`. `judge/api_eval_mcq.py` runs API judges on the MCQ test split. `judge/make_perm.py` and `judge/logprobs.py` produce the per-order letter log-probabilities used in Tab. 5 (c). fp16 results vary with GPU type, so compare runs on the same GPU type.

## Layout

```
benchmark/   judge runner, scoring, API clients
baselines/   no-reference IQA baselines
controls/    input controls E1 to E4, criterion removal, mirror test
judge/       split, DPO training, evaluation, statistics
tables/      Tab. 2 and Tab. 3 generators
figures/     make_teaser.py, which draws an earlier version of Fig. 3 on another question
results/     saved predictions behind every number above
```

## Licence

Code: MIT. Data: see the dataset card (annotations CC BY 4.0; figure crops non-commercial research use).

## Citation

```bibtex
@article{visionq2026,
  title   = {VisionQ: Benchmarking VLM Judges on Criterion-Specific Comparisons of Computer Vision Method Outputs},
  author  = {Vu Dinh Xuan and Duc-Hai Nguyen and Minh-Dung Dao and Vu Quynh Giao and Quang Hong Nguyen and Binh-Son Hua and Barry O'Sullivan and David Murphy and Hoang D. Nguyen},
  journal = {arXiv preprint arXiv:2610.00666},
  year    = {2026}
}
```
