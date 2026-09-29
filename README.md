# VisionQ

VisionQ is a benchmark for **VLM judges on qualitative comparison figures from computer-vision papers**. Each question shows the cropped outputs of 2–4 methods from a peer-reviewed CVPR/ICCV figure and names one visual criterion from a six-axis taxonomy (e.g. *"Which method shows sharper boundaries?"*). A judge is correct when it picks the output the paper's authors identify as best on that criterion: *criterion-conditioned visual discrimination*.

This repository contains the benchmark harness, the code to train **VisionQ-Judge** (a DPO-tuned Gemma-4-E4B judge), the raw predictions behind every number in the paper, and the scripts that reproduce the paper's tables and figures.

| | |
|---|---|
| Paper | arXiv link *(to be added)* |
| Code | https://github.com/ReML-AI/visionq |
| Dataset | https://huggingface.co/datasets/visionq-anon-2026/VisionQ-1k |

## What's here

```
benchmark/        run any VLM on VisionQ-Bench and score it (OpenRouter or AWS Bedrock)
judge/            train and evaluate VisionQ-Judge (DPO + LoRA)
analysis/         reproduce the paper's numbers, tables, and figures from saved predictions
data/
  paper_list.csv            the 1,409 source papers (title, venue, year, task type)
  taxonomy_codebook.json    the six-axis, 51-leaf taxonomy
  benchmark/                VisionQ-Bench questions (332; 309 eligible) and annotated data points
  judge/test_questions.jsonl  the 717-question VisionQ-Judge test split
  eda/                      per-task coverage tables used by the appendix figures
results/
  benchmark/predictions/    raw per-question outputs of all 21 rows on VisionQ-Bench
  judge/                    base, checkpoint-900 (reported), and final predictions on the 717-question split
```

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

The images are in the dataset on Hugging Face. Download it and point `VISIONQ_DATA` at its root:

```bash
export VISIONQ_DATA=/path/to/visionq-dataset   # contains bench/ and mcq/
```

- `bench/` holds the image crops for VisionQ-Bench.
- `mcq/` holds VisionQ-MCQ: `dpo_v5_1.jsonl` (4,524 questions from 513 papers) and one folder per paper with its figures, crops, and annotations.

## Reproduce the paper's numbers (no GPU, no API keys)

Everything below reads the saved predictions in `results/`.

```bash
python analysis/benchmark_stats.py     # §4 and Table 5: accuracy per model and axis, 95% intervals, year check
python analysis/dpo_stats.py           # §5: VisionQ-Judge accuracy, intervals, positional bias, per-leaf changes
python analysis/build_paper_figures.py --no-mcnemar   # ranking and per-axis figures -> results/figures/
python analysis/build_profile_dashboard.py            # interactive dashboard -> results/dashboard_profiles.html
python analysis/figures/taxonomy.py                   # taxonomy, DPO pipeline, and coverage figures
python analysis/figures/dpo_dataset.py
python analysis/figures/coverage_figs.py
```

Confidence intervals come from a bootstrap that resamples source papers, because questions from the same paper are not independent. On VisionQ-Bench, an unparseable answer or a failed API request counts as wrong.

## Evaluate a new judge on VisionQ-Bench

Copy `.env.example` to `.env` and add an OpenRouter key (or AWS Bedrock credentials). Then:

```bash
cd benchmark
python bench_run.py --provider openrouter --model "<openrouter/model-slug>" \
    --composite --tag-quality --concurrency 4 --out ../outputs/my_model.jsonl
python bench_score.py ../outputs/my_model.jsonl --slice eligible
```

Each question is sent as one composite image with hard-rendered (A)–(D) labels and the question text only; no caption, claim, or method name. `--tag-quality` marks the 309 eligible questions (the four validity checks remove 23 of the 332). Failed requests can be retried with `retry_failed.py`.

## Train VisionQ-Judge

`judge/run.slurm` trains and evaluates with the paper's settings (Gemma-4-E4B-it, LoRA r=16, β=0.3, learning rate 5e-6, one epoch, 300 questions per leaf, seed 42, paper-disjoint split). It runs under Slurm or plain bash on one GPU with about 16 GB of memory:

```bash
bash judge/run.slurm              # add FP16=1 on GPUs without bf16 support
```

The split it produces is identical to `data/judge/test_questions.jsonl` (717 test questions, 9,525 training pairs). Checkpoints are saved every 300 steps; the paper reports checkpoint 900. `--fixed-split` switches to a split that keeps each leaf's test share closer to the 15% target; it is off by default so the paper's split is reproduced.

## License

- **Code:** MIT (see `LICENSE`).
- **Annotations, taxonomy, and questions:** CC BY 4.0.
- **Figure crops:** copyright remains with the authors and publishers of the source papers (CVPR and ICCV proceedings, available through CVF Open Access). They are redistributed for non-commercial research use only, each linked to its source paper in `data/paper_list.csv`.

**Intended use.** VisionQ is for diagnostic evaluation of VLM judges. Automated reviewing, scoring, or generation of scientific papers is outside its intended use.

**Opt-out.** If you are an author and do not want your figures included, open an issue; we remove them in the next release.

## Citation

```bibtex
@article{visionq2026,
  title  = {VisionQ: VLM-as-a-Judge Taxonomy, Dataset and Benchmark for Qualitative Analysis in Computer Vision},
  author = {(to be added)},
  year   = {2026}
}
```
