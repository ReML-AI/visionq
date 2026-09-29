#!/bin/bash
#SBATCH --job-name=visionq_judge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --partition=YOUR_PARTITION
#
# Train VisionQ-Judge (DPO + LoRA on google/gemma-4-E4B-it) and evaluate the base
# and tuned models on the held-out split, with the settings used in the paper.
#
#   sbatch judge/run.slurm            # on a Slurm cluster
#   bash   judge/run.slurm            # or directly on a machine with one GPU (~16 GB)
#
# Needs VISIONQ_DATA pointing at the downloaded dataset; the judge uses its "mcq/"
# folder (dpo_v5_1.jsonl + paper_XXXX/ image folders). Override any variable below
# from the environment, e.g.  FP16=1 bash judge/run.slurm  on GPUs without bf16.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

: "${VISIONQ_DATA:?set VISIONQ_DATA to the downloaded dataset root}"
DATASET_ROOT=${DATASET_ROOT:-$VISIONQ_DATA/mcq}
QUESTIONS=${QUESTIONS:-$DATASET_ROOT/dpo_v5_1.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/judge}

MODEL_ID=${MODEL_ID:-google/gemma-4-E4B-it}
EPOCHS=${EPOCHS:-1}
BETA=${BETA:-0.3}
LR=${LR:-5e-6}
LORA_R=${LORA_R:-16}
MAX_IMAGE=${MAX_IMAGE:-512}
MAX_LEN=${MAX_LEN:-1024}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_PER_LEAF=${MAX_PER_LEAF:-300}
SAVE_STEPS=${SAVE_STEPS:-300}
SEED=${SEED:-42}
LORA_PATTERN='.*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
PRECISION=(); [ "${FP16:-0}" = "1" ] && PRECISION=(--fp16)

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u judge/train_dpo.py \
    --questions      "$QUESTIONS" \
    --dataset-root   "$DATASET_ROOT" \
    --model-id       "$MODEL_ID" \
    --output-dir     "$OUTPUT_DIR" \
    --epochs         "$EPOCHS" \
    --beta           "$BETA" \
    --lr             "$LR" \
    --lora-r         "$LORA_R" \
    --seed           "$SEED" \
    --test-frac      0.15 \
    --max-image-size "$MAX_IMAGE" \
    --max-length     "$MAX_LEN" \
    --grad-accum     "$GRAD_ACCUM" \
    --save-steps     "$SAVE_STEPS" \
    --filter-trap4 \
    --max-per-leaf   "$MAX_PER_LEAF" \
    --position-swap \
    --v5-evidence \
    --prepend-context \
    --anti-bias-instruction \
    --lora-target-pattern "$LORA_PATTERN" \
    "${PRECISION[@]}"

# Evaluate base and tuned models on this run's held-out test split.
for TAG in base tuned; do
    ADAPTER=(); [ "$TAG" = "tuned" ] && ADAPTER=(--adapter "${ADAPTER_DIR:-$OUTPUT_DIR/final}")
    python -u judge/evaluate.py \
        --test-questions "$OUTPUT_DIR/test_questions.jsonl" \
        --dataset-root   "$DATASET_ROOT" \
        --model-id       "$MODEL_ID" \
        --out-dir        "$OUTPUT_DIR/eval" --tag "$TAG" \
        --max-image-size "$MAX_IMAGE" \
        --prepend-context --anti-bias-instruction \
        "${ADAPTER[@]}"
done
echo "Done: $OUTPUT_DIR/eval/results.md"
