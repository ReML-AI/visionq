#!/bin/bash
# Train one VisionQ-Judge run (DPO + LoRA on Gemma-4-E4B), then score it on the val and test splits.
# Flags are those of the runs reported in Tab. 5. Run from the repo root, with the dataset in data/.
#   judge/train.sh Cfix_s42 42 all      # every cyclic option order (VisionQ-Judge)
#   judge/train.sh Bfix_s42 42 one      # one reordered copy
#   judge/train.sh B_noswap 42 none     # no reordering
#   judge/train.sh base     0  base     # base model only (no training)
set -euo pipefail
set -f   # no globbing: the LoRA regex contains *
NAME=$1; SEED=$2; ORDERS=$3; PY=${PY:-python}
OUT=output/$NAME
EV="--dataset-root data/images --model-id google/gemma-4-E4B-it --max-image-size 512 --fp16 --prepend-context --anti-bias-instruction"
cat data/mcq/train.jsonl data/mcq/val.jsonl data/mcq/test.jsonl > data/mcq/all.jsonl   # the released split is the filtered pool
if [ "$ORDERS" != base ]; then
  case $ORDERS in all) AUG=--all-orders ;; one) AUG=--position-swap ;; none) AUG= ;; *) echo "orders: all|one|none|base"; exit 1 ;; esac
  LANG_RE='.*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
  $PY -u judge/train_dpo.py --questions data/mcq/all.jsonl --dataset-root data/images \
    --model-id google/gemma-4-E4B-it --output-dir $OUT --split-file judge/split_strat_v1.json \
    --filter-trap4 --max-per-leaf 300 $AUG --v5-evidence --prepend-context --anti-bias-instruction \
    --epochs 1 --sft-weight 0 --beta 0.3 --lr 2e-5 --lora-r 16 --max-image-size 512 --max-length 1024 --grad-accum 8 \
    --seed $SEED --fp16 --lora-target-pattern "$LANG_RE" --lora-vision
  EV="$EV --adapter $OUT/final --tag tuned"
else
  EV="$EV --tag base"
fi
for SPLIT in val test; do
  $PY -u judge/evaluate.py --test-questions data/mcq/$SPLIT.jsonl $EV --out-dir $OUT/eval_$SPLIT
done
