#!/usr/bin/env python3
"""Evaluate a DPO-trained multimodal judge against the held-out test split.

Loads ``test_questions.jsonl`` (written by ``train_dpo.py`` next to the
adapter), runs the model on each question with greedy decoding, parses the
predicted letter, and writes paper-ready Markdown + CSV tables.

Two modes:
  * ``--adapter PATH`` — load the base model + LoRA adapter (= the trained
    judge). This is the post-training metric.
  * (no --adapter) — run the bare base model. Use this to obtain the
    "before" baseline so the paper can report a delta.

You can pass both by running this script twice with the same ``--out-dir``;
on the second run the per-leaf table will include base vs tuned columns.

Usage on the cluster:

    # 1. Base-model baseline (~30 min on Qwen-7B / ~60 min on Gemma-12B):
    srun --gres=gpu:1 python evaluate.py \\
        --test-questions output/judge_v1/test_questions.jsonl \\
        --dataset-root   ~/visionqc-v4.3 \\
        --model-id       google/gemma-3-12b-it \\
        --out-dir        output/judge_v1/eval \\
        --tag            base

    # 2. Tuned model (same time):
    srun --gres=gpu:1 python evaluate.py \\
        --test-questions output/judge_v1/test_questions.jsonl \\
        --dataset-root   ~/visionqc-v4.3 \\
        --model-id       google/gemma-3-12b-it \\
        --adapter        output/judge_v1/final \\
        --out-dir        output/judge_v1/eval \\
        --tag            tuned

After both runs, ``output/judge_v1/eval/`` contains:
    base_predictions.jsonl       per-question predictions, base model
    tuned_predictions.jsonl      per-question predictions, tuned model
    results.md                   the paper-ready report (overall + per-leaf
                                 + position-bias breakdown, base vs tuned)
    per_leaf.csv                 same per-leaf table as a CSV
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Predictions  ──  the GPU half
# ---------------------------------------------------------------------------

LETTER_RE = re.compile(r"\(([A-Z])\)|\b([A-Z])\b")


def parse_letter(text: str, n_choices: int) -> str | None:
    """Pull the first valid letter ``(A)..(D)`` (or bare ``A..D``) from a
    model response. Returns ``None`` if nothing parseable is found."""
    if not text:
        return None
    valid = {chr(ord("A") + i) for i in range(n_choices)}
    for m in LETTER_RE.finditer(text):
        ch = (m.group(1) or m.group(2) or "").upper()
        if ch in valid:
            return ch
    return None


def resolve_image(rec_path: str, dataset_root: Path) -> Path | None:
    p = Path(rec_path)
    if p.is_absolute() and p.exists():
        return p
    cand = (dataset_root / rec_path).resolve()
    return cand if cand.exists() else None


_ANTI_BIAS_LINE = (
    "Look carefully at every option before deciding; do not default to any "
    "particular position. Use the visual evidence."
)


def build_prompt_text(r: dict, *, prepend_context: bool, anti_bias: bool) -> str:
    """MUST stay in lockstep with train_dpo.py::_build_prompt_text. Any
    drift between training and inference prompts will tank the model."""
    parts: list[str] = []
    if prepend_context:
        ctx_bits = []
        row_raw = (r.get("row_raw") or "").strip()
        if row_raw and row_raw.lower() not in {"_default", "default"}:
            ctx_bits.append(f"Row context: {row_raw}.")
        cap = (r.get("caption") or "").strip().replace("\n", " ")
        if cap:
            if len(cap) > 400:
                cap = cap[:397] + "..."
            ctx_bits.append(f"Source paper caption: {cap}")
        if ctx_bits:
            parts.append("Context:\n" + " ".join(ctx_bits))
    parts.append(r["question"])
    labelled = "\n".join(
        f"({chr(ord('A') + j)}) panel {chr(ord('A') + j)} of the image"
        for j in range(len(r["choices"]))
    )
    parts.append(f"Select from the following choices:\n{labelled}")
    if anti_bias:
        parts.append(_ANTI_BIAS_LINE)
    return "\n\n".join(parts)


def _tile_for_eval(images, max_image_size: int, label: str = "ABCDEFGH"):
    """Same tiling logic as train_dpo.DecodeTransform._tile, kept self-
    contained so evaluate.py has zero hard dependency on train_dpo.py."""
    from PIL import Image, ImageDraw, ImageFont
    n = len(images)
    if n == 0:
        return images
    cell = max_image_size if max_image_size > 0 else 448
    cols, rows = (n, 1) if n <= 2 else (2, (n + 1) // 2)
    panels = []
    for im in images:
        w, h = im.size
        side = max(w, h)
        c = Image.new("RGB", (side, side), (255, 255, 255))
        c.paste(im, ((side - w) // 2, (side - h) // 2))
        panels.append(c.resize((cell, cell), Image.LANCZOS))
    gap = max(8, cell // 64)
    W = cols * cell + (cols + 1) * gap
    H = rows * cell + (rows + 1) * gap
    out = Image.new("RGB", (W, H), (240, 240, 240))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(28, cell // 10))
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(out)
    for i, panel in enumerate(panels):
        r, col = i // cols, i % cols
        x = gap + col * (cell + gap)
        y = gap + r * (cell + gap)
        out.paste(panel, (x, y))
        tag = f"({label[i]})"
        bb = font.getbbox(tag)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = 6
        draw.rectangle([x + 4, y + 4, x + 4 + tw + 2 * pad, y + 4 + th + 2 * pad],
                       fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((x + 4 + pad, y + 4 + pad - 2), tag, fill=(0, 0, 0), font=font)
    return [out]


def run_predictions(
    test_path: Path,
    dataset_root: Path,
    model_id: str,
    adapter_path: Path | None,
    out_jsonl: Path,
    *,
    max_image_size: int,
    max_new_tokens: int,
    load_in_4bit: bool,
    tile_multi: bool = True,
    prepend_context: bool = False,
    anti_bias: bool = False,
) -> None:
    """Stream questions from ``test_path``, run greedy generation, dump
    one prediction record per line to ``out_jsonl``.

    Heavy imports happen inside this function so other entry points
    (``aggregate_results`` below) work without a GPU env."""
    import torch                                                           # noqa: WPS433
    from transformers import (                                             # noqa: WPS433
        AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig,
    )
    from PIL import Image                                                  # noqa: WPS433

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"  loading processor from {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print(f"  loading base model from {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    if adapter_path is not None:
        from peft import PeftModel                                         # noqa: WPS433
        print(f"  loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

    # Iterate
    questions = [json.loads(l) for l in test_path.open(encoding="utf-8")]
    print(f"  evaluating on {len(questions)} test questions")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    f_out = out_jsonl.open("w", encoding="utf-8")
    t0 = time.time()
    n_parsed = n_correct = 0

    for i, q in enumerate(questions):
        # Skip records without an answer (shouldn't happen since train_dpo
        # filters them, but safe-guard anyway)
        if not q.get("answer_known"):
            continue

        # Resolve images
        img_paths = [resolve_image(p, dataset_root) for p in q["image_paths"]]
        if any(p is None for p in img_paths):
            f_out.write(json.dumps({**q, "pred_letter": None,
                                     "pred_text": "<missing image>"}) + "\n")
            continue

        images = []
        skip_record = False
        for p in img_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"    ! skipping record {q.get('id')}: bad image {p} ({e})")
                skip_record = True
                break
        if skip_record:
            f_out.write(json.dumps({**q, "pred_letter": None,
                                     "pred_text": "<bad image>"}) + "\n")
            continue

        # Tile multi-image into a single composite, or resize individually.
        # Matches DecodeTransform in train_dpo.py exactly.
        if tile_multi and len(images) > 1:
            images = _tile_for_eval(images, max_image_size)
        elif max_image_size > 0:
            resized = []
            for im in images:
                w, h = im.size
                if max(w, h) > max_image_size:
                    s = max_image_size / max(w, h)
                    im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
                resized.append(im)
            images = resized

        # Build the prompt — keep in lockstep with the trainer's prompt.
        text = build_prompt_text(q, prepend_context=prepend_context, anti_bias=anti_bias)
        # Directive for terse output so the letter appears on the first line.
        text += (
            f"\n\nAnswer with one of {', '.join(q['choices'])} on the "
            f"first line, then optionally explain."
        )

        msgs = [{
            "role": "user",
            "content": [{"type": "image"} for _ in images]
                       + [{"type": "text", "text": text}],
        }]

        prompt_str = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
        inputs = processor(
            text=prompt_str,
            images=images,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
        gen_text = processor.decode(gen_ids, skip_special_tokens=True).strip()
        pred_letter = parse_letter(gen_text, len(q["choices"]))

        gold = q["answer"].strip("()")
        is_correct = (pred_letter == gold)
        n_parsed += int(pred_letter is not None)
        n_correct += int(is_correct)

        f_out.write(json.dumps({
            **q,
            "pred_text":   gen_text[:400],
            "pred_letter": pred_letter,
            "is_correct":  is_correct,
        }, ensure_ascii=False) + "\n")
        f_out.flush()

        if (i + 1) % 25 == 0 or (i + 1) == len(questions):
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-9)
            eta = (len(questions) - i - 1) / max(rate, 1e-9)
            print(f"    [{i+1:4d}/{len(questions)}]  "
                  f"acc={n_correct/max(i+1,1):.3f}  "
                  f"parse={n_parsed/max(i+1,1):.3f}  "
                  f"{rate:.2f} q/s  ETA {eta/60:.1f}m")

    f_out.close()
    print(f"  done. acc={n_correct/max(len(questions),1):.3f} "
          f"parse={n_parsed/max(len(questions),1):.3f}")


# ---------------------------------------------------------------------------
# Aggregation  ──  the CPU half (no torch needed)
# ---------------------------------------------------------------------------

def load_preds(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def acc_breakdown(preds: list[dict], key_fn) -> dict[str, tuple[int, int]]:
    """Group predictions by ``key_fn(rec)``, return {key: (correct, total)}."""
    out: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in preds:
        key = key_fn(r)
        c, t = out[key]
        out[key] = (c + int(bool(r.get("is_correct"))), t + 1)
    return dict(out)


def fmt_acc(c: int, t: int) -> str:
    if t == 0:
        return "n/a"
    return f"{c/t:.3f} ({c}/{t})"


def fmt_pp(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta*100:.1f} pp"


def aggregate_results(
    base_preds: list[dict],
    tuned_preds: list[dict],
    out_md: Path,
    out_csv: Path,
) -> None:
    """Cross-tabulate base vs tuned predictions and write the report."""
    have_base = bool(base_preds)
    have_tuned = bool(tuned_preds)
    primary = tuned_preds or base_preds
    if not primary:
        print("nothing to aggregate", file=sys.stderr)
        return

    # ----- Overall ----------------------------------------------------
    def overall(preds: list[dict]) -> tuple[int, int, int]:
        n = len(preds)
        n_correct = sum(int(bool(r.get("is_correct"))) for r in preds)
        n_parse = sum(int(r.get("pred_letter") is not None) for r in preds)
        return n_correct, n_parse, n

    base_c, base_p, base_n = overall(base_preds) if have_base else (0, 0, 0)
    tun_c,  tun_p,  tun_n  = overall(tuned_preds) if have_tuned else (0, 0, 0)
    n_total = max(base_n, tun_n)

    # ----- Per leaf ---------------------------------------------------
    base_by_leaf  = acc_breakdown(base_preds,  lambda r: r.get("leaf_name", "_unknown"))
    tuned_by_leaf = acc_breakdown(tuned_preds, lambda r: r.get("leaf_name", "_unknown"))
    leaves = sorted(set(base_by_leaf) | set(tuned_by_leaf),
                    key=lambda l: -(tuned_by_leaf.get(l, (0, 0))[1]
                                    or base_by_leaf.get(l, (0, 0))[1]))

    # ----- Per top-tier group ----------------------------------------
    def top_of(r: dict) -> str:
        path = r.get("leaf_path") or []
        return path[1] if len(path) > 1 else "_unknown"

    base_by_top  = acc_breakdown(base_preds,  top_of)
    tuned_by_top = acc_breakdown(tuned_preds, top_of)
    tops = sorted(set(base_by_top) | set(tuned_by_top))

    # ----- Per # choices ---------------------------------------------
    base_by_n  = acc_breakdown(base_preds,  lambda r: f"{len(r.get('choices', []))}-choice")
    tuned_by_n = acc_breakdown(tuned_preds, lambda r: f"{len(r.get('choices', []))}-choice")
    ncs = sorted(set(base_by_n) | set(tuned_by_n))

    # ----- Position-bias: how often the model picks letter X --------
    def letter_dist(preds: list[dict]) -> dict[int, dict[str, float]]:
        """Per-choices-count: distribution of predicted letters."""
        out: dict[int, Counter] = defaultdict(Counter)
        for r in preds:
            n = len(r.get("choices", []))
            if r.get("pred_letter"):
                out[n][r["pred_letter"]] += 1
        return {n: dict(c) for n, c in out.items()}

    def gold_dist(preds: list[dict]) -> dict[int, dict[str, int]]:
        out: dict[int, Counter] = defaultdict(Counter)
        for r in preds:
            n = len(r.get("choices", []))
            out[n][r["answer"].strip("()")] += 1
        return {n: dict(c) for n, c in out.items()}

    base_letters  = letter_dist(base_preds)  if have_base  else {}
    tuned_letters = letter_dist(tuned_preds) if have_tuned else {}
    g_letters = gold_dist(primary)

    # ---- Write Markdown ---------------------------------------------
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# DPO Judge — Evaluation Report\n\n")
        f.write(f"Test set size: **{n_total} questions**.\n\n")

        f.write("## Overall accuracy\n\n")
        f.write("| Model | Letter accuracy | Parse rate |\n")
        f.write("|---|---:|---:|\n")
        if have_base:
            f.write(f"| Base ({base_n} eval'd) | "
                    f"{fmt_acc(base_c, base_n)} | "
                    f"{base_p}/{base_n} ({base_p/max(base_n,1):.1%}) |\n")
        if have_tuned:
            f.write(f"| **DPO-tuned ({tun_n} eval'd)** | "
                    f"**{fmt_acc(tun_c, tun_n)}** | "
                    f"{tun_p}/{tun_n} ({tun_p/max(tun_n,1):.1%}) |\n")
        if have_base and have_tuned:
            delta = (tun_c / max(tun_n, 1)) - (base_c / max(base_n, 1))
            f.write(f"\n**Delta (tuned − base): {fmt_pp(delta)}**\n")
        f.write("\n")

        # Per-top-group
        f.write("## Accuracy by top-tier group\n\n")
        cols = ["Top group"]
        if have_base: cols += ["Base", "n"]
        if have_tuned: cols += ["Tuned", "n"]
        if have_base and have_tuned: cols += ["Δ (pp)"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for top in tops:
            row = [top]
            bc, bt = base_by_top.get(top, (0, 0))
            tc, tt = tuned_by_top.get(top, (0, 0))
            if have_base:  row += [f"{bc/bt:.3f}" if bt else "n/a", str(bt)]
            if have_tuned: row += [f"{tc/tt:.3f}" if tt else "n/a", str(tt)]
            if have_base and have_tuned and bt and tt:
                row += [fmt_pp(tc/tt - bc/bt)]
            elif have_base and have_tuned:
                row += ["n/a"]
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")

        # Per-leaf
        f.write("## Accuracy by leaf (sorted by support)\n\n")
        cols = ["Leaf"]
        if have_base: cols += ["Base acc", "n"]
        if have_tuned: cols += ["Tuned acc", "n"]
        if have_base and have_tuned: cols += ["Δ (pp)"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for leaf in leaves:
            row = [leaf]
            bc, bt = base_by_leaf.get(leaf, (0, 0))
            tc, tt = tuned_by_leaf.get(leaf, (0, 0))
            if have_base:  row += [f"{bc/bt:.3f}" if bt else "n/a", str(bt)]
            if have_tuned: row += [f"{tc/tt:.3f}" if tt else "n/a", str(tt)]
            if have_base and have_tuned and bt and tt:
                row += [fmt_pp(tc/tt - bc/bt)]
            elif have_base and have_tuned:
                row += ["n/a"]
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")

        # By number of choices
        f.write("## Accuracy by question difficulty (number of choices)\n\n")
        cols = ["Choices"]
        if have_base: cols += ["Base", "n"]
        if have_tuned: cols += ["Tuned", "n"]
        if have_base and have_tuned: cols += ["Δ (pp)"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for nc in ncs:
            row = [nc]
            bc, bt = base_by_n.get(nc, (0, 0))
            tc, tt = tuned_by_n.get(nc, (0, 0))
            if have_base:  row += [f"{bc/bt:.3f}" if bt else "n/a", str(bt)]
            if have_tuned: row += [f"{tc/tt:.3f}" if tt else "n/a", str(tt)]
            if have_base and have_tuned and bt and tt:
                row += [fmt_pp(tc/tt - bc/bt)]
            elif have_base and have_tuned:
                row += ["n/a"]
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")

        # Position-bias check
        f.write("## Position-bias: predicted letter distribution\n\n")
        f.write("Compares the model's choice distribution against the gold "
                "distribution. A well-calibrated judge tracks the gold row.\n\n")
        for n in sorted(set(g_letters) | set(base_letters) | set(tuned_letters)):
            f.write(f"### {n}-choice questions\n\n")
            letters = [chr(ord("A") + i) for i in range(n)]
            f.write("| Source | " + " | ".join(letters) + " |\n")
            f.write("|" + "|".join(["---"] * (n + 1)) + "|\n")
            row_total = sum(g_letters.get(n, {}).values()) or 1
            f.write("| Gold | "
                    + " | ".join(f"{g_letters.get(n,{}).get(l,0)/row_total:.1%}"
                                  for l in letters) + " |\n")
            if have_base:
                rt = sum(base_letters.get(n, {}).values()) or 1
                f.write("| Base | "
                        + " | ".join(f"{base_letters.get(n,{}).get(l,0)/rt:.1%}"
                                      for l in letters) + " |\n")
            if have_tuned:
                rt = sum(tuned_letters.get(n, {}).values()) or 1
                f.write("| Tuned | "
                        + " | ".join(f"{tuned_letters.get(n,{}).get(l,0)/rt:.1%}"
                                      for l in letters) + " |\n")
            f.write("\n")

    # ---- Write CSV --------------------------------------------------
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("leaf,base_acc,base_n,tuned_acc,tuned_n,delta_pp\n")
        for leaf in leaves:
            bc, bt = base_by_leaf.get(leaf, (0, 0))
            tc, tt = tuned_by_leaf.get(leaf, (0, 0))
            ba = f"{bc/bt:.4f}" if bt else ""
            ta = f"{tc/tt:.4f}" if tt else ""
            d = f"{((tc/tt) - (bc/bt))*100:.2f}" if (bt and tt) else ""
            f.write(f"\"{leaf}\",{ba},{bt},{ta},{tt},{d}\n")

    print(f"\n[ok] report written -> {out_md}")
    print(f"[ok] per-leaf csv   -> {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--test-questions", type=Path, required=True,
                   help="JSONL of held-out questions (from train_dpo.py).")
    p.add_argument("--dataset-root",   type=Path, required=True,
                   help="Folder of paper_XXXX/. Image paths in JSONL are relative to this.")
    p.add_argument("--model-id",       required=True,
                   help="Same model ID used for training (the LoRA loads on top of it).")
    p.add_argument("--adapter",        type=Path, default=None,
                   help="Path to LoRA adapter folder (e.g. output/judge_v1/final). "
                        "Omit to evaluate the base model and produce the baseline.")
    p.add_argument("--out-dir",        type=Path, required=True,
                   help="Output folder. The aggregator looks for "
                        "{base,tuned}_predictions.jsonl here to build the report.")
    p.add_argument("--tag",            choices=["base", "tuned"], required=True,
                   help="Which slot to write. 'base' -> base_predictions.jsonl; "
                        "'tuned' -> tuned_predictions.jsonl.")
    p.add_argument("--max-image-size", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="VLM judges often emit chain-of-thought before the "
                        "letter, so cap generously and let the parser find "
                        "the first '(X)' anywhere in the output. Lower this "
                        "(e.g. 16) only if you also force terse-output "
                        "via the prompt.")
    p.add_argument("--load-in-4bit",   action="store_true")
    p.add_argument("--no-tile-multi",  action="store_true",
                   help="Disable multi-image tiling at inference. Must match "
                        "what the trainer used.")
    p.add_argument("--prepend-context", action="store_true",
                   help="Prepend row label + caption to the question (must "
                        "match training).")
    p.add_argument("--anti-bias-instruction", action="store_true",
                   help="Append the 'evaluate every option equally' line "
                        "(must match training).")
    p.add_argument("--skip-generate",  action="store_true",
                   help="Skip generation; just re-aggregate existing predictions "
                        "into the report. Useful after both runs are done.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = args.out_dir / f"{args.tag}_predictions.jsonl"

    if not args.skip_generate:
        if args.tag == "base" and args.adapter is not None:
            print("WARN: --tag=base but --adapter was given. "
                  "Will run with adapter and label as 'base' anyway.",
                  file=sys.stderr)
        if args.tag == "tuned" and args.adapter is None:
            print("ERROR: --tag=tuned requires --adapter.", file=sys.stderr)
            sys.exit(2)

        print(f"=== Generating predictions ({args.tag}) -> {pred_path}")
        run_predictions(
            test_path=args.test_questions,
            dataset_root=args.dataset_root,
            model_id=args.model_id,
            adapter_path=args.adapter,
            out_jsonl=pred_path,
            max_image_size=args.max_image_size,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            tile_multi=not args.no_tile_multi,
            prepend_context=args.prepend_context,
            anti_bias=args.anti_bias_instruction,
        )

    # Aggregate whatever is in out_dir (so we can call this twice and
    # progressively build the base-vs-tuned report).
    base_preds  = load_preds(args.out_dir / "base_predictions.jsonl")
    tuned_preds = load_preds(args.out_dir / "tuned_predictions.jsonl")
    aggregate_results(
        base_preds,
        tuned_preds,
        out_md=args.out_dir / "results.md",
        out_csv=args.out_dir / "per_leaf.csv",
    )


if __name__ == "__main__":
    main()
