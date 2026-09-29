#!/usr/bin/env python3
"""DPO fine-tuning of a multimodal VLM on synthesised judgment data.

This script consumes a JSONL file produced by ``synthesize_questions.py`` —
where each line is one multi-choice question grounded in a CV-paper figure —
and fine-tunes a multimodal vision-language model (e.g. Gemma-3, Qwen2.5-VL)
via Direct Preference Optimisation with LoRA adapters.

Pipeline:
  1. Load the JSONL questions file.
  2. Drop records where the answer is unknown (``answer_known == false``).
  3. Build (prompt, chosen, rejected) triples — one per wrong choice.
       * chosen   = correct letter + the natural-language explanation
       * rejected = a wrong letter (one pair per wrong option)
  4. Stratified train/test split, keyed by the leaf-name (taxonomy node) so
     every leaf appears in both splits when its support permits. Records
     belonging to the same paper stay on the same side of the split to avoid
     leakage from a paper's text appearing in both halves.
  5. DPOTrainer with LoRA. Reference model = the frozen base weights
     (the standard PEFT trick that halves memory).
  6. Save adapter + processor + the test split for downstream evaluation.

Why not parquet / why a JSONL: questions can change schema as the taxonomy
evolves; a JSONL with one record per question stays readable and diffable
under git, and the loader below tolerates new optional fields.

Usage on a single GPU (Slurm A40 / L40S, ~46 GB free):

    srun --gres=gpu:1 python train_dpo.py \\
        --questions   ~/mllm-as-judge/dpo/questions.jsonl \\
        --dataset-root ~/visionqc-v4.3 \\
        --model-id    google/gemma-3-12b-it \\
        --output-dir  ~/mllm-as-judge/dpo/output/judge_v1 \\
        --epochs      3

Resume from the last checkpoint (Trainer auto-detects):

    srun --gres=gpu:1 python train_dpo.py \\
        --questions ~/mllm-as-judge/dpo/questions.jsonl \\
        --dataset-root ~/visionqc-v4.3 \\
        --model-id google/gemma-3-12b-it \\
        --output-dir ~/mllm-as-judge/dpo/output/judge_v1 \\
        --resume

Smoke test on 5 records, no GPU work past dataset construction:

    python train_dpo.py --questions questions_sample.jsonl \\
        --dataset-root ../visionqc-v4.3 --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Methods named "method_1", "method-A", "method_b", etc. (Trap-4 abstract labels)
_TRAP4_METHOD_RE = re.compile(r"^method[_-]?[a-z0-9]+$", re.IGNORECASE)


class DecodeTransform:
    """Top-level (picklable) version of the lazy image-decode transform.

    Python 3.14 switched POSIX multiprocessing default from ``fork`` to
    ``forkserver``. ``forkserver`` re-spawns clean processes and ships
    arguments via pickle, so any function/closure defined inside another
    function fails to pickle. Defining this as a top-level class is the
    clean way to keep ``dataloader_num_workers > 0`` working.

    If ``tile_multi`` is True, all images for one question are composited
    into a single labelled image (A / B / C / D over each panel). The
    model then sees one image per prompt, sidestepping TRL's fragile
    multi-image-token / patch-feature reconciliation.
    """
    def __init__(
        self,
        dataset_root: Path,
        max_image_size: int,
        tile_multi: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.max_image_size = max_image_size
        self.tile_multi = tile_multi

    def _read_one(self, rel_path: str):
        """Read one image; return a 224x224 white placeholder on failure
        so a single bad PNG doesn't crash a batch in the middle of training.
        Such records were already filtered up-front by ``_resolve_image_path``,
        but a file might rot between scan and the actual read on slow FS."""
        from PIL import Image                                         # noqa: WPS433
        full = self.dataset_root / rel_path
        try:
            return Image.open(full).convert("RGB")
        except Exception as e:
            print(f"  ! image read failed in worker: {full} ({e}); "
                  f"using placeholder", flush=True)
            return Image.new("RGB", (224, 224), (255, 255, 255))

    def _resize_one(self, im, max_side: int):
        from PIL import Image                                         # noqa: WPS433
        if max_side <= 0:
            return im
        w, h = im.size
        if max(w, h) <= max_side:
            return im
        s = max_side / max(w, h)
        return im.resize((int(w * s), int(h * s)), Image.LANCZOS)

    def _tile(self, images, label: str = "ABCDEFGH"):
        """Pack N PIL images into a single composite, each panel sized to
        ``max_image_size`` and tagged with a corner letter (A, B, C, ...).
        Layout: 1×N for N<=2, 2×ceil(N/2) for N>=3.
        """
        from PIL import Image, ImageDraw, ImageFont                   # noqa: WPS433
        n = len(images)
        if n == 0:
            return images
        cell = self.max_image_size if self.max_image_size > 0 else 448
        if n <= 2:
            cols, rows = n, 1
        else:
            cols, rows = 2, (n + 1) // 2
        # Square-pad each panel so they all share the same cell size
        panels = []
        for im in images:
            w, h = im.size
            side = max(w, h)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(im, ((side - w) // 2, (side - h) // 2))
            panels.append(canvas.resize((cell, cell), Image.LANCZOS))
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
            r, c = i // cols, i % cols
            x = gap + c * (cell + gap)
            y = gap + r * (cell + gap)
            out.paste(panel, (x, y))
            tag = f"({label[i]})"
            # Black-on-white badge for the letter
            tw = font.getbbox(tag)[2] - font.getbbox(tag)[0]
            th = font.getbbox(tag)[3] - font.getbbox(tag)[1]
            pad = 6
            draw.rectangle([x + 4, y + 4, x + 4 + tw + 2 * pad, y + 4 + th + 2 * pad],
                           fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            draw.text((x + 4 + pad, y + 4 + pad - 2), tag, fill=(0, 0, 0), font=font)
        return [out]

    def __call__(self, batch: dict) -> dict:
        from PIL import Image                                         # noqa: WPS433
        out_imgs = []
        out_prompts = []
        for path_list, prompt_json in zip(batch["image_paths"], batch["prompt"]):
            imgs = [self._read_one(p) for p in path_list]
            prompt = json.loads(prompt_json)
            if self.tile_multi and len(imgs) > 1:
                imgs = self._tile(imgs)            # one composite, letter-tagged
                # Prompt content currently has N {"type": "image"} blocks; collapse to 1.
                user = prompt[0]
                new_content = [c for c in user["content"] if c.get("type") != "image"]
                new_content.insert(0, {"type": "image"})
                prompt = [{"role": "user", "content": new_content}]
            else:
                imgs = [self._resize_one(im, self.max_image_size) for im in imgs]
            out_imgs.append(imgs)
            out_prompts.append(prompt)
        return {
            "prompt":   out_prompts,
            "chosen":   [json.loads(c) for c in batch["chosen"]],
            "rejected": [json.loads(r) for r in batch["rejected"]],
            "images":   out_imgs,
        }

# Heavy imports (torch, transformers, trl, peft, PIL) are deferred to main()
# so --help and --dry-run work without a GPU environment.


# ---------------------------------------------------------------------------
# Data loading & DPO pair construction (no torch dependency)
# ---------------------------------------------------------------------------

def load_questions(jsonl_path: Path) -> list[dict]:
    """Read one JSON object per line, drop records without a known answer."""
    rows: list[dict] = []
    skipped_no_answer = 0
    skipped_bad = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                skipped_bad += 1
                continue
            if not r.get("answer_known", False):
                skipped_no_answer += 1
                continue
            if not r.get("answer") or not r.get("choices"):
                skipped_bad += 1
                continue
            rows.append(r)
    print(f"  loaded {len(rows)} usable questions  "
          f"(skipped: no_answer={skipped_no_answer}, malformed={skipped_bad})")
    return rows


def filter_records(
    rows: list[dict],
    *,
    drop_trap4: bool,
    max_per_leaf: int,
    seed: int,
) -> list[dict]:
    """Apply training-time data hygiene filters in this order:

    1. ``drop_trap4`` — drop records where any method name matches
       ``method[_-]?<x>`` (Trap-4 abstract labels). These come from figures
       whose original paper used neutral row tags (ablation grids); the
       resulting question reduces to "which row does the caption favour"
       without the method-name semantics that make a judge useful.

    2. ``max_per_leaf`` — cap the records per leaf at this number, dropping
       the surplus uniformly at random. Used to prevent the dominant
       Perceptual Sharpness leaf (~34% of all records) from drowning out
       the rest of the taxonomy. A value of 0 disables the cap.
    """
    rng = random.Random(seed)
    n_in = len(rows)

    if drop_trap4:
        rows = [
            r for r in rows
            if not any(_TRAP4_METHOD_RE.match(m or "") for m in r.get("methods", []))
        ]
        print(f"  drop Trap-4 abstract methods : {n_in} -> {len(rows)} "
              f"(-{n_in - len(rows)})")

    if max_per_leaf and max_per_leaf > 0:
        by_leaf: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_leaf[r.get("leaf_name", "_unknown")].append(r)
        kept: list[dict] = []
        for leaf, items in by_leaf.items():
            if len(items) > max_per_leaf:
                rng.shuffle(items)
                items = items[:max_per_leaf]
            kept.extend(items)
        n_before = sum(len(v) for v in by_leaf.values())
        print(f"  cap per leaf at {max_per_leaf:>4d}      : "
              f"{n_before} -> {len(kept)} (-{n_before - len(kept)})")
        rows = kept

    return rows


def position_swap_augment(rows: list[dict], seed: int) -> list[dict]:
    """For each record, emit one extra record with the choice order shuffled
    so the gold answer falls on a different letter. Doubles the dataset and
    cancels positional bias (A=39% / C=42% in the un-augmented data).

    The shuffle preserves the (image, method, answer) correspondence — only
    the order in which images / methods are presented changes, and the
    ``answer`` letter is updated to point at the same gold method.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for r in rows:
        out.append(r)  # keep the original

        n = len(r["choices"])
        if n < 2:
            continue
        # Find a permutation different from identity
        perm = list(range(n))
        for _ in range(8):
            rng.shuffle(perm)
            if perm != list(range(n)):
                break
        if perm == list(range(n)):
            continue  # gave up; skip swap for this record

        ans_idx = ord(r["answer"].strip("()")) - ord("A")
        if not (0 <= ans_idx < n):
            continue
        new_ans_idx = perm.index(ans_idx)

        swapped = dict(r)
        swapped["id"] = r["id"] + "__swap"
        swapped["image_paths"] = [r["image_paths"][i] for i in perm]
        swapped["methods"]     = [r["methods"][i]     for i in perm]
        swapped["answer"]      = f"({chr(ord('A') + new_ans_idx)})"
        # choices stays as ["(A)","(B)",...] — the labels themselves don't change
        out.append(swapped)
    return out


def stratified_paper_split(
    rows: list[dict],
    test_frac: float,
    seed: int,
    closest_stop: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Stratify by leaf_name, keep all questions from one paper on one side.

    Strategy: group questions by leaf, then within each leaf bucket assign
    whole papers to train or test until the test target fraction is met.
    This guarantees every leaf with >= 2 papers is represented in both
    splits, while preventing a single paper's caption phrasings from
    leaking across the split boundary.

    BUGFIX 2 (closest_stop=True): the original greedy fill committed each paper
    to test BEFORE checking the running total, so one large paper could push a
    leaf's test fraction to 60-80%, starving training. With closest_stop, a
    paper is skipped when adding it overshoots the target by more than skipping
    it under-shoots (whole papers still never cross the split).
    """
    rng = random.Random(seed)
    by_leaf: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_leaf[r.get("leaf_name", "_unknown")].append(r)

    train: list[dict] = []
    test:  list[dict] = []

    for leaf, items in by_leaf.items():
        # Group by paper within the leaf
        papers: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            papers[it["paper_id"]].append(it)
        paper_ids = list(papers.keys())
        rng.shuffle(paper_ids)

        n_leaf = len(items)
        target_test = max(1, int(round(n_leaf * test_frac))) if len(paper_ids) >= 2 else 0

        chosen_test: set[str] = set()
        running_test = 0
        for pid in paper_ids:
            if running_test >= target_test:
                break
            sz = len(papers[pid])
            if closest_stop and running_test > 0 and (running_test + sz - target_test) > (target_test - running_test):
                continue  # skipping lands closer to target than adding this big paper
            chosen_test.add(pid)
            running_test += sz

        for pid in paper_ids:
            (test if pid in chosen_test else train).extend(papers[pid])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _resolve_image_path(record_path: str, dataset_root: Path) -> Path | None:
    """Resolve a record's image path AND verify the file is a readable PNG.

    The path is stored relative to the dataset root
    (e.g. ``paper_0001/figures/fig_2_p3/crops/bbox_01_xxx.png``).

    We do a cheap header parse (Pillow only reads the IHDR chunk) so corrupted
    files are caught at startup — not 30 minutes into training when the
    dataloader hits them. Adds ~10 ms per file (negligible at our scale).
    """
    from PIL import Image, UnidentifiedImageError                     # noqa: WPS433
    p = Path(record_path)
    if p.is_absolute() and p.exists():
        target = p
    else:
        cand = (dataset_root / record_path).resolve()
        if not cand.exists():
            return None
        target = cand
    try:
        with Image.open(target) as im:
            _ = im.size                       # force header parse
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None
    return target


_ANTI_BIAS_LINE = (
    "Look carefully at every option before deciding; do not default to any "
    "particular position. Use the visual evidence."
)


def _build_prompt_text(r: dict, *, prepend_context: bool, anti_bias: bool) -> str:
    """Compose the user-message text. Order:
        [optional context anchor]  =>  question  =>  labelled choices  =>
        [optional anti-bias instruction]
    Same construction is used by the trainer and the evaluator so they
    can never drift out of sync."""
    parts: list[str] = []
    if prepend_context:
        ctx_bits: list[str] = []
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
        f"({chr(ord('A') + i)}) panel {chr(ord('A') + i)} of the image"
        for i in range(len(r["choices"]))
    )
    parts.append(f"Select from the following choices:\n{labelled}")

    if anti_bias:
        parts.append(_ANTI_BIAS_LINE)
    return "\n\n".join(parts)


def _evidence_snippet(r: dict, max_chars: int = 120) -> str:
    """Short grounded evidence for the chosen (correct) answer.

    Pulls the first sentence of the synthesised explanation; falls back to
    the must_cite_visual rubric terms if the explanation is absent.
    """
    explanation = (r.get("explanation") or "").strip()
    if explanation:
        first_sent = explanation.split(". ")[0]
        if not first_sent.endswith("."):
            first_sent += "."
        if len(first_sent) > max_chars:
            first_sent = first_sent[:max_chars - 3] + "..."
        return first_sent
    cues = r.get("must_cite_visual") or []
    if cues:
        return f"demonstrates {', '.join(cues[:2])}."
    return "is the best choice."


def _swap_evidence(evidence: str, ans_letter: str, wrong_letter: str) -> str | None:
    """Return ``evidence`` with all references to ``ans_letter`` swapped to
    ``wrong_letter`` (case-insensitive for "Panel X" / "(X)" patterns).

    Returns ``None`` if no substitution was made, signalling that the evidence
    does not contain an explicit panel-letter reference and swapped evidence
    cannot be generated — the caller should fall back to bare-letter DPO.

    This produces a positive assertion about the wrong panel ("Panel B maintains
    coherent structure") rather than a denial ("does not demonstrate...").
    Both chosen and rejected are stylistically similar, so the base model assigns
    comparable probability to each — avoiding the margin-saturation problem where
    denial-style rejected text has near-zero base probability and DPO has nothing
    left to learn.
    """
    import re
    result = evidence
    result = re.sub(
        rf"(?i)\bpanel\s+{re.escape(ans_letter)}\b",
        f"Panel {wrong_letter}",
        result,
    )
    result = re.sub(rf"\({re.escape(ans_letter)}\)", f"({wrong_letter})", result)
    return result if result != evidence else None


def build_dpo_pairs(
    rows: list[dict],
    dataset_root: Path,
    skip_missing_images: bool = True,
    chosen_letter_only: bool = False,
    prepend_context: bool = False,
    anti_bias: bool = False,
    evidence_in_response: bool = False,
    v5_evidence: bool = False,
) -> list[dict]:
    """Convert each question into one or more (prompt, chosen, rejected) pairs.

    The pair record stores **paths** (not raw image bytes) so building the
    Arrow table is fast and the table itself stays small. The training-time
    ``set_transform`` reads + decodes + resizes images lazily on the
    dataloader's worker threads. This avoids spending tens of minutes on
    networked filesystems before the model even loads.

    File existence is verified up-front (a cheap ``stat()``) so missing
    files are reported here, not as an opaque crash mid-training.

    Response modes (mutually exclusive, checked in priority order):
      chosen_letter_only   — ``(A)`` vs ``(B)``. Length-matched, pure letter
                             signal. Fixes the v1 length-asymmetry pathology.
      v5_evidence          — ``(A)\\n\\n{v5_claim_norm or v5_visual_attr}`` vs
                             ``(B)\\n\\n{identical body}``. Uses LLM-free
                             evidence from the v4.4 semantic dataset
                             (visual_attribute_text or normalized paper claim).
                             Both responses are structurally identical => near-zero
                             initial margin => no saturation. Falls back to
                             chosen_letter_only when v5 fields absent.
      evidence_in_response — ``(A)\\n\\n{evidence}`` vs ``(B)\\n\\n{evidence}``.
                             Same evidence snippet in both, different letter.
                             Length-matched; DPO gradient runs over evidence
                             tokens conditioned on the letter, richer than
                             bare-letter but avoids the v1 pathology.
      (default)            — ``(A)\\n\\n{full explanation}`` vs ``(B)``.
                             Original verbose-chosen recipe (length-asymmetric).
    """
    pairs: list[dict] = []
    n_skipped_missing = 0

    for r in rows:
        # Resolve + stat each image (no read/decode — purely metadata)
        resolved = [_resolve_image_path(p, dataset_root) for p in r["image_paths"]]
        if any(p is None for p in resolved):
            n_skipped_missing += 1
            if skip_missing_images:
                continue

        # Store paths as strings, relative to dataset_root where possible
        rel_paths: list[str] = []
        for p in resolved:
            if p is None:
                continue
            try:
                rel_paths.append(str(p.relative_to(dataset_root)))
            except ValueError:
                rel_paths.append(str(p))
        if len(rel_paths) != len(r["image_paths"]):
            continue

        # Letter <-> index
        choices: list[str] = list(r["choices"])
        ans_letter = r["answer"].strip("()")
        ans_idx = ord(ans_letter) - ord("A")
        if not (0 <= ans_idx < len(choices)):
            continue

        # Build the prompt that the judge sees
        prompt_text = _build_prompt_text(r, prepend_context=prepend_context,
                                         anti_bias=anti_bias)
        user_content: list[dict] = [{"type": "image"} for _ in rel_paths]
        user_content.append({"type": "text", "text": prompt_text})
        prompt_msgs = [{"role": "user", "content": user_content}]

        # Chosen / rejected response text.
        # Determine v5 evidence body for this record (may be None).
        _v5_body: str | None = None
        if v5_evidence:
            # Prefer richer normalized claim (winner name => "image X"); fall back
            # to visual_attribute_text (always present for enriched records).
            _v5_body = r.get("v5_claim_norm") or r.get("v5_visual_attr") or None

        if chosen_letter_only or (v5_evidence and _v5_body is None):
            # Plain letter mode — either explicitly requested or no v5 body available.
            chosen_text = f"({ans_letter})"
        elif v5_evidence and _v5_body is not None:
            # v5 symmetric evidence: both chosen and rejected carry the same body;
            # only the letter differs.  The base model has near-zero prior between
            # "(A)\n\nX" and "(B)\n\nX" when the body is identical, so initial
            # margin ≈ 0 => strong DPO gradient => no saturation.
            chosen_text = f"({ans_letter})\n\n{_v5_body}"
        elif evidence_in_response:
            # Swapped-evidence mode: chosen = correct letter + grounded evidence;
            # rejected = wrong letter + same evidence with panel letter swapped.
            # Both responses are positive assertions of similar style/length, so
            # the base model assigns comparable probability to each — no margin
            # saturation.  DPO must attend to which panel the evidence describes.
            # Fallback to bare-letter if evidence has no explicit panel reference.
            evidence = _evidence_snippet(r)
            chosen_text = f"({ans_letter})\n\n{evidence}"
        else:
            chosen_text = f"({ans_letter})"
            explanation = (r.get("explanation") or "").strip()
            if explanation:
                chosen_text += f"\n\n{explanation}"

        chosen_msgs = [{"role": "assistant", "content": chosen_text}]

        # One DPO pair per wrong option
        for j in range(len(choices)):
            if j == ans_idx:
                continue
            wrong_letter = chr(ord("A") + j)
            if v5_evidence and _v5_body is not None:
                # Swap any "image {ans_letter}" refs to "image {wrong_letter}" so
                # the body is consistent with the new answer letter.  For pure
                # visual-attr bodies with no letter refs this is a no-op (symmetric).
                _v5_body_rej = re.sub(
                    rf"\bimage\s+{re.escape(ans_letter)}\b",
                    f"image {wrong_letter}",
                    _v5_body,
                    flags=re.IGNORECASE,
                )
                rejected_content = f"({wrong_letter})\n\n{_v5_body_rej}"
            elif evidence_in_response:
                swapped = _swap_evidence(evidence, ans_letter, wrong_letter)
                if swapped is not None:
                    rejected_content = f"({wrong_letter})\n\n{swapped}"
                else:
                    # No panel-letter reference in evidence — fall back to bare
                    # letter so we don't introduce a style/probability asymmetry.
                    rejected_content = f"({wrong_letter})"
            else:
                rejected_content = f"({wrong_letter})"
            rejected_msgs = [{"role": "assistant", "content": rejected_content}]

            pairs.append({
                "prompt":      json.dumps(prompt_msgs),
                "chosen":      json.dumps(chosen_msgs),
                "rejected":    json.dumps(rejected_msgs),
                "image_paths": rel_paths,                 # list[str], read lazily
                # Metadata (stripped from training but kept for inspection)
                "_id":        r.get("id", ""),
                "_leaf":      r.get("leaf_name", ""),
                "_paper_id":  r.get("paper_id", ""),
            })

    if n_skipped_missing:
        print(f"  skipped {n_skipped_missing} records due to missing/unreadable images")
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data ---------------------------------------------------------------
    p.add_argument("--questions",    type=Path, required=True,
                   help="JSONL produced by synthesize_questions.py.")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Root that contains paper_XXXX/ folders. "
                        "Image paths in the JSONL are relative to this.")
    p.add_argument("--test-frac",    type=float, default=0.15,
                   help="Fraction of questions held out for evaluation. "
                        "Stratified by leaf, papers do not cross splits.")

    # Data hygiene -----------------------------------------------------
    p.add_argument("--filter-trap4",   action="store_true",
                   help="Drop records whose method names match "
                        "'method_<n>' / 'method_<a>' (Trap-4 abstract "
                        "labels). Recommended; loses ~17%% of records but "
                        "improves question quality.")
    p.add_argument("--max-per-leaf",   type=int, default=0,
                   help="Cap each taxonomy leaf at N records (0 = no cap). "
                        "Recommended ~300 for the current corpus to keep "
                        "Perceptual Sharpness (~34%% of records) from "
                        "dominating the gradient.")
    p.add_argument("--max-images-per-q", type=int, default=0,
                   help="Drop records where the question has more than N "
                        "images. 0 = no cap.")
    p.add_argument("--no-tile-multi", action="store_true",
                   help="Disable single-image tiling. By default, when a "
                        "question has >1 image the DecodeTransform composites "
                        "them into one labelled (A)(B)(C)(D) panel image — "
                        "the model then sees one image per prompt, which "
                        "sidesteps TRL's fragile multi-image token-feature "
                        "reconciliation. Disable only if you have a fix for "
                        "the upstream multi-image path.")
    p.add_argument("--chosen-letter-only", action="store_true",
                   help="Make the `chosen` response just the answer letter, "
                        "with NO trailing explanation. Equalises chosen/rejected "
                        "length so DPO gradient is concentrated on the letter "
                        "token itself, not on differences in the surrounding "
                        "explanation prose. Fixes the catastrophic length "
                        "asymmetry where the explanation dominates the loss "
                        "and the letter choice never actually shifts.")
    p.add_argument("--evidence-in-response", action="store_true",
                   help="chosen = '(A)\\n\\n{1-sentence grounded evidence}'; "
                        "rejected = '(B)\\n\\n{same evidence}'. "
                        "Length-matched (no v1 pathology); DPO gradient runs "
                        "over the evidence tokens conditioned on the letter, "
                        "richer signal than bare-letter pairs. Mutually "
                        "exclusive with --chosen-letter-only and --v5-evidence.")
    p.add_argument("--v5-evidence", action="store_true",
                   help="Use LLM-free symmetric evidence from the v4.4 semantic "
                        "dataset (requires questions_v5.jsonl produced by "
                        "build_dpo_dataset_v5.py). "
                        "chosen = '(A)\\n\\n{v5_claim_norm or v5_visual_attr}'; "
                        "rejected = '(B)\\n\\n{identical body}'. "
                        "Both responses carry the same body text — only the "
                        "letter differs — so the base model starts with near-zero "
                        "margin and DPO gradient is strong from step 0. "
                        "Records without v5 fields fall back to letter-only. "
                        "Mutually exclusive with --chosen-letter-only and "
                        "--evidence-in-response.")
    p.add_argument("--prepend-context", action="store_true",
                   help="Prepend the figure's row label + caption snippet to "
                        "the question. Gives the VLM a semantic anchor "
                        "(e.g. 'Stegosaurus reconstruction comparison') that "
                        "blind questions lack. Recommended after Gemini's "
                        "Slide 2 'context blindness' diagnosis.")
    p.add_argument("--anti-bias-instruction", action="store_true",
                   help="Append an explicit 'evaluate every option equally' "
                        "instruction to fight architectural position bias "
                        "(Gemma-4 has a strong anti-A preference in 4-choice).")
    p.add_argument("--position-swap",  action="store_true",
                   help="Augment by emitting each record twice with a "
                        "permuted choice order. Cancels positional answer "
                        "bias (A=39%% / C=42%%). Doubles dataset size.")

    # Model --------------------------------------------------------------
    p.add_argument("--model-id", default="google/gemma-3-12b-it",
                   help="HF model ID. Recommendations for ~46 GB GPU "
                        "(A40/L40S in bf16): "
                        "google/gemma-3-4b-it (fastest), "
                        "Qwen/Qwen2.5-VL-7B-Instruct (balanced), "
                        "google/gemma-3-12b-it (strongest, near memory ceiling).")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="QLoRA via bitsandbytes NF4. Use for the 12B model "
                        "if bf16 OOMs or to leave headroom for larger images.")
    p.add_argument("--max-image-size", type=int, default=512,
                   help="Resize images so longest side <= this (px). "
                        "Smaller -> fewer image tokens -> less VRAM. "
                        "0 = no resize. Defaults to 512 (good A40 default).")
    p.add_argument("--max-length", type=int, default=1024,
                   help="Token cap on prompt+response. 1024 is safe in bf16 "
                        "on A40; raise to 2048 with --load-in-4bit.")

    # LoRA ---------------------------------------------------------------
    p.add_argument("--lora-r",       type=int, default=16)
    p.add_argument("--lora-alpha",   type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
                   help="Suffix list — PEFT wraps any module whose name ends "
                        "with one of these. Works for plain transformer LLMs.")
    p.add_argument("--lora-target-pattern", default=None,
                   help="Regex used as PEFT target_modules instead of the list. "
                        "Required for multimodal models whose vision tower wraps "
                        "Linear in a custom class (Gemma 4 -> Gemma4ClippableLinear). "
                        "Example for Gemma 4 family: "
                        r"'.*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'")
    p.add_argument("--no-lora", action="store_true",
                   help="Full fine-tune. Almost certainly OOMs on 46 GB.")

    # Optimisation -------------------------------------------------------
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch-size",   type=int,   default=1)
    p.add_argument("--grad-accum",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=5e-6)
    p.add_argument("--beta",         type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--save-steps",   type=int,   default=0,
                   help="0 = save once per epoch (default). >0 for finer "
                        "checkpointing on long runs (useful with --resume).")
    p.add_argument("--dataloader-workers", type=int, default=2,
                   help="Number of parallel worker processes for image I/O. "
                        "Increase to 4-8 if GPU is under-utilised due to "
                        "slow networked storage; set to 0 to debug.")

    # Misc ---------------------------------------------------------------
    p.add_argument("--output-dir",   type=Path, default=Path("output/judge"))
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--resume",       action="store_true",
                   help="Resume from the latest checkpoint in --output-dir.")
    p.add_argument("--dry-run",      action="store_true",
                   help="Stop after pair construction; do not load the model.")
    p.add_argument("--fast-dry-run", action="store_true",
                   help="Stop right after the stratified split — skip pair "
                        "construction (no file existence checks). Use for "
                        "5-second sanity checks on cluster filesystems.")
    # Optional flags (defaults reproduce the paper's split and precision)
    p.add_argument("--fixed-split", action="store_true",
                   help="Use a closest-stop split that keeps each leaf's test share "
                        "nearer --test-frac. Off (default) reproduces the "
                        "paper's greedy split, whose realised test share is 20%%.")
    p.add_argument("--fp16", action="store_true",
                   help="Train in fp16 instead of bf16. Required on Turing GPUs "
                        "(RTX 8000) which have no native bf16.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)

    print(f"=== Loading questions from {args.questions} ===")
    rows = load_questions(args.questions)
    if not rows:
        print("ERROR: no usable rows.", file=sys.stderr)
        sys.exit(1)

    if args.filter_trap4 or args.max_per_leaf:
        print(f"\n=== Filtering ===")
        rows = filter_records(
            rows,
            drop_trap4=args.filter_trap4,
            max_per_leaf=args.max_per_leaf,
            seed=args.seed,
        )
        print(f"  remaining records           : {len(rows)}")

    if args.max_images_per_q and args.max_images_per_q > 0:
        n_before = len(rows)
        rows = [r for r in rows if len(r.get("image_paths", [])) <= args.max_images_per_q]
        print(f"\n=== Max images per question ===")
        print(f"  max images per question     : {args.max_images_per_q}")
        print(f"  remaining records           : {n_before} -> {len(rows)} "
              f"(-{n_before - len(rows)})")
        if not rows:
            print("ERROR: max-images-per-q removed all rows.", file=sys.stderr)
            sys.exit(1)

    print(f"\n=== Stratified split (test_frac={args.test_frac}) ===")
    train_rows, test_rows = stratified_paper_split(
        rows, args.test_frac, args.seed, closest_stop=args.fixed_split)
    print(f"  split mode: {'closest-stop (--fixed-split)' if args.fixed_split else 'greedy (as in the paper)'}")

    # Position-swap is a TRAIN-side augmentation; the test set stays clean
    # so per-letter position-bias can be measured on the original answers.
    if args.position_swap:
        print(f"\n=== Position-swap augmentation (train only) ===")
        n_before = len(train_rows)
        train_rows = position_swap_augment(train_rows, args.seed)
        print(f"  train records: {n_before} -> {len(train_rows)} "
              f"(+{len(train_rows) - n_before})")

    if args.fast_dry_run:
        print(f"\n--fast-dry-run: train={len(train_rows)} | test={len(test_rows)}  exiting.")
        return
    print(f"  train: {len(train_rows)} | test: {len(test_rows)}")
    leaf_train = Counter(r["leaf_name"] for r in train_rows)
    leaf_test  = Counter(r["leaf_name"] for r in test_rows)
    all_leaves = sorted(set(leaf_train) | set(leaf_test))
    print(f"  leaves with samples: {len(all_leaves)}  "
          f"(missing in test: {sum(1 for l in all_leaves if l not in leaf_test)})")

    print(f"\n=== Building DPO pairs ===")
    if args.chosen_letter_only and args.evidence_in_response:
        print("ERROR: --chosen-letter-only and --evidence-in-response are "
              "mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    n_response_flags = sum([
        bool(args.chosen_letter_only),
        bool(args.evidence_in_response),
        bool(args.v5_evidence),
    ])
    if n_response_flags > 1:
        print("ERROR: --chosen-letter-only, --evidence-in-response, and "
              "--v5-evidence are mutually exclusive (pick at most one).",
              file=sys.stderr)
        sys.exit(1)
    if args.chosen_letter_only:
        print("  chosen-text mode      : letter only (equal-length pairs)")
    elif args.v5_evidence:
        print("  chosen-text mode      : v5 symmetric evidence (LLM-free; "
              "letter + visual_attribute or normalized claim; same body in "
              "rejected => near-zero initial margin)")
    elif args.evidence_in_response:
        print("  chosen-text mode      : evidence-anchored (letter + 1-sentence "
              "evidence; same evidence in rejected)")
    else:
        print("  chosen-text mode      : letter + explanation (verbose chosen)")
    print(f"  prepend caption ctx   : {args.prepend_context}")
    print(f"  anti-bias instruction : {args.anti_bias_instruction}")
    pair_kwargs = dict(
        chosen_letter_only=args.chosen_letter_only,
        prepend_context=args.prepend_context,
        anti_bias=args.anti_bias_instruction,
        evidence_in_response=args.evidence_in_response,
        v5_evidence=args.v5_evidence,
    )
    train_pairs = build_dpo_pairs(train_rows, args.dataset_root, **pair_kwargs)
    test_pairs  = build_dpo_pairs(test_rows,  args.dataset_root, **pair_kwargs)
    print(f"  train pairs: {len(train_pairs)} | test pairs: {len(test_pairs)}")

    # Persist the test split for later evaluation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_dump = args.output_dir / "test_questions.jsonl"
    with test_dump.open("w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  test split saved -> {test_dump}")

    if args.dry_run:
        print("\n--dry-run: stopping before model load.")
        return

    # ------------------------------------------------------------------
    # Heavy imports
    # ------------------------------------------------------------------
    print(f"\n=== Loading model: {args.model_id} ===")
    import torch                                                      # noqa: WPS433
    from datasets import Dataset                                      # noqa: WPS433
    from transformers import (                                        # noqa: WPS433
        AutoProcessor,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )
    from trl import DPOConfig, DPOTrainer                             # noqa: WPS433
    from PIL import Image                                             # noqa: WPS433

    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB")

    # fp16 for Turing (RTX 8000), bf16 otherwise. Compute dtype follows suit.
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16
    print(f"  precision: {'fp16' if args.fp16 else 'bf16'}")

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=compute_dtype,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False  # required with gradient checkpointing

    # LoRA
    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig, TaskType                         # noqa: WPS433
        # If a regex pattern is given, pass it as a string (PEFT then does
        # full-path regex matching). Otherwise fall back to the suffix list.
        target_modules: Any = (args.lora_target_pattern
                                if args.lora_target_pattern
                                else args.lora_targets)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        kind = "regex" if args.lora_target_pattern else "suffix list"
        print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}, "
              f"dropout={args.lora_dropout}, target={kind}")

    # ------------------------------------------------------------------
    # Build HF datasets with on-the-fly decode (keep Arrow tables small)
    # ------------------------------------------------------------------
    # Picklable, top-level transform (see DecodeTransform docstring re: py3.14)
    train_ds = Dataset.from_list(train_pairs)
    train_ds.set_transform(DecodeTransform(
        args.dataset_root,
        args.max_image_size,
        tile_multi=not args.no_tile_multi,
    ))

    # ------------------------------------------------------------------
    # DPO training config
    # ------------------------------------------------------------------
    save_strategy = "epoch" if args.save_steps == 0 else "steps"
    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_ratio=args.warmup_ratio,
        bf16=not args.fp16,
        fp16=args.fp16,
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=3,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_length=args.max_length,
        dataset_num_proc=1,
        dataloader_num_workers=args.dataloader_workers,
        seed=args.seed,
    )

    print(f"\n=== DPO training ===")
    print(f"  pairs={len(train_pairs)}  epochs={args.epochs}  "
          f"effective_batch={args.batch_size * args.grad_accum}")
    steps_per_epoch = max(1, len(train_pairs) // (args.batch_size * args.grad_accum))
    print(f"  ~{steps_per_epoch} optimiser steps per epoch")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                       # frozen base = implicit reference
        args=dpo_config,
        processing_class=processor,
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    train_kwargs = {}
    if args.resume:
        # BUGFIX 1: sort checkpoints by numeric step, not lexicographically.
        # sorted() on paths put checkpoint-1000 before checkpoint-500.
        ckpts = sorted(
            args.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if ckpts:
            train_kwargs["resume_from_checkpoint"] = str(ckpts[-1])
            print(f"  resuming from {ckpts[-1].name}")
        else:
            print("  --resume requested but no checkpoint found; starting fresh.")

    trainer.train(**train_kwargs)

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\n[ok] adapter saved -> {final_dir}")
    print(f"[ok] held-out test split: {test_dump} ({len(test_rows)} questions)")


if __name__ == "__main__":
    main()
