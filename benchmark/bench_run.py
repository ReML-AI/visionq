#!/usr/bin/env python3
"""Run a VLM-as-judge benchmark over dpo_scratch.jsonl.

For each MCQ record, sends the question + N candidate crops to the chosen
provider and saves the model's letter answer (A/B/C/D) alongside the gold
answer for downstream scoring by bench_score.py.

Pre-eval filter: drop records whose `methods` list has duplicate display
names (1.4% of records — the model can't disambiguate from method name
alone, though crops differ).

Usage:
    # Smoke test — 20 records, prints per-record outcome
    python bench_run.py --provider bedrock --model haiku --limit 20

    # Full run
    python bench_run.py --provider bedrock --model haiku \
        --out outputs/run_haiku_full.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.bedrock_vlm import (
    BedrockError,
    BedrockVLMClient,
    DEFAULT_HAIKU_MODEL_ID,
    DEFAULT_SONNET_MODEL_ID,
    DEFAULT_OPUS_MODEL_ID,
    load_local_env,
)
from lib.openrouter_vlm import OpenRouterError, OpenRouterVLMClient

# A handful of crops in the v4.4 corpus are truncated (incomplete bottom rows).
# PIL refuses by default; this flag lets it load whatever bytes are present
# rather than failing the whole batch.
try:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    pass

LOG = logging.getLogger("bench_run")

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parent
# Image root: the "bench/" folder of the downloaded VisionQ dataset (set VISIONQ_DATA, or pass --dataset-root).
DATASET_ROOT_DEFAULT = Path(os.environ.get("VISIONQ_DATA", REPO / "data" / "images")) / "bench"
DATA_POINTS_DEFAULT  = REPO / "data" / "benchmark" / "data_points.jsonl"
QUESTIONS_DEFAULT = REPO / "data" / "benchmark" / "test_questions.jsonl"

# SKIP_BASELINE pattern used by synthesize_dpo_scratch.py — anything matching
# this is treated as a reference/context crop, not a candidate. Used by the
# context-row pull in --include-context.
SKIP_BASELINE = re.compile(
    r"(?:\b(?:ground[_\s]?truth|g\.t\.|gt)\b|input[_\s]?(?:image|point|cloud|view)?"
    r"|\breference[_\s]?(?:image|view)?|\btarget\b|\bscan\b|\bcapture[ds]?\b|\breal(?:\s+image)?\b)",
    re.IGNORECASE,
)
MAX_CONTEXT_TILES = 3   # cap reference-row size to keep payload bounded

# EXTENDED reference detection — used ONLY by quality-flag computation
# (NOT by --include-context, which sticks to the conservative SKIP_BASELINE
# to avoid pulling in panels that look reference-y but might be candidates).
# Extended adds projection/visualization/discovered/annotation/mask/dataset-image
# + bare panel-letter labels like "(a)", "(b)" with no method name attached.
EXT_REF_PATTERN = re.compile(
    r"(?:\b(?:ground[_\s]?truth|g\.t\.|gt)\b|input[_\s]?(?:image|point|cloud|view)?"
    r"|\breference[_\s]?(?:image|view)?|\btarget\b|\bscan\b|\bcapture[ds]?\b"
    r"|\breal(?:\s+image)?\b|\bproject(?:ed|ion)\b|\bvisuali[sz]ation\b"
    r"|\bdiscovered\b|\bannotation(?:s)?\b|\bmask\b|\bdataset\s+image\b)",
    re.IGNORECASE,
)
PANEL_LETTER_ONLY = re.compile(r"^\(?[a-z]\)?\.?$", re.IGNORECASE)


def is_extended_reference(method_raw: str) -> bool:
    """Catch context/reference panels that escape the conservative SKIP_BASELINE,
    e.g. 'Projected 3D keypoints', 'Visualization of …', or bare panel letters.
    """
    s = (method_raw or "").strip()
    if not s:
        return False
    if PANEL_LETTER_ONLY.match(s):
        return True
    return bool(EXT_REF_PATTERN.search(s))


def _y_band_of(bboxes: list[dict], gap_factor: float = 1.0) -> dict[int, int]:
    """Cluster bboxes into vertical sub-row bands by y_center. Two bboxes share
    a band if their y_center differs by <= gap_factor × the median bbox height.

    Returns {bbox_id: band_idx}. Used by cross_subrow_mixing detection — bboxes
    within the same `row` annotation field can still belong to different visual
    sub-rows (different example instances stacked vertically). The y-band split
    catches that.

    Threshold default 0.5×median-height: a 255-px tall bbox needs >127px gap to
    its neighbour to be considered a different sub-row. On the cited
    `paper_0023/faces2comics` case the inter-band gap is ~255px (≈100% of
    height) — clean split.
    """
    if not bboxes:
        return {}
    items = []
    for b in bboxes:
        bb = b["bbox"]
        items.append((b["bbox_id"], (bb[1] + bb[3]) / 2, bb[3] - bb[1]))
    items.sort(key=lambda t: t[1])
    median_h = sorted(t[2] for t in items)[len(items) // 2]
    gap = max(20.0, gap_factor * median_h)
    bands: dict[int, int] = {items[0][0]: 0}
    band_idx = 0
    last_yc = items[0][1]
    for bid, yc, _ in items[1:]:
        if yc - last_yc > gap:
            band_idx += 1
        bands[bid] = band_idx
        last_yc = yc
    return bands


def method_family_stem(method_raw: str) -> str:
    """Normalize a method label down to its 'family' for collapse detection.
    Strips parenthetical/bracket modifiers, parameter sweeps, and self-refs
    so 'Ours (15 kpts)' and 'Ours (30 kpts)' both reduce to 'OURS_TOKEN'.
    """
    s = (method_raw or "").lower()
    s = re.sub(r"^\(?\s*\(?[a-z]\)?\s*[\.\)\]]?\s+", "", s)        # leading "(a) "
    s = re.sub(r"\([^)]*\)", " ", s)                                # parens
    s = re.sub(r"\[[^\]]*\]", " ", s)                                # brackets/citations
    s = re.sub(r"\b\d+\s*(?:kpts?|points?|views?|samples?|epochs?|iter|steps?)\b",
               " ", s)
    s = re.sub(
        r"\b(ours?|our\s+(?:method|approach|model|framework|system|network|work)|"
        r"the\s+proposed\s+(?:method|approach|model|framework|system|network|baseline)|"
        r"proposed)\b",
        "OURS_TOKEN", s,
    )
    return re.sub(r"[\s,;:.\-_/]+", " ", s).strip()

# Bedrock image cap is generous, but downsample anything above this to stay
# safely under per-request payload limits and reduce token cost.
MAX_IMAGE_BYTES = 1_500_000  # 1.5 MB
MAX_IMAGE_DIM = 1024         # downscale longest side to this when too large

LETTER_RE = re.compile(r"\b\(?\s*([A-D])\s*\)?\b")

SYSTEM_PROMPT = (
    "You are a careful image-quality judge. You will see a question and "
    "between 2 and 4 candidate images, each labeled (A), (B), (C), (D). "
    "Pick the SINGLE image that best satisfies the question. "
    "Reply with EXACTLY one of: (A), (B), (C), (D). "
    "Do not explain. Do not add any other text."
)

# Composite-grid layout: tile crops into one image with hard-rendered (A)/(B)/...
# banners above each tile. Sent as ONE image block to remove text-to-image-block
# binding ambiguity (caught in 20-record smoke: 5/5 on gold=B, 1/15 on non-B).
COMPOSITE_TILE_PX = 512        # each tile is resized to fit this on its longest side
COMPOSITE_BANNER_PX = 64       # banner height for letter label
COMPOSITE_PAD_PX = 8           # gap between tiles


def build_composite_grid(image_paths: list[Path],
                         context_tiles: list[ContextTile] | None = None) -> bytes:
    """Stack crops into a single grid image with letter banners.

    If ``context_tiles`` is non-empty, they are rendered in a single row ABOVE
    the candidate grid with gray banners labeled GT/INPUT/REF/etc. Candidate
    tiles keep their black (A)/(B)/... banners.
    """
    from PIL import Image, ImageDraw, ImageFont

    context_tiles = context_tiles or []

    n = len(image_paths)
    if n == 2:   cand_cols, cand_rows = 2, 1
    elif n == 3: cand_cols, cand_rows = 3, 1
    elif n == 4: cand_cols, cand_rows = 2, 2
    else:        cand_cols, cand_rows = n, 1

    def _load_tile(p: Path) -> "Image.Image":
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = COMPOSITE_TILE_PX / max(w, h)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return im

    cand_imgs = [_load_tile(p) for p in image_paths]
    ctx_imgs  = [_load_tile(t.path) for t in context_tiles]

    all_imgs = cand_imgs + ctx_imgs
    tile_w = max(im.size[0] for im in all_imgs)
    tile_h = max(im.size[1] for im in all_imgs)
    cell_w = tile_w
    cell_h = tile_h + COMPOSITE_BANNER_PX

    cand_canvas_w = cand_cols * cell_w + (cand_cols + 1) * COMPOSITE_PAD_PX
    ctx_row_w     = (len(ctx_imgs) * cell_w + (len(ctx_imgs) + 1) * COMPOSITE_PAD_PX
                     if ctx_imgs else 0)
    canvas_w = max(cand_canvas_w, ctx_row_w)
    ctx_canvas_h = (cell_h + COMPOSITE_PAD_PX) if ctx_imgs else 0
    canvas_h = ctx_canvas_h + cand_rows * cell_h + (cand_rows + 1) * COMPOSITE_PAD_PX
    canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")
    draw = ImageDraw.Draw(canvas)

    # Fonts (cache once)
    def _font(size: int):
        for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
            if Path(fp).exists():
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    cand_font = _font(44)
    ctx_font  = _font(32)

    def _paint_tile(x0: int, y0: int, im, banner_text: str, banner_fill: str, text_fill: str, font):
        draw.rectangle([x0, y0, x0 + cell_w, y0 + COMPOSITE_BANNER_PX], fill=banner_fill)
        bbox = draw.textbbox((0, 0), banner_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x0 + (cell_w - tw) // 2, y0 + (COMPOSITE_BANNER_PX - th) // 2 - bbox[1]),
            banner_text, font=font, fill=text_fill,
        )
        ix = x0 + (cell_w - im.size[0]) // 2
        iy = y0 + COMPOSITE_BANNER_PX + (tile_h - im.size[1]) // 2
        canvas.paste(im, (ix, iy))

    # Context row (gray banners)
    if ctx_imgs:
        for c, (im, t) in enumerate(zip(ctx_imgs, context_tiles)):
            x0 = COMPOSITE_PAD_PX + c * (cell_w + COMPOSITE_PAD_PX)
            y0 = COMPOSITE_PAD_PX
            _paint_tile(x0, y0, im, t.label, banner_fill="#666666",
                        text_fill="white", font=ctx_font)

    # Candidate grid (black banners with letters)
    cand_y_offset = ctx_canvas_h
    for idx, im in enumerate(cand_imgs):
        rr, cc = divmod(idx, cand_cols)
        x0 = COMPOSITE_PAD_PX + cc * (cell_w + COMPOSITE_PAD_PX)
        y0 = cand_y_offset + COMPOSITE_PAD_PX + rr * (cell_h + COMPOSITE_PAD_PX)
        _paint_tile(x0, y0, im, f"({chr(ord('A') + idx)})",
                    banner_fill="black", text_fill="white", font=cand_font)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Image loading + optional downsample
# --------------------------------------------------------------------------- #

def load_image_bytes(path: Path) -> tuple[bytes, str]:
    """Return (bytes, format) — downsample if file is too large."""
    raw = path.read_bytes()
    fmt = path.suffix.lstrip(".").lower() or "png"
    if fmt == "jpeg":
        fmt = "jpeg"
    if len(raw) <= MAX_IMAGE_BYTES:
        return raw, fmt

    # Downsample with PIL
    try:
        from PIL import Image
    except ImportError:
        LOG.warning("PIL not installed; sending oversized image %s as-is", path.name)
        return raw, fmt

    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    longest = max(w, h)
    if longest > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / longest
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "png"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class ContextTile:
    path: Path
    label: str               # "GT", "INPUT", "REFERENCE", ...


@dataclass
class Record:
    id: str
    question: str
    choices: list[str]            # ["(A)", "(B)", ...]
    answer_letter: str            # "A".."D"
    image_paths: list[Path]
    methods: list[str]
    axis: str
    leaf: str
    n_choices: int
    data_point_key: str = ""
    # Populated when --include-context / --tag-quality are set
    context_tiles: list[ContextTile] = None  # type: ignore[assignment]
    quality_flags: dict | None = None


# --------------------------------------------------------------------------- #
# Same-group context lookup (loads data_points.jsonl on first need, lazy)
# --------------------------------------------------------------------------- #

_DP_LOOKUP: dict[str, dict] | None = None


def _load_data_points(path: Path) -> dict[str, dict]:
    global _DP_LOOKUP
    if _DP_LOOKUP is not None:
        return _DP_LOOKUP
    print(f"Loading data_points from {path} ...", file=sys.stderr)
    out: dict[str, dict] = {}
    for line in path.open():
        if not line.strip(): continue
        d = json.loads(line)
        out[d["data_point_key"]] = d
    _DP_LOOKUP = out
    print(f"  Loaded {len(out)} data_points", file=sys.stderr)
    return out


def _classify_context_label(method_raw: str) -> str:
    """Pick a short banner label for a reference/context crop."""
    s = method_raw.lower()
    if "ground" in s and "truth" in s: return "GT"
    if re.search(r"\bgt\b|g\.t\.", s):  return "GT"
    if "input" in s:                    return "INPUT"
    if "reference" in s or "ref" in s:  return "REF"
    if "target" in s:                   return "TARGET"
    if "real" in s:                     return "REAL"
    if "scan" in s or "capture" in s:   return "SCAN"
    return "REF"


def _get_context_tiles(dp: dict, dataset_root: Path,
                       max_tiles: int = MAX_CONTEXT_TILES) -> list[ContextTile]:
    """Return up to max_tiles same-group reference bboxes (bbox_type=main only).
    Deduplicates by label so we don't get e.g. three separate GT tiles."""
    tiles: list[ContextTile] = []
    seen_labels: set[str] = set()
    for b in dp["evidence"]["bboxes"]:
        if not SKIP_BASELINE.search(b.get("method_raw", "") or ""):
            continue
        if b.get("bbox_type") != "main":
            continue
        label = _classify_context_label(b["method_raw"])
        if label in seen_labels:
            continue
        path = dataset_root / b["crop_path"]
        if not path.exists():
            continue
        tiles.append(ContextTile(path=path, label=label))
        seen_labels.add(label)
        if len(tiles) >= max_tiles:
            break
    return tiles


def _compute_record_quality_flags(rec_chosen_paths: list[str], dp: dict,
                                  methods: list[str],
                                  v5_claim_norm: str | None) -> dict:
    """Compute the eligibility verdict + orthogonal diagnostic flags per
    reviewer's schema:

    Returns
    -------
    dict with keys:
        benchmark_eligible          : bool — false only on FATAL structural failures
        ineligible_reasons          : list[str] — empty when eligible
        has_reference_context       : bool — same-group GT/INPUT/REF available
        repeated_main_crop          : bool — chosen method has >1 main bbox
        repeated_main_rgb_crop      : bool — chosen method has >1 main+rgb bbox
        distinct_row_viz_ambiguity  : bool — chosen method spans distinct rows/viz
        weak_evidence_grounding     : bool — no v5_claim_norm available

    Fatal failures (drive ineligible_reasons):
        choice_is_context_panel             — any choice is GT/Input/Projected/...
        duplicate_method_label              — two choices share a display name
        single_method_family_after_ref_strip — after stripping context choices,
                                              fewer than 2 distinct method families remain
    """
    # ---- Fatal eligibility checks ----
    inelig: list[str] = []

    if any(is_extended_reference(m) for m in methods):
        inelig.append("choice_is_context_panel")

    if len(set(methods)) < len(methods):
        inelig.append("duplicate_method_label")

    cleaned = [m for m in methods if not is_extended_reference(m)]
    cf = {method_family_stem(m) for m in cleaned}
    cf.discard("")
    if not cleaned or len(cf) < 2:
        inelig.append("single_method_family_after_ref_strip")

    # ---- Diagnostic flags ----
    bbox_by_path = {b["crop_path"]: b for b in dp["evidence"]["bboxes"]}
    chosen_slugs: set[str] = set()
    for p in rec_chosen_paths:
        if p in bbox_by_path:
            chosen_slugs.add(bbox_by_path[p]["method"])

    by_slug: dict[str, list[dict]] = {}
    for b in dp["evidence"]["bboxes"]:
        by_slug.setdefault(b["method"], []).append(b)

    has_ref = any(
        b.get("bbox_type") == "main" and SKIP_BASELINE.search(b.get("method_raw", "") or "")
        for b in dp["evidence"]["bboxes"]
    )
    repeated_main = any(
        sum(1 for x in by_slug.get(s, []) if x["bbox_type"] == "main") > 1
        for s in chosen_slugs
    )
    repeated_main_rgb = any(
        sum(1 for x in by_slug.get(s, [])
            if x["bbox_type"] == "main" and x["viz_type"] == "rgb") > 1
        for s in chosen_slugs
    )
    distinct_row_viz = any(
        len({(x.get("row"), x.get("viz_type")) for x in by_slug.get(s, [])
             if x["bbox_type"] == "main"}) > 1
        for s in chosen_slugs
    )

    # cross_subrow_mixing: do the chosen bboxes span >1 vertical y-band?
    # Catches cases where the synth picked one method from the top sub-row of
    # a multi-instance group and another method from the bottom sub-row, even
    # though both share the same `row` annotation. See `paper_0023/faces2comics`.
    bands = _y_band_of(dp["evidence"]["bboxes"])
    chosen_bbox_ids = [bbox_by_path[p]["bbox_id"] for p in rec_chosen_paths
                       if p in bbox_by_path]
    chosen_bands = {bands.get(bid) for bid in chosen_bbox_ids}
    chosen_bands.discard(None)
    cross_subrow = len(chosen_bands) > 1

    return {
        "benchmark_eligible":         len(inelig) == 0,
        "ineligible_reasons":         inelig,
        "has_reference_context":      bool(has_ref),
        "repeated_main_crop":         bool(repeated_main),
        "repeated_main_rgb_crop":     bool(repeated_main_rgb),
        "distinct_row_viz_ambiguity": bool(distinct_row_viz),
        "cross_subrow_mixing":        bool(cross_subrow),
        "weak_evidence_grounding":    not bool(v5_claim_norm),
    }


def load_records(path: Path, dataset_root: Path, drop_dup_methods: bool,
                 rotate: int = 0,
                 *,
                 include_context: bool = False,
                 tag_quality: bool = False,
                 data_points_path: Path | None = None) -> list[Record]:
    """Load records. If ``rotate`` > 0, cyclically right-shift the image-slot
    assignment by that many positions in EACH record (so what was at slot A
    moves to slot (A + rotate)). The gold letter is shifted the same way.
    Used for rotation-diagnostic: a content-based model should track the
    rotated gold; a position-biased model will keep favoring the same slots.
    """
    out: list[Record] = []
    n_drop_dup = 0
    n_drop_missing = 0
    dp_lookup = (_load_data_points(data_points_path or DATA_POINTS_DEFAULT)
                 if (include_context or tag_quality) else {})

    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if drop_dup_methods and len(set(r["methods"])) < len(r["methods"]):
            n_drop_dup += 1
            continue
        img_paths = [dataset_root / p for p in r["image_paths"]]
        if not all(p.exists() for p in img_paths):
            n_drop_missing += 1
            continue
        n = len(img_paths)
        gold_letter = r["answer"].strip("()")
        if rotate:
            k = rotate % n
            img_paths = [img_paths[(i - k) % n] for i in range(n)]
            gold_idx = ord(gold_letter) - ord("A")
            gold_letter = chr(ord("A") + (gold_idx + k) % n)

        # Context tiles + quality flags
        ctx: list[ContextTile] = []
        qflags: dict | None = None
        dpk = r.get("v5_data_point_key", "")
        if (include_context or tag_quality) and dpk in dp_lookup:
            dp = dp_lookup[dpk]
            if include_context:
                ctx = _get_context_tiles(dp, dataset_root)
            if tag_quality:
                qflags = _compute_record_quality_flags(
                    r["image_paths"], dp, r["methods"], r.get("v5_claim_norm"),
                )

        out.append(Record(
            id=r["id"],
            question=r["question"],
            choices=r["choices"],
            answer_letter=gold_letter,
            image_paths=img_paths,
            methods=r["methods"],
            axis=r.get("v5_taxonomy_axis", "") or "",
            leaf=r.get("v5_taxonomy_leaf", "") or "",
            n_choices=n,
            data_point_key=dpk,
            context_tiles=ctx,
            quality_flags=qflags,
        ))
    print(f"Loaded {len(out)} records "
          f"(dropped dup-methods={n_drop_dup}, missing-img={n_drop_missing}, "
          f"rotate={rotate}, include_context={include_context}, tag_quality={tag_quality})",
          file=sys.stderr)
    if tag_quality:
        n_elig = sum(1 for r in out if r.quality_flags and r.quality_flags["benchmark_eligible"])
        print(f"  Eligible: {n_elig}/{len(out)} "
              f"({100*n_elig/max(1,len(out)):.1f}%)", file=sys.stderr)
    if include_context:
        n_with_ctx = sum(1 for r in out if r.context_tiles)
        print(f"  Records with context tiles attached: {n_with_ctx} "
              f"({100*n_with_ctx/max(1,len(out)):.1f}%)", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# Prompt + parsing
# --------------------------------------------------------------------------- #

def build_user_text(rec: Record, composite: bool = False) -> str:
    n = rec.n_choices
    letters = ", ".join(f"({chr(ord('A') + i)})" for i in range(n))
    has_ctx = bool(rec.context_tiles)
    if composite and has_ctx:
        ctx_labels = ", ".join(t.label for t in rec.context_tiles)
        return (
            f"Question: {rec.question}\n"
            f"The single image below has TWO sections:\n"
            f"  (1) Top row — reference imagery for CONTEXT ONLY, labeled with gray "
            f"banners ({ctx_labels}). Do not pick these.\n"
            f"  (2) Below — {n} candidate tiles labeled with black banners {letters}. "
            f"Pick exactly ONE of these candidates that best satisfies the question.\n"
            f"Reply with EXACTLY one of those letters in parentheses."
        )
    if composite:
        return (
            f"Question: {rec.question}\n"
            f"The single image below contains {n} candidate tiles arranged in a grid. "
            f"Each tile has a label banner above it: {letters}. "
            f"Reply with EXACTLY one of those letters in parentheses."
        )
    return (
        f"Question: {rec.question}\n"
        f"You will see {n} candidate images labeled {letters}. "
        f"Reply with EXACTLY one of those letters in parentheses."
    )


def parse_letter(text: str, n_choices: int) -> str | None:
    """Extract a valid letter A..D within range. For reasoning models that
    write a chain-of-thought, the final answer is at the end — so we prefer
    the LAST `(X)` occurrence over the first. Falls back to last bare letter
    A..D if no parenthesized form is present.
    """
    if not text:
        return None
    last: str | None = None
    # Prefer parenthesized "(X)" form (high precision)
    for m in re.finditer(r"\(\s*([A-D])\s*\)", text):
        letter = m.group(1).upper()
        if 0 <= ord(letter) - ord("A") < n_choices:
            last = letter
    if last is not None:
        return last
    # Fall back to bare letter A..D (last occurrence)
    for m in re.finditer(r"\b([A-D])\b", text):
        letter = m.group(1).upper()
        if 0 <= ord(letter) - ord("A") < n_choices:
            last = letter
    return last


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

async def run_one(client, rec: Record, sem: asyncio.Semaphore,
                  max_tokens: int | None, composite: bool,
                  retries: int = 2) -> dict[str, Any]:
    """Single-record runner with per-record retry on error/unparseable.

    Returns one row including the full raw response (text + reasoning +
    raw_message + raw_response_meta) so downstream analysis can inspect
    everything the model emitted, including chain-of-thought tokens.
    """
    if composite:
        grid = await asyncio.to_thread(
            build_composite_grid, rec.image_paths, rec.context_tiles or [],
        )
        images_bytes = [grid]
        image_format = "png"
    else:
        images_bytes = []
        image_format = "png"
        for p in rec.image_paths:
            b, fmt = load_image_bytes(p)
            images_bytes.append(b)
            image_format = fmt
    user_text = build_user_text(rec, composite=composite)

    last_exc: Exception | None = None
    last_res: dict | None = None
    last_pred: str | None = None
    attempts: list[dict] = []

    async with sem:
        for attempt in range(retries + 1):
            try:
                res = await client.generate(
                    system=SYSTEM_PROMPT,
                    user_text=user_text,
                    images=images_bytes,
                    image_format=image_format,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    label_images=not composite,
                )
                pred = parse_letter(res["text"], rec.n_choices)
                attempts.append({
                    "attempt":   attempt,
                    "status":    "ok" if pred is not None else "unparseable",
                    "pred":      pred,
                    "text":      res.get("text", ""),
                    "reasoning": res.get("reasoning", ""),
                    "input_tokens":  res["usage"]["input_tokens"],
                    "output_tokens": res["usage"]["output_tokens"],
                    "latency_ms":    res["latency_ms"],
                })
                last_res = res
                last_pred = pred
                if pred is not None:
                    break          # success
                # parseable retry: keep going
            except (BedrockError, OpenRouterError) as exc:
                attempts.append({
                    "attempt": attempt,
                    "status":  f"error:{type(exc).__name__}",
                    "error":   str(exc)[:500],
                })
                last_exc = exc

        # ---- Build row (success or final failure) ----
    if last_res is not None:
        row = {
            "id": rec.id, "axis": rec.axis, "leaf": rec.leaf,
            "n_choices": rec.n_choices,
            "gold_letter": rec.answer_letter,
            "pred_letter": last_pred,
            "correct":     (last_pred == rec.answer_letter),
            "raw_response":      last_res.get("text", ""),
            "reasoning_text":    last_res.get("reasoning", ""),
            "reasoning_details": last_res.get("reasoning_details", []),
            "raw_message":       last_res.get("raw_message", {}),
            "input_tokens":  last_res["usage"]["input_tokens"],
            "output_tokens": last_res["usage"]["output_tokens"],
            "latency_ms":    last_res["latency_ms"],
            "status":  "ok" if last_pred is not None else "unparseable",
            "n_attempts": len(attempts),
            "all_attempts": attempts,
            "n_context_tiles": len(rec.context_tiles or []),
        }
    else:
        row = {
            "id": rec.id, "axis": rec.axis, "leaf": rec.leaf,
            "n_choices": rec.n_choices,
            "gold_letter": rec.answer_letter,
            "pred_letter": None, "correct": False,
            "raw_response": "", "reasoning_text": "", "reasoning_details": [],
            "raw_message": {},
            "input_tokens": 0, "output_tokens": 0, "latency_ms": 0,
            "status":  f"error:{type(last_exc).__name__ if last_exc else 'Unknown'}",
            "error_msg": str(last_exc)[:500] if last_exc else "",
            "n_attempts": len(attempts),
            "all_attempts": attempts,
            "n_context_tiles": len(rec.context_tiles or []),
        }
    if rec.quality_flags is not None:
        row["quality_flags"] = rec.quality_flags
    return row


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def amain(args: argparse.Namespace) -> None:
    load_local_env()
    if args.model == "haiku":
        model_id = DEFAULT_HAIKU_MODEL_ID
    elif args.model == "sonnet":
        model_id = DEFAULT_SONNET_MODEL_ID
    elif args.model == "opus":
        model_id = DEFAULT_OPUS_MODEL_ID
    else:
        model_id = args.model  # raw model ID

    if args.provider == "bedrock":
        client = BedrockVLMClient(model_id=model_id)
    elif args.provider == "openrouter":
        # raw slug expected — OpenRouter has no shortcut aliases in this script
        client = OpenRouterVLMClient(model_id=args.model)
    else:
        raise SystemExit(f"unsupported provider: {args.provider}")

    if args.include_context and not args.composite:
        print("--include-context implies --composite; enabling.", file=sys.stderr)
        args.composite = True

    sem = asyncio.Semaphore(args.concurrency)

    records = load_records(args.questions, args.dataset_root,
                           drop_dup_methods=True, rotate=args.rotate,
                           include_context=args.include_context,
                           tag_quality=args.tag_quality,
                           data_points_path=args.data_points)
    if args.limit:
        records = records[: args.limit]
        print(f"--limit {args.limit}: running on first {len(records)} records", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_fh = args.out.open("w", encoding="utf-8")

    correct = 0
    done = 0
    errors = 0

    async def run_and_write(rec: Record) -> None:
        nonlocal correct, done, errors
        result = await run_one(client, rec, sem, max_tokens=args.max_tokens,
                               composite=args.composite, retries=args.retries)
        out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_fh.flush()
        done += 1
        if result["status"] == "ok" and result["correct"]:
            correct += 1
        if result["status"] != "ok":
            errors += 1
        if args.limit or done % 20 == 0:
            mark = "✓" if result["correct"] else ("✗" if result["status"] == "ok" else "E")
            print(f"  [{done:4d}/{len(records)}] {mark} {result['id'][:60]:60s} "
                  f"pred={result['pred_letter']} gold={result['gold_letter']} "
                  f"({result['latency_ms']}ms, in={result['input_tokens']}, out={result['output_tokens']})",
                  file=sys.stderr)

    tasks = [asyncio.create_task(run_and_write(r)) for r in records]
    await asyncio.gather(*tasks)
    out_fh.close()

    answered = done - errors
    print(file=sys.stderr)
    print(f"Wrote {done} predictions -> {args.out}", file=sys.stderr)
    print(f"Errors            : {errors}", file=sys.stderr)
    if answered:
        print(f"Accuracy (of {answered} answered): {correct}/{answered} = "
              f"{100*correct/answered:.2f}%", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions",   type=Path, default=QUESTIONS_DEFAULT)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    p.add_argument("--provider",    default="bedrock", choices=["bedrock", "openrouter"])
    p.add_argument("--model",       default="haiku",
                   help="'haiku' / 'sonnet' / 'opus' shortcut, or a raw Bedrock model ID")
    p.add_argument("--out",         type=Path,
                   default=REPO / "outputs" / "run.jsonl")
    p.add_argument("--limit",       type=int, default=0,
                   help="Run only the first N records (smoke test).")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens",  type=int, default=None,
                   help="If unset, the model decides output length freely. "
                        "Required to be unset for reasoning models that emit "
                        "long chain-of-thought before the final letter.")
    p.add_argument("--retries",     type=int, default=2,
                   help="Per-record retry budget. A record retries up to N "
                        "times if the call errors OR if the response can't be "
                        "parsed into a letter A..D.")
    p.add_argument("--composite",   action="store_true",
                   help="Send one composite-grid image with hard-rendered (A)/(B)/... "
                        "banners instead of N separate image blocks. Recommended — "
                        "removes text-block-to-image-block binding ambiguity.")
    p.add_argument("--rotate",      type=int, default=0,
                   help="Cyclically right-shift candidate slots by K positions. "
                        "Used for rotation-diagnostic to separate position bias "
                        "from content judgment.")
    p.add_argument("--data-points", type=Path, default=DATA_POINTS_DEFAULT,
                   help="data_points.jsonl source (needed for --include-context "
                        "and --tag-sampling-ambiguity).")
    p.add_argument("--include-context", action="store_true",
                   help="Pull same-group GT/INPUT/REFERENCE bboxes (bbox_type=main) "
                        "and render in a top REFERENCE row of the composite. "
                        "Implies --composite.")
    p.add_argument("--tag-quality", action="store_true",
                   help="Attach a quality_flags dict to each output row. "
                        "Includes benchmark_eligible (bool), ineligible_reasons, "
                        "and orthogonal diagnostic flags (has_reference_context, "
                        "repeated_main_crop, distinct_row_viz_ambiguity, etc.) "
                        "so bench_score.py can slice without re-running.")
    p.add_argument("--log-level",   default="WARNING")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
