"""Score an OpenRouter VLM on the VisionQ-MCQ test split with the exact prompt, tiling and parser of evaluate.py
(the prompt VisionQ-Judge is evaluated with), so its accuracy sits next to base / tuned Gemma-4-E4B.
One call per question, temperature 0, no retry on unparseable answers (local eval also generates once).
Usage: python api_eval_mcq.py QUESTIONS.jsonl DATASET_ROOT MODEL_SLUG OUT.jsonl [--limit N] [--stop-usd 30]
Rows already answered in OUT are skipped; failed calls are retried on rerun."""
import argparse, asyncio, base64, io, json, os
from pathlib import Path

import httpx
from PIL import Image

import evaluate as E

KEY = next(l.split("=", 1)[1].strip().strip("\"'") for l in open(Path(__file__).resolve().parents[1] / ".env")
           if l.startswith("OPENROUTER_API_KEY="))
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "X-Title": "VisionQ-MCQ"}


def request(q, root):
    imgs = [Image.open(E.resolve_image(p, root)).convert("RGB") for p in q["image_paths"]]
    imgs = E._tile_for_eval(imgs, 512) if len(imgs) > 1 else imgs     # evaluate: --max-image-size 512, tiled
    text = E.build_prompt_text(q, prepend_context=True, anti_bias=True)
    text += f"\n\nAnswer with one of {', '.join(q['choices'])} on the first line, then optionally explain."
    content = []
    for im in imgs:
        b = io.BytesIO(); im.save(b, "PNG")
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}})
    content.append({"type": "text", "text": text})                   # images first, then text, as in the local chat template
    return content


async def call(cl, model, content):
    for a in range(5):
        try:
            r = await cl.post("https://openrouter.ai/api/v1/chat/completions", headers=H, json={
                "model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0, "usage": {"include": True}})
            if r.status_code == 429 or r.status_code >= 500:
                await asyncio.sleep(2 * 2 ** a); continue
            r.raise_for_status(); d = r.json()
            return (d["choices"][0]["message"].get("content") or ""), d.get("usage") or {}
        except (httpx.TimeoutException, httpx.TransportError):
            await asyncio.sleep(2 * 2 ** a)
    raise RuntimeError("retries exhausted")


async def spent(cl):
    return (await cl.get("https://openrouter.ai/api/v1/key", headers=H)).json()["data"]["usage"]


async def main(a):
    qs = [json.loads(l) for l in open(a.questions)]
    keep = ("pred_text", "pred_letter", "is_correct")
    qs = [{k: v for k, v in q.items() if k not in keep} for q in qs][: a.limit or None]
    done = {}
    if a.out.exists():
        for l in open(a.out):
            r = json.loads(l)
            if r.get("status") == "ok": done[r["id"]] = r
    todo = [q for q in qs if q["id"] not in done]
    print(f"{len(qs)} questions, {len(done)} done, {len(todo)} to call; model {a.model}")
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=300) as cl:
        start = await spent(cl)

        async def one(q):
            async with sem:
                try:
                    text, u = await call(cl, a.model, request(q, a.root))
                    p = E.parse_letter(text, len(q["choices"]))
                    return {**q, "pred_text": text[:400], "pred_letter": p, "is_correct": p == q["answer"].strip("()"),
                            "status": "ok", "usage": u}
                except Exception as e:
                    return {**q, "pred_letter": None, "is_correct": False, "status": f"error:{type(e).__name__}", "error": str(e)[:300]}

        with open(a.out, "a") as f:
            for i in range(0, len(todo), 40):
                for r in await asyncio.gather(*(one(q) for q in todo[i:i + 40])):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                s = await spent(cl)
                print(f"  {min(i + 40, len(todo))}/{len(todo)}  key usage ${s:.3f} (this run ${s - start:.3f})")
                if s > a.stop_usd:
                    raise SystemExit(f"stop: key usage ${s:.2f} > ${a.stop_usd}")
    rows = {}
    for l in open(a.out):
        r = json.loads(l); rows[r["id"]] = r if r.get("status") == "ok" or r["id"] not in rows else rows[r["id"]]
    ok = [r for r in rows.values() if r.get("status") == "ok"]
    print(f"answered {len(ok)}/{len(rows)}  parsed {sum(r['pred_letter'] is not None for r in ok)}  "
          f"acc {sum(r['is_correct'] for r in rows.values()) / len(rows):.3f} (failures count wrong)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("questions", type=Path); p.add_argument("root", type=Path); p.add_argument("model"); p.add_argument("out", type=Path)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--stop-usd", type=float, default=30.0)
    asyncio.run(main(p.parse_args()))
