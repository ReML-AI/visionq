"""OpenRouter chat client with vision (image) content support for the VisionQ benchmark.

Self-contained OpenRouter client.
Adds OpenAI-style image_url content blocks
so the existing P1 composite-grid prompt works with any vision-capable
OpenRouter model.

Auth: OPENROUTER_API_KEY env var (or repo-root /.env loaded by lib.bedrock_vlm
if already present in os.environ).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

LOG = logging.getLogger(__name__)
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_repo_env() -> None:
    """Load repo-root /.env (where OPENROUTER_API_KEY lives) without overwriting."""
    p = REPO_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_openrouter_key() -> str:
    load_repo_env()
    return os.environ.get("OPENROUTER_API_KEY", "")


class OpenRouterError(RuntimeError):
    pass


def _parse_letter_lenient(content: str) -> str:
    return (content or "").strip()


def _usage(payload: dict) -> dict[str, int]:
    u = payload.get("usage") or {}
    i = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    o = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


@dataclass
class OpenRouterVLMClient:
    model_id: str = ""
    api_key: str = ""
    timeout_s: float = 240.0
    max_retries: int = 4
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    def _ensure(self) -> tuple[httpx.AsyncClient, str]:
        if not self.api_key:
            self.api_key = load_openrouter_key()
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY not set (looked in env + repo-root /.env)."
            )
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client, self.api_key

    async def generate(
        self,
        system: str,
        user_text: str,
        images: list[bytes],
        image_format: str = "png",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        label_images: bool = True,
    ) -> dict[str, Any]:
        """Match the BedrockVLMClient.generate() interface exactly so bench_run
        can dispatch to either provider without other code changes.

        Returns dict with text / usage / latency_ms / raw_blocks.
        """
        client, api_key = self._ensure()

        # Build OpenAI-style multimodal content. Composite-grid mode (the only
        # one used by bench_run today) sends one image and label_images=False;
        # the per-image label fallback path is included for completeness.
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for idx, img_bytes in enumerate(images):
            if label_images:
                letter = chr(ord("A") + idx)
                user_content.append({"type": "text", "text": f"Image ({letter}):"})
            b64 = base64.b64encode(img_bytes).decode("ascii")
            mime = "image/png" if image_format == "png" else f"image/{image_format}"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            "temperature": float(temperature),
            "usage": {"include": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/spatial-as-judge",
            "X-Title":       "VisionQ-benchmark",
        }

        delay = 2.0
        last_err = ""
        t0 = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                resp = await client.post(
                    f"{OPENROUTER_BASE}/chat/completions",
                    json=payload, headers=headers,
                )
                if resp.status_code == 429:
                    await asyncio.sleep(delay * (2 ** attempt))
                    last_err = "429 rate-limit"; continue
                if resp.status_code >= 500:
                    await asyncio.sleep(delay * (2 ** attempt))
                    last_err = f"HTTP {resp.status_code} server-err"; continue
                if resp.status_code in (400, 403, 404):
                    try:
                        err = resp.json()
                    except Exception:
                        err = {"error": resp.text[:300]}
                    raise OpenRouterError(
                        f"HTTP {resp.status_code} for model={self.model_id!r}: "
                        f"{str(err)[:300]}"
                    )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise OpenRouterError(
                        f"empty choices from {self.model_id!r}: {str(data)[:300]}"
                    )
                msg = choices[0].get("message") or {}
                content = msg.get("content", "") or ""
                # Many providers stash chain-of-thought in `reasoning` or
                # `reasoning_details`. Capture everything verbatim.
                reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                reasoning_details = msg.get("reasoning_details") or []
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return {
                    "text": content,
                    "reasoning": reasoning,
                    "reasoning_details": reasoning_details,
                    "raw_message": msg,        # full message dict (verbatim)
                    "raw_response": data,      # full top-level response
                    "usage": _usage(data),
                    "latency_ms": latency_ms,
                    "raw_blocks": [{"text": content}],
                }
            except httpx.TimeoutException:
                await asyncio.sleep(delay * (2 ** attempt))
                last_err = "timeout"
            except OpenRouterError:
                raise
            except Exception as e:
                await asyncio.sleep(delay * (2 ** attempt))
                last_err = f"{type(e).__name__}: {e}"
        raise OpenRouterError(
            f"openrouter call exhausted retries for {self.model_id!r}: {last_err}"
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
