"""Bedrock Converse client with image (vision) support for the VisionQ benchmark.

Self-contained Bedrock client.
extended with image content blocks and trimmed to only what this experiment needs
(no local-llama path, no JSON-only mode — this benchmark expects free-text MCQ
answers).

Auth flows entirely through boto3's default credential chain. Supported env vars:
  - AWS_BEARER_TOKEN_BEDROCK    (Bedrock API key, 2024+)
  - AWS_PROFILE                 (named profile from ~/.aws/credentials)
  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY [+ AWS_SESSION_TOKEN]
Region from AWS_REGION or AWS_DEFAULT_REGION.

Credential strings are sanitized out of any logged error / traceback.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

DEFAULT_HAIKU_MODEL_ID  = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_SONNET_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
DEFAULT_OPUS_MODEL_ID   = "us.anthropic.claude-opus-4-6-v1"

_DOTENV_LOADED = False


def load_local_env(env_path: Path | None = None) -> None:
    """Load the repo-root .env without printing or persisting secret values."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    p = env_path or (Path(__file__).resolve().parents[2] / ".env")
    if not p.exists():
        return
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        LOG.warning("Failed to load .env: %s", _sanitize(repr(exc)))


class BedrockError(RuntimeError):
    pass


_SECRET_PATTERNS = [
    re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
    re.compile(r"(?<![A-Za-z0-9])ABSK[A-Za-z0-9+/=]{20,}"),
    re.compile(r"(?i)\b(authorization|x-amz-security-token)\s*:\s*\S+"),
    re.compile(r"(?i)\b(aws_secret_access_key|aws_session_token|aws_bearer_token_bedrock)\s*=\s*\S+"),
]


def _sanitize(msg: str) -> str:
    out = msg
    for pat in _SECRET_PATTERNS:
        out = pat.sub("<REDACTED_CREDENTIAL>", out)
    return out


def _usage(payload: dict[str, Any]) -> dict[str, int]:
    u = payload.get("usage") or {}
    i = int(u.get("inputTokens") or u.get("input_tokens") or 0)
    o = int(u.get("outputTokens") or u.get("output_tokens") or 0)
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


@dataclass
class BedrockVLMClient:
    model_id: str = DEFAULT_HAIKU_MODEL_ID
    region: str = ""
    timeout_s: float = 240.0
    max_retries: int = 3
    _client: Any = field(default=None, repr=False)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise BedrockError(
                "boto3 is not installed. `pip install boto3` (>=1.35 for Bedrock API-key auth)."
            ) from exc
        ver = tuple(int(x) for x in boto3.__version__.split(".")[:3] if x.isdigit())
        if os.environ.get("AWS_BEARER_TOKEN_BEDROCK") and ver < (1, 35, 0):
            raise BedrockError(
                f"AWS_BEARER_TOKEN_BEDROCK requires boto3>=1.35 (have {boto3.__version__})."
            )
        cfg = Config(
            retries={"max_attempts": self.max_retries, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=int(self.timeout_s),
        )
        kwargs: dict[str, Any] = {"config": cfg}
        region = self.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            kwargs["region_name"] = region
        LOG.info("Bedrock vision client init: model=%s region=%s", self.model_id, region or "<env>")
        self._client = boto3.client("bedrock-runtime", **kwargs)
        return self._client

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
        """Invoke Bedrock Converse with text + image blocks. Returns dict with
        keys: text, usage, latency_ms, raw_blocks.

        When ``label_images`` is False (composite-grid mode), images are sent
        without per-image "Image (X):" text prefixes — labels are expected to
        be rendered into the pixels.
        """
        client = self._ensure_client()

        user_blocks: list[dict[str, Any]] = [{"text": user_text}]
        for idx, img_bytes in enumerate(images):
            if label_images:
                letter = chr(ord("A") + idx)
                user_blocks.append({"text": f"Image ({letter}):"})
            user_blocks.append({
                "image": {
                    "format": image_format,
                    "source": {"bytes": img_bytes},
                }
            })

        def _invoke() -> dict[str, Any]:
            inference_cfg: dict[str, Any] = {"temperature": float(temperature)}
            if max_tokens is not None:
                inference_cfg["maxTokens"] = int(max_tokens)
            return client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": user_blocks}],
                system=[{"text": system}],
                inferenceConfig=inference_cfg,
            )

        try:
            t0 = time.perf_counter()
            response = await asyncio.to_thread(_invoke)
            latency_ms = int((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            tb = _sanitize(traceback.format_exc())
            LOG.error("Bedrock call failed: %s\n%s", _sanitize(repr(exc)), tb)
            raise BedrockError(_sanitize(f"bedrock call failed: {type(exc).__name__}")) from exc

        message = response.get("output", {}).get("message", {})
        blocks = message.get("content") or []
        text = ""
        reasoning_text = ""
        reasoning_blocks: list[dict[str, Any]] = []
        for blk in blocks:
            # Bedrock extended-thinking returns blocks like {"reasoningContent": ...}
            if "text" in blk and not text:
                text = blk["text"]
            if "reasoningContent" in blk:
                reasoning_blocks.append(blk["reasoningContent"])
                rc = blk["reasoningContent"]
                if isinstance(rc, dict) and "reasoningText" in rc:
                    inner = rc["reasoningText"]
                    if isinstance(inner, dict):
                        reasoning_text += inner.get("text", "")
                    else:
                        reasoning_text += str(inner)
        return {
            "text": text,
            "reasoning": reasoning_text,
            "reasoning_details": reasoning_blocks,
            "raw_message": message,
            "raw_response": response,
            "usage": _usage(response),
            "latency_ms": latency_ms,
            "raw_blocks": blocks,
        }
