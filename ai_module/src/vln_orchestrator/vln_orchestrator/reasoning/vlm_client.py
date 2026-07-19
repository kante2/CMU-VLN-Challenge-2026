#!/usr/bin/env python3
"""OpenAI-compatible VLM client for the challenge reasoning pipeline.

Provider config adapted from SysNav (vlm_node/constants.py, BSD-3, Haokun Zhu).
Supports Gemini and Qwen (DashScope) through their OpenAI-compatible endpoints.

`openai` is imported lazily so this module (and the decomposition/verification
logic that imports it) can be unit-tested off-robot without the dependency or an
API key. Construct VLMClient() only when you actually need to call the model.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from the nearest `.env` into os.environ (without
    overriding already-set vars). Dependency-free: walks up from this file to
    the repo root looking for `.env`. Lets API keys live in a gitignored file
    instead of being exported manually before launch. No-op if none found."""
    for base in [Path(__file__).resolve(), Path.cwd().resolve()]:
        for parent in [base, *base.parents]:
            env = parent / ".env"
            if env.is_file():
                for line in env.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
                return


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    model_lite: str


def _answer_gemini_key() -> str:
    """Gemini key for the answer path. Prefers ORCHESTRATOR_GEMINI_API_KEY so the
    orchestrator can run on a separate free-tier quota from vlm_node (exploration),
    which reads GEMINI_API_KEY. Falls back to GEMINI_API_KEY for single-key runs."""
    return (os.environ.get("ORCHESTRATOR_GEMINI_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", ""))


def _answer_dashscope_key() -> str:
    return (os.environ.get("ORCHESTRATOR_DASHSCOPE_API_KEY", "")
            or os.environ.get("DASHSCOPE_API_KEY", ""))


def _answer_groq_key() -> str:
    return (os.environ.get("ORCHESTRATOR_GROQ_API_KEY", "")
            or os.environ.get("GROQ_API_KEY", ""))


def answer_key_available() -> bool:
    """True if the answer path has any usable provider key. Used by handlers to
    decide whether to build a VLMClient (vs. heuristic fallback)."""
    _load_dotenv()
    return bool(_answer_gemini_key() or _answer_dashscope_key() or _answer_groq_key())


def resolve_provider() -> ProviderConfig:
    """Pick provider from VLM_PROVIDER, else whichever key is set."""
    _load_dotenv()
    gemini = _answer_gemini_key()
    dashscope = _answer_dashscope_key()
    groq = _answer_groq_key()
    provider = os.environ.get("VLM_PROVIDER", "").lower()
    if provider not in ("gemini", "qwen", "groq"):
        if gemini:
            provider = "gemini"
        elif groq:
            provider = "groq"
        elif dashscope:
            provider = "qwen"
        else:
            provider = "gemini"

    if provider == "groq":
        # Groq is OpenAI-compatible. Use a VISION model (verify_candidate sends
        # image crops): llama-4-scout supports image_url. Groq lacks the typed
        # beta.parse helper -> parse() falls back to JSON-object mode.
        return ProviderConfig(
            provider="groq",
            api_key=groq,
            base_url="https://api.groq.com/openai/v1",
            model=os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
            model_lite=os.environ.get("GROQ_MODEL_LITE", "meta-llama/llama-4-scout-17b-16e-instruct"),
        )
    if provider == "qwen":
        return ProviderConfig(
            provider="qwen",
            api_key=dashscope,
            base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            model=os.environ.get("QWEN_MODEL", "qwen3.6-plus"),
            model_lite=os.environ.get("QWEN_MODEL_LITE", "qwen3.6-flash"),
        )
    return ProviderConfig(
        provider="gemini",
        api_key=gemini,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        model_lite=os.environ.get("GEMINI_MODEL_LITE", "gemini-2.5-flash-lite"),
    )


def encode_image_jpg(image_bgr) -> str:
    """Encode an OpenCV BGR image to a base64 JPEG data string (no header)."""
    import cv2  # local import; only needed at runtime on the robot

    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return base64.b64encode(buf).decode("utf-8")


class VLMClient:
    """Thin wrapper over the OpenAI SDK with structured-output helpers."""

    def __init__(self, cfg: ProviderConfig | None = None) -> None:
        from openai import OpenAI  # lazy

        self.cfg = cfg or resolve_provider()
        if not self.cfg.api_key:
            raise RuntimeError(
                f"No API key for provider '{self.cfg.provider}'. Set "
                "GEMINI_API_KEY, GROQ_API_KEY or DASHSCOPE_API_KEY."
            )
        self._client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)

    def parse(self, system: str, content, response_format, lite: bool = False):
        """Structured-output call. `content` is the OpenAI user-content list;
        `response_format` is a pydantic model. Returns the parsed object.

        Providers without OpenAI's typed `beta.parse` helper (e.g. Groq) use a
        JSON-object fallback: ask for strict JSON, then validate with pydantic."""
        model = self.cfg.model_lite if lite else self.cfg.model
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        if self.cfg.provider != "groq":
            try:
                completion = self._client.beta.chat.completions.parse(
                    model=model, messages=msgs, response_format=response_format,
                )
                return completion.choices[0].message.parsed
            except Exception:
                pass  # fall through to JSON-object mode
        return self._parse_json(model, msgs, response_format)

    def _parse_json(self, model, msgs, response_format):
        """JSON-object structured output: inject the schema into the system prompt,
        request strict JSON, validate into the pydantic model."""
        import json
        schema = response_format.model_json_schema()
        msgs = list(msgs)
        msgs[0] = {"role": "system", "content": msgs[0]["content"]
                   + "\nReturn ONLY a JSON object matching this schema (no prose): "
                   + json.dumps(schema)}
        completion = self._client.chat.completions.create(
            model=model, messages=msgs, response_format={"type": "json_object"},
        )
        return response_format.model_validate_json(completion.choices[0].message.content)

    @staticmethod
    def text(text: str) -> dict:
        return {"type": "text", "text": text}

    @staticmethod
    def image_b64(b64: str) -> dict:
        return {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
