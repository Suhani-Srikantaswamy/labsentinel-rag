"""
Unified LLM Client — wraps OpenAI and Google Gemini.

Gemini supports multiple keys with automatic rotation:
  When key 1 hits its daily quota (429), key 2 is used, then key 3, etc.

Keys are read from .env:
  GOOGLE_API_KEY      ← key 1 (required)
  GOOGLE_API_KEY_2    ← key 2 (optional)
  GOOGLE_API_KEY_3    ← key 3 (optional)
  GOOGLE_API_KEY_4    ← key 4 (optional)

Free tier: ~20 requests/day per key.
With 4 keys: ~80 requests/day = ~5 full video investigations/day free.
"""

from __future__ import annotations
import os
from loguru import logger
from src.utils.json_utils import parse_llm_json


class LLMResponse:
    def __init__(self, text: str):
        self.text = text.strip()
    def json(self) -> dict:
        return parse_llm_json(self.text)


# ── Gemini Client with key rotation ──────────────────────────────────────────

class GeminiClient:
    def __init__(self, keys: list[str], model: str = "gemini-3.5-flash-lite"):
        if not keys:
            raise ValueError("No Gemini API keys provided.")
        from google import genai
        from google.genai import types
        self._types   = types
        self._genai   = genai
        self._keys    = keys
        self._idx     = 0
        self.model    = model
        self._client  = self._make(keys[0])
        logger.info(f"GeminiClient — model: {model} | {len(keys)} key(s)")

    def _make(self, key):
        return self._genai.Client(api_key=key)

    def _rotate(self) -> bool:
        if self._idx + 1 < len(self._keys):
            self._idx   += 1
            self._client = self._make(self._keys[self._idx])
            logger.warning(f"Rotated to Gemini key {self._idx + 1}/{len(self._keys)}")
            return True
        return False

    def chat(self, prompt: str, images: list[str] | None = None,
             temperature: float = 0.1, max_tokens: int = 900) -> LLMResponse:

        for attempt in range(len(self._keys) + 1):
            try:
                return self._call(prompt, images, temperature, max_tokens)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if self._rotate():
                        continue
                    raise RuntimeError(
                        f"All {len(self._keys)} Gemini key(s) have hit their daily quota.\n\n"
                        "To fix:\n"
                        "  1. Add more keys to .env (GOOGLE_API_KEY_2=..., GOOGLE_API_KEY_3=...)\n"
                        "  2. Get free keys at https://aistudio.google.com/apikey\n"
                        "  3. Or wait until midnight UTC for quota reset\n"
                        "  4. Or reduce Max Frames to 10 in Advanced Settings"
                    ) from e
                raise

    def _call(self, prompt, images, temperature, max_tokens) -> LLMResponse:
        parts = []
        if images:
            import base64
            for b64 in images:
                parts.append(self._types.Part.from_bytes(
                    data=base64.b64decode(b64), mime_type="image/jpeg"))
        parts.append(self._types.Part.from_text(text=prompt))
        resp = self._client.models.generate_content(
            model    = self.model,
            contents = [self._types.Content(role="user", parts=parts)],
            config   = self._types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=max_tokens),
        )
        return LLMResponse(resp.text or "")


# ── OpenAI Client ─────────────────────────────────────────────────────────────

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model  = model
        logger.info(f"OpenAIClient — model: {model}")

    def chat(self, prompt: str, images: list[str] | None = None,
             temperature: float = 0.1, max_tokens: int = 900) -> LLMResponse:
        content: list = []
        if images:
            content.append({"type": "text", "text": prompt})
            for b64 in images:
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})
        else:
            content = [{"type": "text", "text": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens, temperature=temperature)
        return LLMResponse(resp.choices[0].message.content or "")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_gemini_keys() -> list[str]:
    """
    Load all Gemini API keys from environment.
    Supports unlimited keys: GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ...
    Stops scanning when it finds a gap (no key at position N means done).
    """
    keys = []
    # Primary key
    k = os.getenv("GOOGLE_API_KEY", "").strip()
    if k:
        keys.append(k)
    # Additional keys — scan until gap
    for i in range(2, 1000):
        k = os.getenv(f"GOOGLE_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
        else:
            break   # stop at first gap — no need to scan further
    return keys
    return keys


# ── Factory ───────────────────────────────────────────────────────────────────

_client_cache: dict = {}


def _invalidate_cache():
    """Clear cached clients so new keys are picked up after .env changes."""
    global _client_cache
    _client_cache = {}


def get_client(force_provider: str | None = None):
    from dotenv import load_dotenv
    load_dotenv(override=True)          # always pick up latest .env
    from config.settings import settings

    provider = force_provider or settings.llm.active_provider()

    # Re-create Gemini client each time so key list is always fresh
    if provider == "gemini":
        keys = _load_gemini_keys()
        if not keys:
            raise ValueError(
                "No Gemini API key found.\n"
                "Add to .env:  GOOGLE_API_KEY=your-key\n"
                "Get free key: https://aistudio.google.com/apikey")
        client = GeminiClient(keys=keys, model=settings.llm.gemini_model)
        return client

    if provider in _client_cache:
        return _client_cache[provider]

    if provider == "openai":
        if not settings.llm.api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        client = OpenAIClient(api_key=settings.llm.api_key, model=settings.llm.model)
    else:
        raise ValueError(
            "No API key configured.\n\n"
            "FREE: Add GOOGLE_API_KEY to .env — get at https://aistudio.google.com\n"
            "PAID: Add OPENAI_API_KEY to .env — get at https://platform.openai.com")

    _client_cache[provider] = client
    return client


def get_provider_info() -> dict:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    from config.settings import settings
    provider = settings.llm.active_provider()
    keys     = _load_gemini_keys() if provider == "gemini" else []
    info = {
        "provider": provider, "configured": provider in ("openai", "gemini"),
        "model": "", "is_free": provider == "gemini",
        "num_keys": len(keys), "status_msg": "",
    }
    if provider == "gemini":
        info["model"]      = settings.llm.gemini_model
        info["status_msg"] = (f"✅ Gemini ({settings.llm.gemini_model}) — "
                               f"{len(keys)} key(s), auto-rotation ON")
    elif provider == "openai":
        info["model"]      = settings.llm.model
        info["status_msg"] = f"✅ OpenAI ({settings.llm.model})"
    else:
        info["status_msg"] = "❌ No API key — add GOOGLE_API_KEY or OPENAI_API_KEY to .env"
    return info
