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
import time
from loguru import logger
from src.utils.json_utils import parse_llm_json


class LLMResponse:
    def __init__(self, text: str, truncated: bool = False):
        self.text = text.strip()
        self.truncated = truncated
    def json(self) -> dict:
        return parse_llm_json(self.text)


class FatalLLMError(RuntimeError):
    """Unrecoverable for the whole run (all keys exhausted / project blocked) — must not be swallowed per frame."""


class TruncatedResponseError(RuntimeError):
    """The model stopped because it hit the output-token limit (reply is incomplete)."""


# Gemini 2.5+/3.x models "think" before answering and those thinking tokens count
# against max_output_tokens. Without headroom, JSON answers get cut off mid-string
# and the frame is silently dropped. Callers pass the size of the *answer* they
# need; this headroom is added on top for thinking.
_THINKING_HEADROOM = int(os.getenv("LABSENTINEL_THINKING_HEADROOM", "3072"))
# Optional hard cap on thinking (e.g. 0 to disable on models that allow it). Unset = model default.
_THINKING_BUDGET = os.getenv("LABSENTINEL_THINKING_BUDGET", "").strip()
# Per-request timeout so a stuck call can't hang an investigation forever.
_REQUEST_TIMEOUT_SEC = float(os.getenv("LABSENTINEL_REQUEST_TIMEOUT", "120"))
# Retries for transient server errors (503 overloaded, 500, timeouts).
_TRANSIENT_RETRIES = int(os.getenv("LABSENTINEL_TRANSIENT_RETRIES", "4"))
_TRANSIENT_MARKERS = ("503", "unavailable", "500", "internal", "overloaded",
                      "deadline", "timed out", "timeout", "connection", "504")


def _is_transient_error(error: Exception | str) -> bool:
    msg = str(error).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


# ── Gemini Client with key rotation ──────────────────────────────────────────

class GeminiClient:
    # Remember which key last worked (per key list) so new clients created by other
    # pipeline stages don't start again at an already-exhausted key 1.
    _shared_idx: dict[tuple, int] = {}

    def __init__(self, keys: list[str], model: str = "gemini-3.5-flash-lite"):
        if not keys:
            raise ValueError("No Gemini API keys provided.")
        from google import genai
        from google.genai import types
        self._types   = types
        self._genai   = genai
        self._keys    = keys
        self._idx     = min(GeminiClient._shared_idx.get(tuple(keys), 0), len(keys) - 1)
        self.model    = model
        self._client  = self._make(keys[self._idx])
        logger.info(f"GeminiClient — model: {model} | {len(keys)} key(s)")

    @staticmethod
    def _is_blocked_project_error(error: Exception | str) -> bool:
        msg = str(error).lower()
        if "permission_denied" not in msg and "403" not in msg and "project unavailable" not in msg:
            return False
        blocked_markers = (
            "free tier is not available",
            "project unavailable",
            "billing to upgrade to the paid tier",
            "this project is unavailable",
            "not available for this project",
            "billing",
        )
        return any(marker in msg for marker in blocked_markers)

    def _make(self, key):
        return self._genai.Client(
            api_key=key,
            http_options=self._types.HttpOptions(timeout=int(_REQUEST_TIMEOUT_SEC * 1000)),
        )

    def _rotate(self) -> bool:
        if self._idx + 1 < len(self._keys):
            self._idx   += 1
            self._client = self._make(self._keys[self._idx])
            GeminiClient._shared_idx[tuple(self._keys)] = self._idx
            logger.warning(f"Rotated to Gemini key {self._idx + 1}/{len(self._keys)}")
            return True
        return False

    def chat(self, prompt: str, images: list[str] | None = None,
             temperature: float = 0.1, max_tokens: int = 900) -> LLMResponse:
        """
        max_tokens is the size of the answer needed; thinking headroom is added.
        Retries transient errors with backoff, rotates keys on quota errors,
        and retries once with a bigger budget if the reply was cut off.
        Raises instead of returning a silently-truncated answer.
        """
        budget = max_tokens + _THINKING_HEADROOM
        transient_attempts = 0
        truncation_retried = False

        while True:
            try:
                resp = self._call(prompt, images, temperature, budget)
                if resp.truncated:
                    if not truncation_retried:
                        truncation_retried = True
                        budget *= 2
                        logger.warning(f"Gemini reply hit token limit — retrying with max_output_tokens={budget}")
                        continue
                    raise TruncatedResponseError(
                        f"Gemini reply was cut off at max_output_tokens={budget}. "
                        "Raise LABSENTINEL_THINKING_HEADROOM in .env.")
                return resp
            except TruncatedResponseError:
                raise
            except Exception as e:
                if self._is_blocked_project_error(e):
                    msg = (
                        "Gemini API access is blocked for this project.\n\n"
                        "The Google project behind this key is not allowed to use the Gemini API "
                        "for free-tier access, and billing/project access is not enabled.\n\n"
                        "Fix:\n"
                        "  1. Open https://aistudio.google.com and confirm the project is enabled\n"
                        "  2. Turn on billing or create a new Gemini-enabled project\n"
                        "  3. Add a valid key to .env (GOOGLE_API_KEY=...)\n"
                        "  4. Restart the app"
                    )
                    logger.error(msg)
                    raise FatalLLMError(msg) from e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if self._rotate():
                        continue
                    # Start from key 1 again next time (quotas reset daily)
                    GeminiClient._shared_idx.pop(tuple(self._keys), None)
                    msg = (
                        f"All {len(self._keys)} configured Gemini API key(s) are exhausted.\n\n"
                        "Please add a new key or wait for the daily quota to reset.\n\n"
                        "To fix:\n"
                        "  1. Add more keys to .env (GOOGLE_API_KEY_2=..., GOOGLE_API_KEY_3=...)\n"
                        "  2. Get free keys at https://aistudio.google.com/apikey\n"
                        "  3. Wait until midnight UTC for quota reset\n"
                        "  4. Reduce Max Frames to 10 in Advanced Settings"
                    )
                    logger.error(msg)
                    raise FatalLLMError(msg) from e
                if _is_transient_error(e) and transient_attempts < _TRANSIENT_RETRIES:
                    transient_attempts += 1
                    delay = min(2 ** transient_attempts, 30)
                    logger.warning(f"Transient Gemini error ({str(e)[:80]}) — "
                                   f"retry {transient_attempts}/{_TRANSIENT_RETRIES} in {delay}s")
                    time.sleep(delay)
                    continue
                raise

    def _call(self, prompt, images, temperature, max_tokens) -> LLMResponse:
        parts = []
        if images:
            import base64
            for b64 in images:
                parts.append(self._types.Part.from_bytes(
                    data=base64.b64decode(b64), mime_type="image/jpeg"))
        parts.append(self._types.Part.from_text(text=prompt))
        cfg = dict(temperature=temperature, max_output_tokens=max_tokens)
        if _THINKING_BUDGET:
            cfg["thinking_config"] = self._types.ThinkingConfig(thinking_budget=int(_THINKING_BUDGET))
        resp = self._client.models.generate_content(
            model    = self.model,
            contents = [self._types.Content(role="user", parts=parts)],
            config   = self._types.GenerateContentConfig(**cfg),
        )
        truncated = False
        try:
            fr = resp.candidates[0].finish_reason if resp.candidates else None
            truncated = fr is not None and "MAX_TOKENS" in str(fr)
        except Exception:
            pass
        return LLMResponse(resp.text or "", truncated=truncated)


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
            max_tokens=max_tokens, temperature=temperature, timeout=_REQUEST_TIMEOUT_SEC)
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise TruncatedResponseError(f"OpenAI reply was cut off at max_tokens={max_tokens}.")
        return LLMResponse(choice.message.content or "")


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
