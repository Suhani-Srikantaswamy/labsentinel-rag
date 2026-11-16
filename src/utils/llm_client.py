"""
Unified LLM Client — wraps OpenAI and Google Gemini.

Gemini keys are managed by a shared, thread-safe KeyPool:
  • A key that hits its DAILY quota is skipped until the reset time Google reports
    (no more re-trying exhausted keys on every request).
  • A key that hits a PER-MINUTE limit rests only for the delay Google asks for.
  • Several "lanes" (parallel workers) can each use a different key at the same time,
    which is what makes batch/parallel video analysis fast on the free tier.

Keys are read from .env:  GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, …
Free-tier quotas are per Google *project* — keys from separate projects add capacity.
"""

from __future__ import annotations
import os
import re
import threading
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
_REQUEST_TIMEOUT_SEC = float(os.getenv("LABSENTINEL_REQUEST_TIMEOUT", "180"))
# Retries for transient server errors (503 overloaded, 500, timeouts).
_TRANSIENT_RETRIES = int(os.getenv("LABSENTINEL_TRANSIENT_RETRIES", "4"))
# If every key is only resting for a short per-minute limit, wait up to this long rather than fail
_MAX_WAIT_FOR_KEY_SEC = float(os.getenv("LABSENTINEL_MAX_WAIT_FOR_KEY", "90"))
_TRANSIENT_MARKERS = ("503", "unavailable", "500", "internal", "overloaded",
                      "deadline", "timed out", "timeout", "connection", "504")


def _is_transient_error(error: Exception | str) -> bool:
    msg = str(error).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _is_quota_error(error: Exception | str) -> bool:
    msg = str(error)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _quota_cooldown_seconds(error: Exception | str) -> tuple[float, str]:
    """
    How long a key should rest after a 429, and why.
    Uses Google's own RetryInfo / quotaId when present.
    """
    msg = str(error)
    delay = None
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?([0-9.]+)s", msg)
    if m:
        delay = float(m.group(1))
    else:
        m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([0-9.]+)s", msg)
        if m:
            delay = int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))
    per_day = "PerDay" in msg or "per day" in msg.lower()
    per_min = "PerMinute" in msg or "per minute" in msg.lower()
    if per_day:
        return (delay if delay else 6 * 3600), "daily quota used up"
    if per_min:
        return (delay if delay else 60.0) + 1.0, "per-minute limit"
    return (delay if delay else 60.0), "rate limited"


# ── Key pool (shared by every client / lane in the process) ──────────────────

class KeyPool:
    _pools: dict[tuple, "KeyPool"] = {}
    _pools_lock = threading.Lock()

    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self._until: dict[str, float] = {}       # key → unix time it may be used again
        self._why: dict[str, str] = {}
        self._lock = threading.Lock()

    @classmethod
    def for_keys(cls, keys: list[str]) -> "KeyPool":
        k = tuple(keys)
        with cls._pools_lock:
            if k not in cls._pools:
                cls._pools[k] = cls(keys)
            return cls._pools[k]

    def pick(self, lane: int = 0) -> str | None:
        """A usable key for this lane — lanes are spread across the keys that are usable now."""
        now = time.time()
        with self._lock:
            usable = [k for k in self.keys if self._until.get(k, 0) <= now]
            return usable[lane % len(usable)] if usable else None

    def rest(self, key: str, seconds: float, why: str) -> None:
        with self._lock:
            self._until[key] = time.time() + max(1.0, seconds)
            self._why[key] = why

    def mark_ok(self, key: str) -> None:
        with self._lock:
            self._until.pop(key, None)
            self._why.pop(key, None)

    def seconds_until_any_free(self) -> float:
        now = time.time()
        with self._lock:
            if not self.keys:
                return float("inf")
            return max(0.0, min(self._until.get(k, 0) for k in self.keys) - now)

    def available_count(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for k in self.keys if self._until.get(k, 0) <= now)

    def status(self) -> list[dict]:
        """Per-key status as known to this process (no API calls)."""
        now = time.time()
        out = []
        with self._lock:
            for i, k in enumerate(self.keys, 1):
                until = self._until.get(k, 0)
                out.append({
                    "index": i, "suffix": k[-4:],
                    "available": until <= now,
                    "reason": self._why.get(k, "") if until > now else "",
                    "free_in_sec": max(0, int(until - now)),
                })
        return out


# ── Gemini Client ─────────────────────────────────────────────────────────────

_request_count = 0
_request_lock = threading.Lock()


def request_count() -> int:
    """Successful LLM requests made by this process (for speed/quota reporting)."""
    return _request_count


def _count_request() -> None:
    global _request_count
    with _request_lock:
        _request_count += 1


class GeminiClient:
    _sdk_clients: dict[str, object] = {}
    _sdk_lock = threading.Lock()

    def __init__(self, keys: list[str], model: str = "gemini-3.5-flash-lite", lane: int = 0):
        if not keys:
            raise ValueError("No Gemini API keys provided.")
        from google import genai
        from google.genai import types
        self._types = types
        self._genai = genai
        self._keys  = keys
        self.lane   = lane
        self.model  = model
        self.pool   = KeyPool.for_keys(keys)
        logger.info(f"GeminiClient — model: {model} | {len(keys)} key(s) | lane {lane}")

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

    def _sdk(self, key: str):
        with GeminiClient._sdk_lock:
            c = GeminiClient._sdk_clients.get(key)
            if c is None:
                c = self._genai.Client(
                    api_key=key,
                    http_options=self._types.HttpOptions(timeout=int(_REQUEST_TIMEOUT_SEC * 1000)),
                )
                GeminiClient._sdk_clients[key] = c
            return c

    def _exhausted_message(self) -> str:
        free_in = self.pool.seconds_until_any_free()
        when = (f"about {int(free_in // 3600)}h {int(free_in % 3600 // 60)}m"
                if free_in != float("inf") else "after the daily reset")
        return (
            f"All {len(self._keys)} configured Gemini API key(s) are exhausted.\n\n"
            f"The next key becomes available in {when} (quotas reset at midnight Pacific time).\n\n"
            "To fix:\n"
            "  1. Add more keys from separate Google projects to .env (GOOGLE_API_KEY_2=…)\n"
            "  2. Get free keys at https://aistudio.google.com/apikey\n"
            "  3. Or wait for the reset"
        )

    def chat(self, prompt: str, images: list[str] | None = None,
             temperature: float = 0.1, max_tokens: int = 900) -> LLMResponse:
        """
        max_tokens is the size of the answer needed; thinking headroom is added.
        Picks a usable key for this lane, rests keys that hit quota (for as long as
        Google says), retries transient errors with backoff, and retries once with a
        bigger budget if the reply was cut off. Never returns a truncated answer.
        """
        budget = max_tokens + _THINKING_HEADROOM
        transient_attempts = 0
        truncation_retried = False

        while True:
            key = self.pool.pick(self.lane)
            if key is None:
                wait = self.pool.seconds_until_any_free()
                if wait <= _MAX_WAIT_FOR_KEY_SEC:
                    logger.info(f"All keys resting (per-minute limits) — waiting {wait:.0f}s")
                    time.sleep(wait + 0.5)
                    continue
                msg = self._exhausted_message()
                logger.error(msg)
                raise FatalLLMError(msg)

            try:
                resp = self._call(key, prompt, images, temperature, budget)
                self.pool.mark_ok(key)
                _count_request()
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
                    # This key's project can't use Gemini — sideline it and try the others
                    self.pool.rest(key, 24 * 3600, "project blocked / billing required")
                    logger.error(f"Gemini key …{key[-4:]} is blocked for this project — skipping it")
                    if self.pool.available_count() == 0:
                        raise FatalLLMError(
                            "Gemini API access is blocked for every configured key's project.\n\n"
                            "Open https://aistudio.google.com, enable the project (or billing), "
                            "add a valid key to .env and restart.") from e
                    continue
                if _is_quota_error(e):
                    seconds, why = _quota_cooldown_seconds(e)
                    self.pool.rest(key, seconds, why)
                    logger.warning(f"Gemini key …{key[-4:]}: {why} — resting {seconds/60:.0f} min; "
                                   f"{self.pool.available_count()} key(s) still available")
                    continue
                if _is_transient_error(e) and transient_attempts < _TRANSIENT_RETRIES:
                    transient_attempts += 1
                    delay = min(2 ** transient_attempts, 30)
                    logger.warning(f"Transient Gemini error ({str(e)[:80]}) — "
                                   f"retry {transient_attempts}/{_TRANSIENT_RETRIES} in {delay}s")
                    time.sleep(delay)
                    continue
                raise

    def _call(self, key, prompt, images, temperature, max_tokens) -> LLMResponse:
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
        resp = self._sdk(key).models.generate_content(
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
        _count_request()
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
    k = os.getenv("GOOGLE_API_KEY", "").strip()
    if k:
        keys.append(k)
    for i in range(2, 1000):
        k = os.getenv(f"GOOGLE_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
        else:
            break
    return list(dict.fromkeys(keys))     # drop accidental duplicates


# ── Factory ───────────────────────────────────────────────────────────────────

_client_cache: dict = {}


def _invalidate_cache():
    """Clear cached clients so new keys are picked up after .env changes."""
    global _client_cache
    _client_cache = {}


def get_client(force_provider: str | None = None, lane: int = 0):
    """`lane` lets parallel workers start on different keys."""
    from dotenv import load_dotenv
    load_dotenv(override=True)          # always pick up latest .env
    from config.settings import settings

    provider = force_provider or settings.llm.active_provider()

    if provider == "gemini":
        keys = _load_gemini_keys()
        if not keys:
            raise ValueError(
                "No Gemini API key found.\n"
                "Add to .env:  GOOGLE_API_KEY=your-key\n"
                "Get free key: https://aistudio.google.com/apikey")
        return GeminiClient(keys=keys, model=settings.llm.gemini_model, lane=lane)

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


def parallel_lanes(max_lanes: int | None = None) -> int:
    """
    How many requests can run at once. Each request takes tens of seconds, so one key
    can carry a few requests in flight without hitting its per-minute limit
    (LABSENTINEL_REQUESTS_PER_KEY); total is capped by LABSENTINEL_PARALLEL_REQUESTS.
    """
    from config.settings import settings
    cap = max_lanes or int(os.getenv("LABSENTINEL_PARALLEL_REQUESTS", "8"))
    per_key = max(1, int(os.getenv("LABSENTINEL_REQUESTS_PER_KEY", "3")))
    if settings.llm.active_provider() == "gemini":
        keys = _load_gemini_keys()
        usable = KeyPool.for_keys(keys).available_count() if keys else 0
        return max(1, min(cap, per_key * max(usable, 1)))
    return max(1, cap)


def key_status() -> list[dict]:
    """Per-key availability as known to this process (no API calls)."""
    keys = _load_gemini_keys()
    return KeyPool.for_keys(keys).status() if keys else []


def probe_keys() -> list[dict]:
    """
    Live check of every Gemini key (one tiny request per usable key — uses quota).
    Updates the pool so the app stops trying exhausted keys.
    """
    from config.settings import settings
    keys = _load_gemini_keys()
    if not keys:
        return []
    pool = KeyPool.for_keys(keys)
    client = GeminiClient(keys, model=settings.llm.gemini_model)
    for key in keys:
        try:
            client._call(key, "Reply OK", None, 0.0, 1024)
            pool.mark_ok(key)
        except Exception as e:
            if _is_quota_error(e):
                seconds, why = _quota_cooldown_seconds(e)
                pool.rest(key, seconds, why)
            elif GeminiClient._is_blocked_project_error(e):
                pool.rest(key, 24 * 3600, "project blocked / billing required")
            else:
                pool.rest(key, 300, f"error: {str(e)[:60]}")
    return pool.status()


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
        avail = KeyPool.for_keys(keys).available_count() if keys else 0
        info["model"]      = settings.llm.gemini_model
        info["keys_available"] = avail
        info["status_msg"] = (f"✅ Gemini ({settings.llm.gemini_model}) — "
                              f"{avail}/{len(keys)} key(s) available, auto-rotation ON")
    elif provider == "openai":
        info["model"]      = settings.llm.model
        info["status_msg"] = f"✅ OpenAI ({settings.llm.model})"
    else:
        info["status_msg"] = "❌ No API key — add GOOGLE_API_KEY or OPENAI_API_KEY to .env"
    return info
