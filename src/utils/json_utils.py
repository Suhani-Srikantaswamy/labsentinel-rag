"""
Shared JSON parsing utility used by VLM and reasoning modules.
Lives here so tests can import it without needing openai or chromadb.
"""
import json
from loguru import logger


def parse_llm_json(raw: str) -> dict:
    """
    Parse JSON from LLM output.
    Strips markdown code fences (```json ... ```) if present.
    Returns an empty dict on failure.
    """
    text = raw.strip()
    if not text:
        return {}
    # Strip ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed on LLM output: {text[:200]}")
        return {}


def build_event_query(description: str, objects: list, actions: list, violation: bool, severity: str) -> str:
    """Build a semantic search query from event fields."""
    parts = [description]
    if objects:
        parts.append("Equipment and materials: " + ", ".join(objects))
    if actions:
        parts.append("Actions observed: " + ", ".join(actions))
    if violation:
        parts.append(f"Potential safety violation. Severity: {severity}.")
    return " ".join(parts)
