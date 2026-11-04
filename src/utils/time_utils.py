"""Shared timestamp formatting for human-readable display in reports and summaries."""

from __future__ import annotations


def format_human_time(seconds: float | int | None) -> str:
    """Return a display time in MM:SS or H:MM:SS format.

    Examples:
    - 0 -> "00:00"
    - 12.5 -> "00:12"
    - 3661 -> "1:01:01"
    """
    if seconds is None:
        return "00:00"
    total = int(float(seconds))
    if total < 0:
        total = 0

    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
