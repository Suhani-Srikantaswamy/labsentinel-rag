"""
Shared report theme — one place for severity colours and labels used by the
HTML template, the PDF renderer and the Gradio UI.
"""

SEVERITY_ORDER = ["low", "medium", "high", "critical"]
SEVERITY_RANK = {s: i + 1 for i, s in enumerate(SEVERITY_ORDER)}

# Badge / marker colours (work on both dark and light backgrounds)
SEVERITY_COLORS = {
    "low":      "#16a34a",
    "medium":   "#d97706",
    "high":     "#ea580c",
    "critical": "#dc2626",
}
UNKNOWN_COLOR = "#94a3b8"
FALSE_ALARM_COLOR = "#64748b"

REVIEW_LABELS = {
    "pending":     "Awaiting review",
    "confirmed":   "Confirmed by reviewer",
    "false_alarm": "Marked false alarm",
}
REVIEW_COLORS = {
    "pending":     "#94a3b8",
    "confirmed":   "#16a34a",
    "false_alarm": "#64748b",
}


def severity_color(severity: str | None) -> str:
    return SEVERITY_COLORS.get(str(severity or "").lower(), UNKNOWN_COLOR)
