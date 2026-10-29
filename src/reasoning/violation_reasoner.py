"""
Violation Reasoner — Stage 4.

Takes each detected event + retrieved evidence + the original manual text
and asks GPT-4o to confirm violations, cite the exact rule, and recommend actions.

Nothing hardcoded — all rules come from the uploaded manual or universal principles.
"""

from __future__ import annotations
import uuid

from loguru import logger

from config.settings import settings, OUTPUT_DIR
from src.utils.models import (
    DetectedEvent, RetrievedEvidence, Violation,
    VideoAnalysisResult, InvestigationReport,
)
from src.utils.json_utils import parse_llm_json
from src.utils.llm_client import get_client


VIOLATION_PROMPT = """\
You are a certified laboratory safety investigator.

{manual_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETECTED EVENT (from video frame analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Timestamp  : {timestamp}s  ← this is the EXACT video time of the violation
Description: {description}
Objects    : {objects}
Actions    : {actions}
AI severity estimate: {vlm_severity}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED PASSAGES FROM THE LAB MANUAL / SOPs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{sop_evidence}

RETRIEVED EQUIPMENT / PROCEDURE NOTES
{manual_evidence}

SIMILAR PAST INCIDENTS
{incident_evidence}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Is this a genuine safety violation? (be inclusive — if in doubt, say yes)
2. Cite the EXACT rule from the manual above. If not in manual, state the universal principle.
3. Give the EXACT timestamp (already provided above — use it verbatim).
4. Explain specifically what went wrong and the concrete danger it creates.
5. Give clear, actionable recommendations specific to what was observed.

Rules for citation:
- Quote the specific section/clause number and text if available.
- If the manual does not cover this, start your rule_violated with "Universal: ".
- NEVER cite a rule that is not relevant to what was actually observed.

Respond ONLY with a valid JSON object — no prose, no markdown fences:
{{
  "is_violation": <true|false>,
  "violation_type": "<PPE | Chemical Handling | Equipment Misuse | Fire Hazard | Biological Safety | Waste Disposal | Emergency Procedure | Food/Drink | Other>",
  "rule_violated": "<exact quote or section from the manual, or 'Universal: ...' if not in manual>",
  "explanation": "<2-3 sentences: exactly what was observed, why it violates the rule, and the specific danger>",
  "severity": "<low|medium|high|critical>",
  "timestamp_sec": {timestamp},
  "immediate_action": "<specific action needed RIGHT NOW — name the person and the action>",
  "recommended_action": "<corrective action for next 24-72 hours>",
  "preventive_measure": "<systemic change to prevent recurrence>"
}}
"""

SUMMARY_PROMPT = """\
You are a laboratory safety director writing an official investigation report.

Video duration : {duration}s
Total events   : {total_events}
Violations found: {num_violations}

VIOLATIONS DETECTED:
{violations_summary}

Write a professional executive summary (4-6 sentences) covering:
- What happened overall
- The key violations and their severity
- The overall risk level
- The top 3 most urgent actions

Respond ONLY with a valid JSON object — no prose, no markdown fences:
{{
  "executive_summary": "<paragraph>",
  "overall_risk_level": "<low|medium|high|critical>",
  "key_findings": ["<finding 1>", "<finding 2>", ...],
  "immediate_actions": ["<action 1>", "<action 2>", "<action 3>"]
}}
"""

_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_NO_MANUAL_CTX = (
    "No lab manual was uploaded. Apply universal laboratory safety principles "
    "(PPE requirements, chemical handling, equipment safety, emergency procedures)."
)


class ViolationReasoner:

    def __init__(self):
        self.client = get_client()
        logger.info(f"ViolationReasoner ready")

    def analyse_event(
        self,
        event: DetectedEvent,
        evidence: dict[str, list[RetrievedEvidence]],
        manual_text: str = "",
    ) -> Violation | None:
        manual_ctx = _build_manual_ctx(manual_text)
        prompt = VIOLATION_PROMPT.format(
            manual_context     = manual_ctx,
            timestamp          = event.timestamp_sec,
            description        = event.description,
            objects            = ", ".join(event.objects_involved) or "N/A",
            actions            = ", ".join(event.actions_observed) or "N/A",
            vlm_severity       = event.severity,
            sop_evidence       = _fmt(evidence.get("sops", [])),
            manual_evidence    = _fmt(evidence.get("manuals", [])),
            incident_evidence  = _fmt(evidence.get("incidents", [])),
        )
        try:
            resp = self.client.chat(
                prompt      = prompt,
                temperature = 0.15,
                max_tokens  = 800,
            )
            data = parse_llm_json(resp.text)
            if not data.get("is_violation", False):
                logger.debug(f"No violation confirmed @ {event.timestamp_sec}s")
                return None
            return Violation(
                violation_id       = str(uuid.uuid4()),
                event              = event,
                violation_type     = data.get("violation_type", "Unknown"),
                rule_violated      = data.get("rule_violated", "Unspecified"),
                explanation        = data.get("explanation", ""),
                severity           = data.get("severity", event.severity),
                sop_evidence       = evidence.get("sops", []),
                similar_incidents  = evidence.get("incidents", []),
                equipment_notes    = evidence.get("manuals", []),
                immediate_action   = data.get("immediate_action", ""),
                recommended_action = data.get("recommended_action", ""),
                preventive_measure = data.get("preventive_measure", ""),
            )
        except Exception as e:
            logger.error(f"Reasoning failed @ {event.timestamp_sec}s: {e}")
            return None

    def build_report(
        self,
        video_result: VideoAnalysisResult,
        violations: list[Violation],
        report_id: str | None = None,
        output_dir: str | None = None,
    ) -> InvestigationReport:
        report_id = report_id or str(uuid.uuid4())
        viol_lines = "\n".join(
            f"- [{v.severity.upper()}] {v.violation_type} @ {v.event.timestamp_sec:.1f}s: {v.explanation[:120]}"
            for v in violations
        ) or "No violations confirmed."

        prompt = SUMMARY_PROMPT.format(
            duration          = video_result.duration_sec,
            total_events      = len(video_result.events),
            num_violations    = len(violations),
            violations_summary = viol_lines,
        )

        summary      = video_result.summary
        overall_risk = _derive_risk(violations)
        findings: list[str]  = []
        actions: list[str]   = []

        try:
            resp = self.client.chat(
                prompt      = prompt,
                temperature = 0.2,
                max_tokens  = 600,
            )
            data         = parse_llm_json(resp.text)
            summary      = data.get("executive_summary", summary)
            overall_risk = data.get("overall_risk_level", overall_risk)
            findings     = data.get("key_findings", [])
            actions      = data.get("immediate_actions", [])
        except Exception as e:
            logger.warning(f"Report summary failed (fallback): {e}")
            # Build a real summary from the violations we have
            if violations:
                viol_types = list({v.violation_type for v in violations})
                overall_risk = _derive_risk(violations)
                summary = (
                    f"The investigation identified {len(violations)} safety violation(s) in the video footage. "
                    f"Violation categories detected: {', '.join(viol_types)}. "
                    f"Overall risk level is assessed as {overall_risk.upper()}. "
                    f"Immediate corrective actions are required as detailed in the violation section below."
                )
                findings = [f"{v.violation_type} at {v.event.timestamp_sec:.1f}s" for v in violations[:5]]
                actions  = [v.immediate_action for v in violations[:3] if v.immediate_action]

        return InvestigationReport(
            report_id                  = report_id,
            video_path                 = video_result.video_path,
            incident_summary           = summary,
            timeline                   = video_result.events,
            violations                 = violations,
            overall_risk_level         = overall_risk,
            key_findings               = findings,
            immediate_actions_required = actions,
            metadata = {
                "duration_sec":          video_result.duration_sec,
                "frames_sampled":        video_result.total_frames_sampled,
                "total_events_detected": len(video_result.events),
                "output_dir":            output_dir or str(OUTPUT_DIR),
            },
        )


def _derive_risk(violations: list[Violation]) -> str:
    if not violations:
        return "low"
    return max(violations, key=lambda v: _SEVERITY_RANK.get(v.severity, 0)).severity


def _fmt(items: list[RetrievedEvidence], n: int = 3) -> str:
    if not items:
        return "  (none retrieved)"
    lines = []
    for i, e in enumerate(items[:n], 1):
        snippet = e.chunk.content[:300].replace("\n", " ")
        lines.append(f"  {i}. [{e.chunk.title}] (score={e.score:.2f}) — {snippet}…")
    return "\n".join(lines)


def _build_manual_ctx(text: str, max_chars: int = 6000) -> str:
    if not text or not text.strip():
        return _NO_MANUAL_CTX
    t = text.strip()
    if len(t) > max_chars:
        cut = t.rfind(" ", 0, max_chars)
        t   = t[:cut] + "\n[...truncated...]"
    return (
        "LAB MANUAL CONTEXT (treat as authoritative):\n"
        "───────────────────────────────────────────\n"
        f"{t}\n"
        "───────────────────────────────────────────"
    )
