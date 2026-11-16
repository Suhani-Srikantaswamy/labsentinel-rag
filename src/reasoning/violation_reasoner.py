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
from src.utils.llm_client import get_client, FatalLLMError
from src.utils.time_utils import format_human_time


VIOLATION_PROMPT = """\
You are a certified laboratory safety investigator specialising in {lab_type} laboratories.

{manual_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETECTED EVENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lab type   : {lab_type}
Timestamp  : {timestamp}s
Description: {description}
People     : {people}
Objects    : {objects}
ALREADY REPORTED EARLIER (do NOT report these again): {already_reported}
Actions    : {actions}
AI severity: {vlm_severity}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED PASSAGES FROM MANUAL / SOPs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{sop_evidence}

EQUIPMENT / PROCEDURE NOTES
{manual_evidence}

SIMILAR PAST INCIDENTS
{incident_evidence}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK — apply rules for a {lab_type} laboratory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. List every GENUINE safety violation in this event for a {lab_type} lab — one entry per
   distinct rule broken (e.g. missing PPE items together = ONE "PPE" entry; a spill is a
   separate entry; food at the bench is a separate entry). Only include what the evidence
   directly supports.
2. Ignore anything listed under ALREADY REPORTED EARLIER — it is already in the report.
   If nothing new remains, return an empty "violations" list.
3. For each entry, cite the EXACT rule — from the manual if present, or "Universal [{lab_type}]: ..."
4. Explain what went wrong and the specific danger, and give actionable recommendations.
5. "issues" = the issue keys this entry covers, from: {issue_keys}

Respond ONLY with valid JSON — no prose, no markdown:
{{
  "violations": [
    {{
      "violation_type": "<PPE | Chemical Handling | Equipment Misuse | Fire Hazard | Biological Safety | Electrical Safety | Ergonomics | Waste Disposal | Emergency | Food/Drink | Other>",
      "issues": ["<issue key>"],
      "person_ids": ["<P1 …, empty for scene hazards>"],
      "rule_violated": "<exact quote from manual, or 'Universal [{lab_type}]: ...' if not in manual>",
      "explanation": "<2-3 sentences: what was observed, why it violates the rule, the specific danger>",
      "severity": "<low|medium|high|critical>",
      "immediate_action": "<specific action needed RIGHT NOW>",
      "recommended_action": "<corrective action for next 24-72 hours>",
      "preventive_measure": "<systemic change to prevent recurrence>"
    }}
  ]
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
    "No lab manual was uploaded. The AI will identify the lab type from the video "
    "and apply appropriate rules for that lab type.\n"
)

_UNIVERSAL_RULES_FOOTER = (
    "\nUNIVERSAL RULES (apply for anything not explicitly in the manual above):\n"
    "Chemistry/Biology: lab coat + gloves + eye protection mandatory; no eating/drinking; "
    "containers labelled; fume hood sash at working height; centrifuge balanced.\n"
    "Physics/Engineering: appropriate PPE for equipment; no loose clothing near machinery; guards in place; "
    "lockout/tagout before service; no exposed wiring/bypassed interlocks.\n"
    "Computer Science: no food/drink near equipment; no trip hazards; exits clear; cables managed.\n"
    "Medical/Clinical: appropriate PPE and sharps disposal; disinfect equipment between uses; biohazard waste managed correctly.\n"
    "All labs: no working alone with hazardous materials; closed-toe shoes; exits clear.\n"
)


class ViolationReasoner:

    def __init__(self):
        self.client = get_client()
        self.failed_events = 0
        logger.info("ViolationReasoner ready")

    def analyse_event(
        self,
        event: DetectedEvent,
        evidence: dict[str, list[RetrievedEvidence]],
        manual_text: str = "",
    ) -> Violation | None:
        """Backwards-compatible: first confirmed violation for the event (or None)."""
        found = self.analyse_event_all(event, evidence, manual_text)
        return found[0] if found else None

    def analyse_event_all(
        self,
        event: DetectedEvent,
        evidence: dict[str, list[RetrievedEvidence]],
        manual_text: str = "",
    ) -> list[Violation]:
        """All distinct, NEW violations in one event (one per rule broken)."""
        from src.vlm.video_analyzer import ISSUE_KEYS, _norm_issue
        manual_ctx = _build_manual_ctx(manual_text)
        prompt = VIOLATION_PROMPT.format(
            manual_context     = manual_ctx,
            lab_type           = next((o.split(":")[1] for o in event.objects_involved if o.startswith("lab_type:")), "general"),
            timestamp          = event.timestamp_sec,
            description        = event.description,
            people             = _fmt_people(event),
            already_reported   = "; ".join(event.already_reported) or "nothing",
            issue_keys         = ", ".join(ISSUE_KEYS),
            objects            = ", ".join(o for o in event.objects_involved if not o.startswith("lab_type:")) or "N/A",
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
                max_tokens  = 1600,
            )
            data = parse_llm_json(resp.text)
            if not data:
                raise ValueError("unparseable reasoning reply")

            # Accept the list format, and the older single-object format
            if isinstance(data.get("violations"), list):
                items = [x for x in data["violations"] if isinstance(x, dict)]
            elif _as_bool(data.get("is_violation", False)):
                items = [data]
            else:
                items = []

            # Issue keys already reported earlier (e.g. "P1: missing gloves" → ("P1","missing_gloves"))
            done = set()
            for entry in event.already_reported:
                who, _, iss = entry.partition(":")
                done.add((who.strip(), _norm_issue(iss)))

            out: list[Violation] = []
            seen_types: set[str] = set()
            for item in items:
                issues = [k for k in (_norm_issue(i) for i in (item.get("issues") or [])) if k]
                pids = [str(x).strip().upper() for x in (item.get("person_ids") or []) if str(x).strip()]
                owners = pids or [p.person_id for p in event.people if set(p.issues) & set(issues)] or ["scene"]
                # Safety net: drop entries whose issues were ALL reported before
                if issues and all(any((o, i) in done for o in owners) or ("scene", i) in done for i in issues):
                    logger.debug(f"Skipping already-reported {item.get('violation_type')} @ {event.timestamp_sec}s")
                    continue
                vtype = str(item.get("violation_type") or "Unknown")
                key = (vtype.lower(), tuple(sorted(owners)))
                if key in seen_types:          # same type for same people twice in one reply
                    continue
                seen_types.add(key)
                out.append(Violation(
                    violation_id       = str(uuid.uuid4()),
                    event              = event,
                    violation_type     = vtype,
                    rule_violated      = str(item.get("rule_violated") or "Unspecified"),
                    explanation        = str(item.get("explanation") or ""),
                    severity           = _norm_severity(item.get("severity"), event.severity),
                    sop_evidence       = evidence.get("sops", []),
                    similar_incidents  = evidence.get("incidents", []),
                    equipment_notes    = evidence.get("manuals", []),
                    immediate_action   = str(item.get("immediate_action") or ""),
                    recommended_action = str(item.get("recommended_action") or ""),
                    preventive_measure = str(item.get("preventive_measure") or ""),
                    issues             = issues,
                    person_ids         = [o for o in owners if o != "scene"],
                ))
            if not out:
                logger.debug(f"No new violation confirmed @ {event.timestamp_sec}s")
            return out
        except FatalLLMError:
            raise
        except Exception as e:
            # Don't silently drop an event the vision model flagged — keep it, marked unverified
            self.failed_events += 1
            logger.error(f"Reasoning failed @ {event.timestamp_sec}s: {e} — keeping as unverified")
            return [Violation(
                violation_id       = str(uuid.uuid4()),
                event              = event,
                violation_type     = "Unverified (reasoning step failed)",
                rule_violated      = "Not verified — review footage manually",
                explanation        = f"The vision model flagged: {event.description}. "
                                     "Automatic rule verification failed, so this needs manual review.",
                severity           = _norm_severity(event.severity, "medium"),
                sop_evidence       = evidence.get("sops", []),
                similar_incidents  = evidence.get("incidents", []),
                equipment_notes    = evidence.get("manuals", []),
                immediate_action   = "Review this moment of the footage.",
            )]

    def build_report(
        self,
        video_result: VideoAnalysisResult,
        violations: list[Violation],
        report_id: str | None = None,
        output_dir: str | None = None,
    ) -> InvestigationReport:
        report_id = report_id or str(uuid.uuid4())
        viol_lines = "\n".join(
            f"- [{v.severity.upper()}] {v.violation_type} @ {format_human_time(v.event.timestamp_sec)}: {v.explanation[:120]}"
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
            data = parse_llm_json(resp.text)
            if not data:
                raise ValueError("unparseable summary reply")
            summary  = str(data.get("executive_summary") or summary)
            findings = _as_str_list(data.get("key_findings"))
            actions  = _as_str_list(data.get("immediate_actions"))
            llm_risk = _norm_severity(data.get("overall_risk_level"), overall_risk)
            # Overall risk can never be lower than the worst confirmed violation
            overall_risk = max([llm_risk, overall_risk], key=lambda s: _SEVERITY_RANK.get(s, 0))
            if not violations:
                overall_risk = "low"
        except FatalLLMError:
            raise
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
                findings = [f"{v.violation_type} at {format_human_time(v.event.timestamp_sec)}" for v in violations[:5]]
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


def _fmt_people(event: DetectedEvent) -> str:
    parts = []
    for p in event.people:
        if p.issues:
            parts.append(f"{p.person_id} ({p.appearance or 'no description'}): {', '.join(p.issues)}")
    if event.scene_issues:
        parts.append("Scene: " + ", ".join(event.scene_issues))
    if event.repeat_timestamps:
        parts.append("Same issue seen again at: " + ", ".join(format_human_time(t) for t in event.repeat_timestamps))
    return "; ".join(parts) or "N/A"


def _norm_severity(value, default: str = "medium") -> str:
    v = str(value or "").strip().lower()
    if v in _SEVERITY_RANK:
        return v
    d = str(default or "").strip().lower()
    return d if d in _SEVERITY_RANK else "medium"


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _as_str_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]


def _derive_risk(violations: list[Violation]) -> str:
    if not violations:
        return "low"
    return _norm_severity(max(violations, key=lambda v: _SEVERITY_RANK.get(v.severity, 0)).severity, "medium")


def _fmt(items: list[RetrievedEvidence], n: int = 4) -> str:
    if not items:
        return "  (none retrieved)"
    lines = []
    for i, e in enumerate(items[:n], 1):
        snippet = e.chunk.content[:700].replace("\n", " ")
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
        + _UNIVERSAL_RULES_FOOTER
    )
