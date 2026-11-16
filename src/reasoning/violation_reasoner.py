"""
Violation Reasoner — Stage 4.

Takes each detected event + retrieved evidence + the original manual text
and asks GPT-4o to confirm violations, cite the exact rule, and recommend actions.

Nothing hardcoded — all rules come from the uploaded manual or universal principles.
"""

from __future__ import annotations
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from config.settings import settings, OUTPUT_DIR
from src.utils.models import (
    DetectedEvent, RetrievedEvidence, Violation,
    VideoAnalysisResult, InvestigationReport,
)
from src.utils.json_utils import parse_llm_json
from src.utils.llm_client import get_client, parallel_lanes, FatalLLMError
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

EVENTS_PER_REQUEST = max(1, int(os.getenv("LABSENTINEL_EVENTS_PER_REQUEST", "5")))

BATCH_VIOLATION_PROMPT = """\
You are a certified laboratory safety investigator.

{manual_context}

Below are {n} flagged moments from ONE incident video. Assess EACH event separately.

{events_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK — for EACH event
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. List every GENUINE safety violation in the event — one entry per distinct rule broken
   (missing PPE items together = ONE "PPE" entry; a spill is separate; food at the bench is
   separate). Only include what the evidence directly supports, for that event's lab type.
2. Ignore anything under that event's ALREADY REPORTED EARLIER — it is already in the report.
   If nothing new remains for an event, give it an empty "violations" list.
3. Cite the EXACT rule — from the manual if present, or "Universal [<lab type>]: ..."
4. Explain what went wrong and the specific danger, and give actionable recommendations.
5. "issues" = the issue keys each entry covers (prefer: {issue_keys}).

Respond ONLY with valid JSON — no prose, no markdown. One entry per event:
{{
  "events": [
    {{
      "event": "<E1|E2|…>",
      "violations": [
        {{
          "violation_type": "<PPE | Chemical Handling | Equipment Misuse | Fire Hazard | Biological Safety | Electrical Safety | Ergonomics | Waste Disposal | Emergency | Food/Drink | Other>",
          "issues": ["<issue key>"],
          "person_ids": ["<P1 …, empty for scene hazards>"],
          "rule_violated": "<exact quote from manual, or 'Universal [<lab type>]: ...'>",
          "explanation": "<2-3 sentences>",
          "severity": "<low|medium|high|critical>",
          "immediate_action": "<specific action needed RIGHT NOW>",
          "recommended_action": "<corrective action for next 24-72 hours>",
          "preventive_measure": "<systemic change to prevent recurrence>"
        }}
      ]
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
        self._lock = threading.Lock()
        self._lane_clients = {0: self.client}
        logger.info("ViolationReasoner ready")

    def _client_for(self, lane: int):
        if lane not in self._lane_clients:
            self._lane_clients[lane] = get_client(lane=lane)
        return self._lane_clients[lane]

    def analyse_events(
        self,
        events: list[DetectedEvent],
        evidence_by_event: dict[str, dict[str, list[RetrievedEvidence]]],
        manual_text: str = "",
        on_progress=None,
    ) -> list[Violation]:
        """
        Verify many events quickly: EVENTS_PER_REQUEST events per request, groups in
        parallel across usable keys. A group that fails falls back to one request per event.
        Returns violations in time order.
        """
        if not events:
            return []
        manual_ctx = _build_manual_ctx(manual_text)
        groups = [events[i:i + EVENTS_PER_REQUEST] for i in range(0, len(events), EVENTS_PER_REQUEST)]
        lanes = max(1, min(parallel_lanes(), len(groups)))
        logger.info(f"Reasoning over {len(events)} event(s) in {len(groups)} request(s), {lanes} in parallel")
        found: dict[str, list[Violation]] = {}
        done = 0
        for wave_start in range(0, len(groups), lanes):
            wave = groups[wave_start:wave_start + lanes]
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = {pool.submit(self._analyse_group, self._client_for(slot), g, evidence_by_event,
                                       manual_text, manual_ctx): g for slot, g in enumerate(wave)}
                for fut in as_completed(futures):
                    g = futures[fut]
                    found.update(fut.result())          # FatalLLMError propagates
                    done += len(g)
                    if on_progress:
                        n_v = sum(len(found.get(e.event_id, [])) for e in g)
                        on_progress(done / len(events), f"Verified {done}/{len(events)} flagged moments"
                                                        + (f" — {n_v} violation(s) confirmed" if n_v else ""))
        out: list[Violation] = []
        for e in sorted(events, key=lambda e: e.timestamp_sec):
            out.extend(found.get(e.event_id, []))
        return out

    def _analyse_group(self, client, group, evidence_by_event, manual_text, manual_ctx) -> dict[str, list[Violation]]:
        from src.vlm.video_analyzer import ISSUE_KEYS
        if len(group) == 1:
            e = group[0]
            return {e.event_id: self.analyse_event_all(e, evidence_by_event.get(e.event_id, {}), manual_text, client)}
        blocks = []
        for i, e in enumerate(group, 1):
            ev = evidence_by_event.get(e.event_id, {})
            blocks.append(
                f"━━ EVENT E{i} ━━\n"
                f"Lab type   : {_lab_type(e)}\n"
                f"Timestamp  : {e.timestamp_sec}s ({format_human_time(e.timestamp_sec)})\n"
                f"Description: {e.description}\n"
                f"People     : {_fmt_people(e)}\n"
                f"Objects    : {', '.join(o for o in e.objects_involved if not o.startswith('lab_type:')) or 'N/A'}\n"
                f"Actions    : {', '.join(e.actions_observed) or 'N/A'}\n"
                f"AI severity: {e.severity}\n"
                f"ALREADY REPORTED EARLIER (do NOT report these again): {'; '.join(e.already_reported) or 'nothing'}\n"
                f"Relevant manual / SOP passages:\n{_fmt(ev.get('sops', []), n=3, chars=500)}\n"
                f"Equipment notes:\n{_fmt(ev.get('manuals', []), n=2, chars=300)}\n"
                f"Similar past incidents:\n{_fmt(ev.get('incidents', []), n=2, chars=300)}\n")
        prompt = BATCH_VIOLATION_PROMPT.format(manual_context=manual_ctx, n=len(group),
                                               events_block="\n".join(blocks), issue_keys=", ".join(ISSUE_KEYS))
        try:
            resp = client.chat(prompt=prompt, temperature=0.15, max_tokens=900 * len(group))
            data = parse_llm_json(resp.text)
            entries = data.get("events") if isinstance(data, dict) else None
            if not isinstance(entries, list):
                raise ValueError("unparseable batch reasoning reply")
            by_label = {}
            for pos, entry in enumerate(entries):
                if isinstance(entry, dict):
                    label = str(entry.get("event") or f"E{pos + 1}").strip().upper()
                    by_label[label] = entry
            out = {}
            for i, e in enumerate(group, 1):
                entry = by_label.get(f"E{i}")
                if entry is None:       # model skipped this event → verify it on its own
                    out[e.event_id] = self.analyse_event_all(e, evidence_by_event.get(e.event_id, {}),
                                                             manual_text, client)
                    continue
                items = [x for x in (entry.get("violations") or []) if isinstance(x, dict)]
                out[e.event_id] = self._items_to_violations(e, evidence_by_event.get(e.event_id, {}), items)
            return out
        except FatalLLMError:
            raise
        except Exception as ex:
            logger.warning(f"Batch reasoning failed ({ex}) — verifying {len(group)} event(s) one by one")
            return {e.event_id: self.analyse_event_all(e, evidence_by_event.get(e.event_id, {}), manual_text, client)
                    for e in group}

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
        client=None,
    ) -> list[Violation]:
        """All distinct, NEW violations in one event (one per rule broken)."""
        from src.vlm.video_analyzer import ISSUE_KEYS
        client = client or self.client
        manual_ctx = _build_manual_ctx(manual_text)
        prompt = VIOLATION_PROMPT.format(
            manual_context     = manual_ctx,
            lab_type           = _lab_type(event),
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
            resp = client.chat(
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
            return self._items_to_violations(event, evidence, items)
        except FatalLLMError:
            raise
        except Exception as e:
            return self._unverified(event, evidence, e)

    def _items_to_violations(self, event, evidence, items) -> list[Violation]:
        """Model output entries → Violation objects (drops already-reported / duplicate entries)."""
        from src.vlm.video_analyzer import _norm_issue
        if True:
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

    def _unverified(self, event, evidence, e) -> list[Violation]:
        if True:
            # Don't silently drop an event the vision model flagged — keep it, marked unverified
            with self._lock:
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


_NO_CONSOLIDATE = {"other", "unknown", ""}


def consolidate_by_person(violations: list[Violation]) -> tuple[list[Violation], int]:
    """
    One violation per person per violation type for the whole video. A later violation
    of the same type for the same person (e.g. P5: gloves at 03:10, then goggles at 03:40)
    is folded into the first: its issues are added, the time is recorded under
    "Also seen at", and severity is the worst of the two.
    Scene hazards, unverified items and "Other" are left as they are.
    Returns (violations, number_folded).
    """
    out: list[Violation] = []
    index: dict[tuple[str, str], int] = {}
    folded = 0
    for v in sorted(violations, key=lambda x: x.event.timestamp_sec):
        vtype = v.violation_type.strip().lower()
        if not v.person_ids or vtype in _NO_CONSOLIDATE or vtype.startswith("unverified"):
            out.append(v)
            continue
        target = next((index[(vtype, p)] for p in v.person_ids if (vtype, p) in index), None)
        if target is None:
            out.append(v)
            for p in v.person_ids:
                index[(vtype, p)] = len(out) - 1
            continue

        base = out[target]
        ts = v.event.timestamp_sec
        new_issues = [i for i in v.issues if i not in base.issues]
        base.issues = base.issues + new_issues
        base.person_ids = base.person_ids + [p for p in v.person_ids if p not in base.person_ids]
        for p in v.person_ids:
            index.setdefault((vtype, p), target)
        if _SEVERITY_RANK.get(v.severity, 0) > _SEVERITY_RANK.get(base.severity, 0):
            base.severity = v.severity
        # Record this later sighting so it shows under "Also seen at"
        base.event = base.event.model_copy(deep=True)
        for p in v.person_ids:
            for i in (v.issues or ["other"]):
                times = base.event.repeat_sightings.setdefault(f"{p}:{i}", [])
                if ts not in times:
                    times.append(ts)
        if ts not in base.event.repeat_timestamps:
            base.event.repeat_timestamps.append(ts)
        if new_issues:
            base.explanation = (base.explanation.rstrip() +
                                f" Also at {format_human_time(ts)}: {', '.join(i.replace('_', ' ') for i in new_issues)}.")
        folded += 1
    return out, folded


def _lab_type(event: DetectedEvent) -> str:
    return next((o.split(":", 1)[1] for o in event.objects_involved if o.startswith("lab_type:")), "general")


def _fmt(items: list[RetrievedEvidence], n: int = 4, chars: int = 700) -> str:
    if not items:
        return "  (none retrieved)"
    lines = []
    for i, e in enumerate(items[:n], 1):
        snippet = e.chunk.content[:chars].replace("\n", " ")
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
