"""
Video Analyzer — Stage 2 of LabSentinel-RAG.

Accuracy design:
  - Structured checklist prompt: forces GPT-4o to check EVERY PPE item and hazard category
    explicitly, not just describe what it sees generally.
  - Context window: the previous frame's findings are passed in so the model can notice
    when a violation STARTS or ENDS (continuity awareness).
  - Low severity threshold: borderline cases flagged as potential violations rather than missed.
  - Two-pass synthesis: individual frame findings → cross-frame timeline summary that
    catches violations spanning multiple frames.
"""

from __future__ import annotations
import re
import uuid

from loguru import logger

from config.settings import settings
from src.utils.models import DetectedEvent, VideoAnalysisResult
from src.utils.json_utils import parse_llm_json
from src.utils.llm_client import get_client, FatalLLMError
from src.utils.time_utils import format_human_time
from .video_chunker import extract_frames_chunked, get_video_info


# ── Frame analysis prompt ─────────────────────────────────────────────────────
# Structured checklist forces the model to check every category, not just
# describe what it sees. This dramatically reduces missed violations.

FRAME_PROMPT = """\
You are a trained laboratory safety auditor reviewing CCTV footage.

{manual_context}

PREVIOUS FRAME (t={prev_ts}s): {prev_context}

Analyse this frame at timestamp {timestamp}s.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — IDENTIFY THE LAB TYPE from visual clues in the frame:
- Chemistry lab: fume hoods, chemical bottles, glassware, Bunsen burners, centrifuges, titration equipment
- Biology/Microbiology: biosafety cabinets, petri dishes, microscopes, autoclaves, biohazard symbols, pipettes
- Physics lab: lasers, optics tables, oscilloscopes, electrical apparatus, high-voltage signs
- Engineering/Workshop: machinery, lathes, welding equipment, tools, metal work
- Computer Science/IT: computers, servers, monitors, cables — no chemicals
- Medical/Clinical: blood samples, sterile packaging, clinical equipment, IV equipment
- General/Unknown: apply broad principles

STEP 2 — APPLY RULES BASED ON THE LAB TYPE IDENTIFIED:

CHEMISTRY lab rules:
□ Lab coat worn and fastened?
□ Chemical splash goggles worn (not just safety glasses) when handling liquids?
□ Gloves on both hands when near chemicals?
□ Fume hood sash at or below working height when using volatile chemicals?
□ All containers labelled with name, concentration, date?
□ No open containers left unattended?
□ No flames within 3 metres of flammable solvents?

BIOLOGY lab rules:
□ Lab coat worn and fastened?
□ Gloves on both hands?
□ Eye protection worn when handling liquids?
□ Biosafety cabinet sash at correct operating height?
□ No mouth pipetting (must use mechanical pipette aids)?
□ Sharps going directly into rigid sharps containers?
□ Biohazard waste in labelled biohazard bags?
□ Surfaces decontaminated before and after use?

PHYSICS/ENGINEERING lab rules:
□ Appropriate PPE for the equipment (safety glasses, ear defenders, hard hat)?
□ No loose clothing or jewellery near rotating machinery?
□ Machinery guards in place?
□ Laser safety eyewear worn if lasers visible?
□ No exposed electrical wiring?

COMPUTER SCIENCE/IT lab rules:
□ No food or drink near computers or servers?
□ No trip hazards from loose cables?
□ No water or liquids near electrical equipment?
□ Emergency exits clear?

ALL lab types — always check:
□ No eating, drinking, or phone use at the work bench?
□ Emergency exits and aisles clear of obstructions?
□ Appropriate closed-toe footwear worn?
□ Long hair tied back near equipment?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULES FOR FLAGGING:
- Apply ONLY the rules relevant to the lab type you identified
- Do NOT apply chemistry rules in a computer science lab
- If PPE is not visible but the activity requires it → flag as missing
- If no people are present but equipment is in a hazardous state → still flag
- potential_violation=true if ANY rule is broken

Respond ONLY with valid JSON — no prose, no markdown:
{{
  "lab_type": "<chemistry|biology|physics|engineering|computer_science|medical|general>",
  "description": "<1-2 sentence factual description>",
  "people_count": <number>,
  "objects_involved": ["<every item visible>"],
  "actions_observed": ["<every action>"],
  "ppe_status": {{
    "lab_coat": "<worn|missing|not_required|cannot_determine>",
    "gloves": "<worn|missing|not_required|cannot_determine>",
    "eye_protection": "<worn|missing|not_required|cannot_determine>",
    "other_ppe": "<any other PPE observation>"
  }},
  "checklist_findings": ["<every issue found, with specific detail>"],
  "potential_violation": <true|false>,
  "violation_hint": "<describe every issue found>",
  "severity": "<none|low|medium|high|critical>",
  "confidence": <0.0-1.0>
}}
"""

# ── Timeline synthesis prompt ─────────────────────────────────────────────────
# Cross-frame analysis: catches violations that span multiple frames and
# produces a coherent narrative rather than a list of disconnected observations.

TIMELINE_PROMPT = """\
You are a laboratory safety investigator writing an incident report.

Below are timestamped observations from a video analysis.
{frame_count} frames were analysed covering {duration}s of footage.

OBSERVATIONS:
{observations}

CROSS-FRAME ANALYSIS TASKS:
1. Identify any violation that PERSISTED across multiple frames (e.g. researcher working
   without PPE for 30 seconds — this is one ongoing violation, not many separate ones).
2. Identify any violation that appears, disappears, and reappears.
3. Write a concise 3-5 sentence incident summary covering the overall pattern.
4. Do NOT speculate beyond what was observed.

Respond ONLY with a valid JSON object — no prose, no markdown fences:
{{
  "incident_summary": "<concise paragraph>",
  "persistent_violations": [
    {{
      "description": "<what the ongoing violation is>",
      "start_time": <seconds>,
      "end_time": <seconds>
    }}
  ]
}}
"""

_NO_MANUAL_CTX = """\
LAB MANUAL: None uploaded. The AI will identify the lab type from the video and apply appropriate rules.
"""

# Universal rules appended to ALL manual contexts — these fill gaps the manual may not cover
_UNIVERSAL_RULES = """\
UNIVERSAL SAFETY BASELINES (apply these for anything the manual does not explicitly cover):

For CHEMISTRY/BIOLOGY labs:
- Lab coat, gloves, and eye protection mandatory at all times near chemicals or biological material
- No eating, drinking, or phone use at the bench
- All containers must be labelled (name, concentration, date)
- Fume hood sash at or below working height line during all chemical work
- Centrifuge samples balanced; lid closed during run
- Spills cleaned up immediately with appropriate spill kit
- Sharps in puncture-resistant containers only
- Emergency exits and aisles must remain clear

For PHYSICS/ENGINEERING labs:
- Appropriate PPE for equipment in use (safety glasses, ear protection as needed)
- No loose clothing or jewellery near rotating machinery
- Guards in place on all machinery
- Lockout/tagout before maintenance or repair
- No exposed electrical wiring or bypassed interlocks
- Keep aisles and walkways clear around machines

For COMPUTER SCIENCE/IT labs:
- No food or drink near electrical equipment
- No water or liquids near computers
- Cables managed to prevent trip hazards
- Surge-protected equipment only; no overloaded power strips
- Emergency exits clear

For MEDICAL/CLINICAL labs:
- Appropriate PPE for patient-facing or biohazard procedures
- Sharps disposed in approved puncture-resistant containers only
- Clean/disinfect equipment between patient or sample handling steps
- Waste and biohazard bins used correctly and not overfilled
- No blocked emergency access or patient pathways

For ALL labs regardless of type:
- No working alone with hazardous materials
- Appropriate closed-toe footwear
- Long hair tied back near equipment
- Never block fire exits or safety equipment access
"""


class VideoAnalyzer:
    """GPT-4o Vision video analysis with structured checklist prompts."""

    def __init__(self):
        self.client = get_client()
        from config.settings import settings
        self.model  = settings.llm.active_provider()
        logger.info(f"VideoAnalyzer ready — provider: {self.model}")

    def analyse_video(
        self,
        video_path: str,
        manual_text: str = "",
        sample_rate: int | None = None,
        max_frames: int | None = None,
        chunk_duration_sec: int | None = None,
        on_progress=None,
    ) -> VideoAnalysisResult:
        """
        Full video analysis pipeline with accuracy-first design.

        manual_text: Full extracted text of the uploaded lab manual.
                     Empty = universal safety rules used as fallback.
        """
        info     = get_video_info(video_path)
        duration = info["duration_sec"]
        logger.info(f"Video: {info['duration_hms']} | {info['size_mb']} MB | "
                    f"{info['width']}x{info['height']} @ {info['fps']} FPS")

        if on_progress:
            on_progress(0.03, f"Video loaded — {info['duration_hms']}, {info['size_mb']} MB")

        manual_ctx = _build_manual_context(manual_text)

        frames = extract_frames_chunked(
            video_path         = video_path,
            sample_rate        = sample_rate,
            max_frames         = max_frames,
            chunk_duration_sec = chunk_duration_sec,
            on_progress        = (lambda p, m: on_progress(0.03 + p * 0.27, m))
                                  if on_progress else None,
        )

        logger.info(f"Analysing {len(frames)} frames …")
        if on_progress:
            on_progress(0.30, f"Analysing {len(frames)} frames with {self.model} …")

        events: list[DetectedEvent] = []
        prev_event: DetectedEvent | None = None
        self._failed_frames = 0

        for i, frame in enumerate(frames):
            event = self._analyse_frame(frame, manual_ctx, prev_event)
            if event:
                events.append(event)
                prev_event = event
            if on_progress and frames:
                pct = 0.30 + (i + 1) / len(frames) * 0.55
                on_progress(pct,
                    f"Frame {i+1}/{len(frames)} @ {frame['timestamp_sec']:.1f}s"
                    + (" — ⚠ violation flagged" if event and event.potential_violation else ""))

        events = _collapse_adjacent_events(events)

        if on_progress:
            on_progress(0.87, "Cross-frame timeline analysis …")

        summary = self._synthesise_timeline(events, duration)

        n_viol = sum(1 for e in events if e.potential_violation)
        if self._failed_frames:
            logger.warning(f"{self._failed_frames}/{len(frames)} frames could not be analysed")
        logger.success(f"Done — {len(events)} events, {n_viol} potential violations")
        if on_progress:
            on_progress(1.0, f"Video analysis done — {n_viol} potential violations in {len(events)} events")

        return VideoAnalysisResult(
            video_path           = video_path,
            duration_sec         = duration,
            total_frames_sampled = len(frames),
            events               = events,
            summary              = summary,
            failed_frames        = self._failed_frames,
        )

    def analyse_image(self, image_path: str) -> DetectedEvent | None:
        import base64
        self._failed_frames = 0
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return self._analyse_frame(
            {"frame_index": 0, "timestamp_sec": 0.0, "frame_path": image_path, "base64": b64},
            _build_manual_context(""),
            None,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _analyse_frame(
        self,
        frame: dict,
        manual_ctx: str,
        prev_event: DetectedEvent | None,
    ) -> DetectedEvent | None:
        # Build previous-frame context so model can detect continuity
        if prev_event:
            prev_ctx = (
                f"Violation flag={prev_event.potential_violation}, "
                f"severity={prev_event.severity}. "
                f"Observed: {prev_event.description[:160]}"
            )
            prev_ts = prev_event.timestamp_sec
        else:
            prev_ctx = "First frame — no previous context."
            prev_ts  = 0.0

        prompt = FRAME_PROMPT.format(
            manual_context = manual_ctx,
            prev_ts        = prev_ts,
            prev_context   = prev_ctx,
            timestamp      = frame["timestamp_sec"],
        )

        try:
            resp = self.client.chat(
                prompt      = prompt,
                images      = [frame["base64"]],
                temperature = 0.05,
                max_tokens  = 900,
            )
            data = parse_llm_json(resp.text)
            if not data:
                self._failed_frames += 1
                logger.warning(f"Unparseable VLM reply @ {frame['timestamp_sec']}s — frame skipped")
                return None

            data = _sanitize_frame_result(data)

            # Combine checklist findings into the objects list for downstream retrieval
            lab_type   = data.get("lab_type", "general")
            objects    = data.get("objects_involved", [])
            chk        = data.get("checklist_findings", [])
            ppe_status = data.get("ppe_status", {})

            severity = str(data.get("severity", "none")).lower()
            is_violation = bool(data.get("potential_violation", False))
            missing_ppe = [k for k, v in ppe_status.items() if str(v).lower() == "missing"]

            # Only escalate severity for frames that are actually flagged
            if is_violation and missing_ppe and severity in ("none", "low"):
                severity = "medium"
            if is_violation and "lab_coat" in missing_ppe and len(missing_ppe) >= 2:
                severity = "high"
            if is_violation and severity == "none":
                severity = "low"

            return DetectedEvent(
                event_id            = str(uuid.uuid4()),
                timestamp_sec       = frame["timestamp_sec"],
                frame_index         = frame["frame_index"],
                description         = f"[{lab_type.upper()} LAB] {data.get('description', '')}",
                objects_involved    = [str(o) for o in objects] + [str(c) for c in chk] + [f"lab_type:{lab_type}"],
                actions_observed    = [str(a) for a in data.get("actions_observed", []) or []],
                potential_violation = is_violation,
                severity            = severity,
                confidence          = max(0.0, min(1.0, float(data.get("confidence", 0.5) or 0.0))),
                frame_path          = frame["frame_path"],
            )
        except FatalLLMError:
            raise   # all keys exhausted / project blocked — abort the run, don't fake a clean report
        except Exception as e:
            self._failed_frames += 1
            logger.warning(f"Frame analysis failed @ {frame['timestamp_sec']}s: {e}")
            return None

    def _synthesise_timeline(self, events: list[DetectedEvent], duration: float) -> str:
        if not events:
            return "No significant events detected."

        obs = "\n".join(
            f"[{format_human_time(e.timestamp_sec)}] {e.description} "
            f"(violation={e.potential_violation}, severity={e.severity})"
            for e in events
        )

        # Keep observations concise to avoid token limit issues
        if len(obs) > 4000:
            obs = obs[:4000] + "\n[...additional events truncated...]"

        prompt = TIMELINE_PROMPT.format(
            frame_count  = len(events),
            duration     = f"{duration:.0f}",
            observations = obs,
        )
        for attempt in range(2):
            try:
                resp = self.client.chat(
                    prompt      = prompt,
                    temperature = 0.1,
                    max_tokens  = 500,
                )
                data = parse_llm_json(resp.text)
                summary = data.get("incident_summary", "")
                if summary:
                    return summary
            except FatalLLMError:
                break
            except Exception as e:
                logger.warning(f"Timeline synthesis attempt {attempt+1} failed: {e}")

        # Fallback: build summary from events without LLM
        violations = [e for e in events if e.potential_violation]
        if violations:
            v_desc = "; ".join(
                f"{e.description[:60]} at {format_human_time(e.timestamp_sec)}"
                for e in violations[:4]
            )
            return (f"Video analysis detected {len(violations)} potential safety violation(s) "
                    f"across {len(events)} observed events covering {format_human_time(duration)} of footage. "
                    f"Key concerns: {v_desc}.")
        return (f"Video analysis completed: {len(events)} events observed across "
                f"{format_human_time(duration)} of footage. No clear violations detected.")


# ── Helpers ───────────────────────────────────────────────────────────────────

_PPE_WORDS = ("ppe", "glove", "goggle", "lab coat", "coat", "eye protection", "safety glasses",
              "face shield", "respirator", "mask", "apron", "footwear", "shoes")
_NO_ISSUE_PREFIXES = ("none", "n/a", "na", "no issue", "no issues", "no violation", "no violations",
                      "nothing", "all safety", "compliant", "fully compliant")
_UNCERTAIN_PHRASES = ("cannot determine", "can't determine", "unable to determine", "no clear evidence",
                      "too blurry", "image unclear", "not enough detail")
_MIN_CONFIDENCE = 0.55


def _mentions_ppe(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _PPE_WORDS)


def _sanitize_frame_result(data: dict) -> dict:
    """
    Guard against unsupported flags without discarding real findings.

    The model's own structured output (ppe_status, checklist_findings,
    violation_hint) is treated as the evidence. A flag is kept when there is
    concrete evidence and confidence >= 0.55. PPE findings are ignored when no
    person is in frame (nobody to wear it), and purely uncertain hints are not
    counted as evidence.
    """
    if not isinstance(data, dict):
        return {}

    cleaned = dict(data)
    try:
        people_count = int(cleaned.get("people_count", 0) or 0)
    except (TypeError, ValueError):
        people_count = 0
    try:
        confidence = float(cleaned.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    ppe_status = cleaned.get("ppe_status", {}) or {}
    if not isinstance(ppe_status, dict):
        ppe_status = {}
    missing_ppe = [k for k, v in ppe_status.items() if str(v).lower() == "missing"]

    findings = cleaned.get("checklist_findings", []) or []
    if isinstance(findings, str):
        findings = [findings]
    findings = [str(f).strip() for f in findings if str(f).strip()]
    findings = [f for f in findings if not f.lower().startswith(_NO_ISSUE_PREFIXES)]

    hint = str(cleaned.get("violation_hint", "") or "").strip()
    if hint.lower().startswith(_NO_ISSUE_PREFIXES):
        hint = ""

    if people_count == 0:
        # Nobody present → PPE can't be "missing"; keep equipment/hazard findings only
        missing_ppe = []
        findings = [f for f in findings if not _mentions_ppe(f)]
        if _mentions_ppe(hint):
            hint = ""

    def _certain(text: str) -> bool:
        return bool(text) and not any(p in text.lower() for p in _UNCERTAIN_PHRASES)

    has_evidence = bool(missing_ppe) or any(_certain(f) for f in findings) or _certain(hint)

    flagged = bool(cleaned.get("potential_violation"))
    # Model reported missing PPE on a person but forgot to set the flag
    if not flagged and missing_ppe and confidence >= _MIN_CONFIDENCE:
        flagged = True

    if flagged and (not has_evidence or confidence < _MIN_CONFIDENCE):
        flagged = False

    cleaned["potential_violation"] = flagged
    if not flagged:
        cleaned["severity"] = "none"
        cleaned["confidence"] = min(confidence, 0.4)
    return cleaned


_STOPWORDS = {"the", "a", "an", "is", "are", "and", "or", "of", "in", "on", "at", "to", "with",
              "while", "for", "by", "be", "lab", "laboratory", "worker", "researcher", "person",
              "frame", "this", "that", "their", "his", "her", "who", "near", "from"}
_SEV_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _tokens(text: str) -> set[str]:
    text = re.sub(r"^\[[^\]]*\]\s*", "", text.lower())       # drop "[CHEMISTRY LAB]" prefix
    return {w for w in re.findall(r"[a-z0-9]+", text) if w not in _STOPWORDS and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _same_issue(a: DetectedEvent, b: DetectedEvent) -> bool:
    """True when two flagged events describe the same violation (not merely the same lab)."""
    if a.description.lower() == b.description.lower():
        return True
    if _jaccard(_tokens(a.description), _tokens(b.description)) >= 0.5:
        return True
    objs = lambda e: {str(x).lower() for x in e.objects_involved if not str(x).startswith("lab_type:")}
    return _jaccard(objs(a), objs(b)) >= 0.6


def _collapse_adjacent_events(events: list[DetectedEvent], max_gap: float = 8.0) -> list[DetectedEvent]:
    """
    Merge consecutive frames showing the SAME ongoing violation into one event.
    Different violations close together in time are kept separate. The gap is
    measured from the last frame where the violation was seen, so an issue that
    persists for a long time stays one event.
    """
    if not events:
        return []

    ordered = sorted(events, key=lambda e: (e.timestamp_sec, e.frame_index))
    collapsed: list[DetectedEvent] = []
    last_seen: dict[int, float] = {}     # index in collapsed → last timestamp observed

    for event in ordered:
        if not event.potential_violation:
            if event.severity in ("none", "low", "unknown"):
                continue
            collapsed.append(event)
            continue

        merged_into = None
        for idx in range(len(collapsed) - 1, -1, -1):
            prev = collapsed[idx]
            if not prev.potential_violation:
                continue
            if event.timestamp_sec - last_seen.get(idx, prev.timestamp_sec) > max_gap:
                continue
            if _same_issue(prev, event):
                merged_into = idx
                break

        if merged_into is None:
            collapsed.append(event)
            last_seen[len(collapsed) - 1] = event.timestamp_sec
            continue

        last = collapsed[merged_into]
        merged = last.model_copy()
        merged.description = last.description if len(last.description) >= len(event.description) else event.description
        merged.confidence = max(last.confidence, event.confidence)
        merged.severity = max([last.severity, event.severity], key=lambda s: _SEV_RANK.get(s, 0))
        merged.objects_involved = list(dict.fromkeys(last.objects_involved + event.objects_involved))
        collapsed[merged_into] = merged
        last_seen[merged_into] = event.timestamp_sec

    return collapsed


def _build_manual_context(manual_text: str) -> str:
    """
    Build the context block injected into every frame prompt.
    - If a manual is provided: use it as the primary authority
    - Universal rules are ALWAYS appended so basic violations are never missed
      even when the manual doesn't explicitly cover them
    """
    if not manual_text or not manual_text.strip():
        return _NO_MANUAL_CTX + _UNIVERSAL_RULES

    text = manual_text.strip()
    if len(text) > 9_000:
        cut  = text.rfind("\n", 0, 9_000)
        text = text[:cut] + "\n[... manual truncated — additional sections available via retrieval ...]"

    return (
        "LAB MANUAL (uploaded by user — treat as primary authority):\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{text}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "ADDITIONAL UNIVERSAL RULES (apply these for anything not covered by the manual above):\n"
        + _UNIVERSAL_RULES
    )


def _parse_json(raw: str) -> dict:
    """Alias kept for any external references."""
    return parse_llm_json(raw)
