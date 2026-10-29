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
import uuid

from loguru import logger

from config.settings import settings
from src.utils.models import DetectedEvent, VideoAnalysisResult
from src.utils.json_utils import parse_llm_json
from src.utils.llm_client import get_client
from .video_chunker import extract_frames_chunked, get_video_info


# ── Frame analysis prompt ─────────────────────────────────────────────────────
# Structured checklist forces the model to check every category, not just
# describe what it sees. This dramatically reduces missed violations.

FRAME_PROMPT = """\
You are a trained laboratory safety auditor reviewing CCTV footage.
Your job is to catch EVERY safety issue — missing nothing is more important than avoiding false positives.

{manual_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREVIOUS FRAME CONTEXT (t={prev_ts}s):
{prev_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now analyse this frame at timestamp {timestamp}s.

Work through this checklist SYSTEMATICALLY for every person visible:

PEOPLE & PPE:
□ Lab coat — worn and fastened? (not just nearby, actually being WORN)
□ Gloves — worn on BOTH hands?
□ Eye protection — safety glasses or splash goggles actually ON the face?
□ Footwear — closed-toe shoes visible?
□ Hair — tied back if long?
□ Any food, drink, or phone visible at the bench?

CHEMICAL HANDLING:
□ Are containers labelled?
□ Are containers capped/closed when not in immediate use?
□ Is acid/base/flammable work happening OUTSIDE a fume hood?
□ Are incompatible chemicals stored or used together?

EQUIPMENT:
□ Fume hood: is the sash at the working height line or raised above it?
□ Centrifuge: lid open during spin? Unbalanced load visible?
□ Hotplate: unattended heating visible?
□ Any cracked, broken, or damaged glassware in use?

ENVIRONMENT:
□ Are exits or aisles blocked?
□ Any spill not being contained?
□ Any fire/explosion hazard visible?

For each issue found, state the EXACT timestamp, the EXACT observation, and which rule is violated.
If something cannot be determined from the image, state "cannot determine".
Flag as potential_violation=true even if you are only 40% confident — it is better to flag and let
the reasoning stage confirm than to miss a real violation.

Respond ONLY with a valid JSON object — no prose, no markdown fences:
{{
  "description": "<factual 1-2 sentence description of the overall scene>",
  "people_count": <integer number of people visible>,
  "objects_involved": ["<every piece of equipment, chemical, PPE item visible>"],
  "actions_observed": ["<every action or behaviour by each person>"],
  "ppe_status": {{
    "lab_coat": "<worn|missing|cannot_determine>",
    "gloves": "<worn|missing|cannot_determine>",
    "eye_protection": "<worn|missing|cannot_determine>",
    "closed_toe_shoes": "<worn|missing|cannot_determine>"
  }},
  "checklist_findings": ["<specific finding 1>", "<specific finding 2>", ...],
  "potential_violation": <true|false>,
  "violation_hint": "<describe every suspected violation, or empty string>",
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
LAB MANUAL: None uploaded.
Apply universal laboratory safety principles:
- PPE mandatory: lab coat, gloves, eye protection when handling chemicals or biological material
- No eating, drinking, or phone use in the laboratory
- All chemical containers must be labelled and capped when not in use
- Volatile chemicals must be handled in a fume hood with sash at working height
- Centrifuge loads must be balanced before running
- No unattended heating equipment
- Spills must be contained immediately with appropriate spill kit
- Emergency exits and aisles must remain clear at all times
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

        if on_progress:
            on_progress(0.87, "Cross-frame timeline analysis …")

        summary = self._synthesise_timeline(events, duration)

        n_viol = sum(1 for e in events if e.potential_violation)
        logger.success(f"Done — {len(events)} events, {n_viol} potential violations")
        if on_progress:
            on_progress(1.0, f"Video analysis done — {n_viol} potential violations in {len(events)} events")

        return VideoAnalysisResult(
            video_path           = video_path,
            duration_sec         = duration,
            total_frames_sampled = len(frames),
            events               = events,
            summary              = summary,
        )

    def analyse_image(self, image_path: str) -> DetectedEvent | None:
        import base64
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
                f"PPE observed: {prev_event.description[:120]}"
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
                return None

            # Combine checklist findings into the objects list for downstream retrieval
            objects    = data.get("objects_involved", [])
            chk        = data.get("checklist_findings", [])
            ppe_status = data.get("ppe_status", {})

            # Auto-upgrade severity if PPE is clearly missing
            severity = data.get("severity", "none")
            if (ppe_status.get("lab_coat") == "missing" or
                ppe_status.get("gloves")    == "missing" or
                ppe_status.get("eye_protection") == "missing"):
                if severity in ("none", "low"):
                    severity = "medium"

            return DetectedEvent(
                event_id            = str(uuid.uuid4()),
                timestamp_sec       = frame["timestamp_sec"],
                frame_index         = frame["frame_index"],
                description         = data.get("description", ""),
                objects_involved    = objects + chk,  # checklist findings feed into retrieval
                actions_observed    = data.get("actions_observed", []),
                potential_violation = bool(data.get("potential_violation", False)),
                severity            = severity,
                confidence          = float(data.get("confidence", 0.5)),
                frame_path          = frame["frame_path"],
            )
        except Exception as e:
            logger.warning(f"Frame analysis failed @ {frame['timestamp_sec']}s: {e}")
            return None

    def _synthesise_timeline(self, events: list[DetectedEvent], duration: float) -> str:
        if not events:
            return "No significant events detected."

        obs = "\n".join(
            f"[{e.timestamp_sec:.1f}s] {e.description} "
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
            except Exception as e:
                logger.warning(f"Timeline synthesis attempt {attempt+1} failed: {e}")

        # Fallback: build summary from events without LLM
        violations = [e for e in events if e.potential_violation]
        if violations:
            v_desc = "; ".join(f"{e.description[:60]} at {e.timestamp_sec:.0f}s" for e in violations[:4])
            return (f"Video analysis detected {len(violations)} potential safety violation(s) "
                    f"across {len(events)} observed events covering {duration:.0f}s of footage. "
                    f"Key concerns: {v_desc}.")
        return (f"Video analysis completed: {len(events)} events observed across "
                f"{duration:.0f}s of footage. No clear violations detected.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_manual_context(manual_text: str) -> str:
    """
    Build the manual context block.
    Truncates at ~10,000 chars to stay within token limits while maximising coverage.
    Uses a higher limit than before so more of the manual fits.
    """
    if not manual_text or not manual_text.strip():
        return _NO_MANUAL_CTX

    text = manual_text.strip()
    if len(text) > 10_000:
        cut  = text.rfind("\n", 0, 10_000)  # cut at newline, not mid-word
        text = text[:cut] + "\n[... manual continues — additional sections available via retrieval ...]"

    return (
        "LAB MANUAL (uploaded by user — treat as authoritative over any general knowledge):\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{text}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Also apply universal lab safety principles for anything not covered above.\n"
    )


def _parse_json(raw: str) -> dict:
    """Alias kept for any external references."""
    return parse_llm_json(raw)
