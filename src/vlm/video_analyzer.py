"""
Video Analyzer — Stage 2 of LabSentinel-RAG.

Accuracy design:
  - Structured checklist prompt: the vision model checks EVERY PPE item and hazard
    category explicitly, not just a general description.
  - Consecutive frames are sent together (batches), so the model sees continuity
    (gloves removed between frames, a spill appearing) and keeps person IDs consistent.
  - People are tracked across the whole video: later batches are told who was already
    seen, and a final reconciliation step merges IDs across parallel batches.

Speed design:
  - Several frames per request (LABSENTINEL_FRAMES_PER_REQUEST, default 6).
  - Batches run in parallel across usable API keys (LABSENTINEL_PARALLEL_REQUESTS).
  - If a batch reply is incomplete, the missing frames fall back to one-frame requests.
"""

from __future__ import annotations
import os
import re
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from config.settings import settings
from src.utils.models import DetectedEvent, VideoAnalysisResult, PersonObservation
from src.utils.json_utils import parse_llm_json
from src.utils.llm_client import get_client, parallel_lanes, FatalLLMError
from src.utils.time_utils import format_human_time
from .video_chunker import extract_frames_chunked, get_video_info


# ── Frame analysis prompt ─────────────────────────────────────────────────────
# Structured checklist forces the model to check every category, not just
# describe what it sees. This dramatically reduces missed violations.

FRAME_PROMPT = """\
You are a trained laboratory safety auditor reviewing CCTV footage.

{manual_context}

PREVIOUS FRAME (t={prev_ts}s): {prev_context}

KNOWN PEOPLE (seen in earlier frames): {known_people}

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

PEOPLE TRACKING:
- List every visible person in "people". If a person matches one of the KNOWN PEOPLE
  (same clothing, hair, build, position), REUSE that person_id. Otherwise give the next
  new id (P1, P2, P3 …). Never reuse an id for a clearly different person.
- "appearance" = short visual identifier only (clothing colours, hair, accessories, where
  they stand). Do not guess names, gender, age or identity — never use words like man,
  woman, boy, girl, he or she; say "person" instead.
- "issues" = this person's problems. Use these keys where one fits:
  {issue_keys}
  If none fits, write a short new snake_case key (e.g. "missing_hard_hat"). Use the SAME
  key every time for the same kind of problem.
- Hazards not tied to a person (spill, blocked exit, unlabelled bottle …) go in "scene_issues"
  using the same keys.

Respond ONLY with valid JSON — no prose, no markdown:
{{
  "lab_type": "<chemistry|biology|physics|engineering|computer_science|medical|general>",
  "description": "<one factual sentence>",
  "people_count": <number>,
  "objects_involved": ["<up to 8 safety-relevant items>"],
  "actions_observed": ["<up to 4 actions>"],
  "ppe_status": {{
    "lab_coat": "<worn|missing|not_required|cannot_determine>",
    "gloves": "<worn|missing|not_required|cannot_determine>",
    "eye_protection": "<worn|missing|not_required|cannot_determine>",
    "other_ppe": "<any other PPE observation>"
  }},
  "people": [
    {{"person_id": "<P1|P2|…>", "appearance": "<short visual identifier>", "issues": ["<issue key>"]}}
  ],
  "scene_issues": ["<issue key>"],
  "checklist_findings": ["<one short line per issue found; [] if none>"],
  "potential_violation": <true|false>,
  "violation_hint": "<one short line summarising the issues, or \"\" if none>",
  "severity": "<none|low|medium|high|critical>",
  "confidence": <0.0-1.0>
}}
"""

# ── Batch prompt: same checklist, several consecutive frames per request ──────
_RESPOND = "Respond ONLY with valid JSON — no prose, no markdown:\n"
_CHECK_START = FRAME_PROMPT.index("━━━━", FRAME_PROMPT.index("Analyse this frame"))
_CHECKLIST = FRAME_PROMPT[_CHECK_START:FRAME_PROMPT.index(_RESPOND)]
_FRAME_SCHEMA = FRAME_PROMPT[FRAME_PROMPT.index(_RESPOND) + len(_RESPOND):].strip()
assert _FRAME_SCHEMA.startswith("{{")

BATCH_PROMPT = (
    "You are a trained laboratory safety auditor reviewing CCTV footage.\n\n"
    "{manual_context}\n\n"
    "You are given {n} consecutive frames from the same video, in time order:\n"
    "{image_list}\n"
    "(Image 1 is the first attached image, Image 2 the second, and so on.)\n\n"
    "KNOWN PEOPLE (seen in earlier parts of the video): {known_people}\n\n"
    "Analyse EACH image separately with the checklist below. Use the sequence for continuity\n"
    "(e.g. gloves removed between images) and keep each person's ID the same across these images.\n"
    "Be thorough in checking but concise in writing: short phrases, no repetition.\n\n"
    + _CHECKLIST
    + "Respond ONLY with valid JSON — no prose, no markdown. Return exactly one entry per image, in order:\n"
    + "{{\n  \"frames\": [\n"
    + _FRAME_SCHEMA.replace("{{", "{{\n  \"image\": <image number, 1-based>,", 1)
    + "\n  ]\n}}\n"
)

RECONCILE_PROMPT = """\
People were detected in different parts of ONE CCTV video. The same person may have been
given different IDs in different parts. Group the IDs that refer to the SAME individual,
judging by clothing, hair, accessories, build and position.

IDs that appear together in the same frame are DIFFERENT people: {cooccur}

PEOPLE:
{people}

Respond ONLY with JSON — every ID exactly once:
{{"groups": [["<id>", "<id>"], ["<id>"]]}}
"""

FRAMES_PER_REQUEST = max(1, int(os.getenv("LABSENTINEL_FRAMES_PER_REQUEST", "6")))

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

# Fixed vocabulary so the same problem gets the same key in every frame (enables de-duplication)
ISSUE_KEYS = (
    "missing_lab_coat", "missing_gloves", "missing_eye_protection", "missing_other_ppe",
    "eating_drinking", "phone_use", "unsafe_chemical_handling", "fume_hood_misuse",
    "equipment_misuse", "loose_hair_or_clothing", "improper_footwear", "working_alone",
    "spill", "unlabelled_container", "open_container_unattended", "blocked_exit",
    "fire_hazard", "electrical_hazard", "improper_waste_disposal", "other",
)
_PPE_ISSUE = {"lab_coat": "missing_lab_coat", "gloves": "missing_gloves",
              "eye_protection": "missing_eye_protection"}
# Issues that are about what a person is doing/wearing (never a scene hazard)
_PERSON_ISSUES = {"missing_lab_coat", "missing_gloves", "missing_eye_protection", "missing_other_ppe",
                  "eating_drinking", "phone_use", "unsafe_chemical_handling", "loose_hair_or_clothing",
                  "improper_footwear", "working_alone"}
# Hazards that belong to the scene/object, whoever happens to be standing nearby
_SCENE_ONLY_ISSUES = {"spill", "unlabelled_container", "open_container_unattended", "blocked_exit",
                      "fire_hazard", "electrical_hazard", "improper_waste_disposal"}
_MAX_KNOWN_PEOPLE = 12
# A scene hazard (e.g. the same spill) seen again within this window is the same hazard
SCENE_REPEAT_WINDOW_SEC = 120.0

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
    """Vision-model video analysis: batched, parallel, with people tracking."""

    def __init__(self):
        self.client = get_client()
        from config.settings import settings
        self.model  = settings.llm.active_provider()
        self._lane_clients = {0: self.client}
        self._failed_frames = 0
        self._known_people: dict[str, str] = {}
        logger.info(f"VideoAnalyzer ready — provider: {self.model}")

    def _client_for(self, lane: int):
        if lane not in self._lane_clients:
            self._lane_clients[lane] = get_client(lane=lane)
        return self._lane_clients[lane]

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
        Full video analysis.

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
            on_progress        = (lambda p, m: on_progress(0.03 + p * 0.17, m))
                                  if on_progress else None,
        )

        batches = [frames[i:i + FRAMES_PER_REQUEST] for i in range(0, len(frames), FRAMES_PER_REQUEST)]
        lanes = max(1, min(parallel_lanes(), len(batches) or 1))
        logger.info(f"Analysing {len(frames)} frames in {len(batches)} request(s), {lanes} in parallel")
        if on_progress:
            on_progress(0.20, f"Analysing {len(frames)} frames — {len(batches)} request(s), "
                              f"{lanes} at a time …")

        self._failed_frames = 0
        known: dict[str, str] = {}          # person_id → latest appearance (from earlier waves)
        results: dict[int, list] = {}
        done = 0

        for wave_start in range(0, len(batches), lanes):
            wave = list(range(wave_start, min(wave_start + lanes, len(batches))))
            snapshot = dict(known)
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = {
                    pool.submit(self._analyse_batch, self._client_for(slot), batches[bi],
                                manual_ctx, snapshot, 1000 * (bi + 1)): bi
                    for slot, bi in enumerate(wave)
                }
                for fut in as_completed(futures):
                    bi = futures[fut]
                    evs, n_failed = fut.result()        # FatalLLMError propagates and aborts the run
                    results[bi] = evs
                    self._failed_frames += n_failed
                    done += len(batches[bi])
                    if on_progress:
                        flagged = sum(1 for e in evs if e and e.potential_violation)
                        on_progress(0.20 + done / max(len(frames), 1) * 0.70,
                                    f"Analysed {done}/{len(frames)} frames"
                                    + (f" — ⚠ {flagged} flagged in "
                                       f"{format_human_time(batches[bi][0]['timestamp_sec'])}–"
                                       f"{format_human_time(batches[bi][-1]['timestamp_sec'])}" if flagged else ""))
            # People seen in this wave become KNOWN PEOPLE for the next wave
            for bi in wave:
                for ev in results.get(bi, []):
                    for person in (ev.people if ev else []):
                        if _trackable(person.person_id) and person.appearance:
                            known.pop(person.person_id, None)
                            known[person.person_id] = person.appearance
            while len(known) > _MAX_KNOWN_PEOPLE:
                known.pop(next(iter(known)))

        events = [ev for bi in sorted(results) for ev in results[bi] if ev]
        if on_progress:
            on_progress(0.92, "Matching people across the video …")
        events = self._reconcile_people(events, multi_batch=len(batches) > 1)

        # Merge window follows the real frame spacing (not a fixed number of seconds)
        ts = sorted(f["timestamp_sec"] for f in frames)
        spacing = statistics.median([b - a for a, b in zip(ts, ts[1:]) if b > a] or [6.0])
        events = _collapse_adjacent_events(events, max_gap=max(8.0, 1.6 * spacing))
        events, repeats_merged = _merge_repeat_sightings(events)
        if repeats_merged:
            logger.info(f"Merged {repeats_merged} repeat sighting(s) of already-flagged person/issue pairs")

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
            repeats_merged       = repeats_merged,
        )

    def analyse_image(self, image_path: str) -> DetectedEvent | None:
        import base64
        self._failed_frames = 0
        self._known_people = {}
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return self._analyse_frame(
            {"frame_index": 0, "timestamp_sec": 0.0, "frame_path": image_path, "base64": b64},
            _build_manual_context(""),
            None,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _analyse_batch(self, client, frames: list[dict], manual_ctx: str,
                       known: dict[str, str], temp_base: int):
        """
        Analyse consecutive frames in ONE request. Returns (events, n_failed).
        Frames missing from the reply are retried one by one. New person IDs are
        namespaced per batch (P1001…) and renumbered after reconciliation.
        """
        entries = None
        if len(frames) > 1:
            image_list = "\n".join(f"  Image {i + 1} → t={f['timestamp_sec']:.2f}s "
                                   f"({format_human_time(f['timestamp_sec'])})"
                                   for i, f in enumerate(frames))
            prompt = BATCH_PROMPT.format(
                manual_context = manual_ctx,
                n              = len(frames),
                image_list     = image_list,
                known_people   = _known_txt(known),
                issue_keys     = ", ".join(ISSUE_KEYS),
            )
            try:
                resp = client.chat(prompt=prompt, images=[f["base64"] for f in frames],
                                   temperature=0.05, max_tokens=450 * len(frames))
                data = parse_llm_json(resp.text)
                entries = data.get("frames") if isinstance(data, dict) else None
                if not isinstance(entries, list):
                    logger.warning(f"Batch reply unparseable ({len(frames)} frames from "
                                   f"{frames[0]['timestamp_sec']:.0f}s) — retrying frames one by one")
                    entries = None
            except FatalLLMError:
                raise
            except Exception as e:
                logger.warning(f"Batch request failed ({e}) — retrying frames one by one")

        by_index = {}
        for pos, entry in enumerate(entries or []):
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("image", pos + 1)) - 1
            except (TypeError, ValueError):
                idx = pos
            if 0 <= idx < len(frames) and idx not in by_index:
                by_index[idx] = entry

        out, failed, id_map = [], 0, {}
        prev_event = None
        for i, frame in enumerate(frames):
            if i in by_index:
                try:
                    ev = _frame_data_to_event(frame, by_index[i], known)
                except Exception as e:
                    logger.warning(f"Could not read result for frame @ {frame['timestamp_sec']}s: {e}")
                    ev = None
            else:
                ev = self._analyse_frame_with(client, frame, manual_ctx, prev_event, known)
            if ev is None:
                failed += 1
            else:
                for person in ev.people:
                    if _trackable(person.person_id) and person.person_id not in known:
                        person.person_id = id_map.setdefault(person.person_id, f"P{temp_base + len(id_map) + 1}")
                prev_event = ev
            out.append(ev)
        return out, failed

    def _reconcile_people(self, events: list[DetectedEvent], multi_batch: bool) -> list[DetectedEvent]:
        """Give every individual ONE id for the whole video (P1, P2 … by first appearance)."""
        info: dict[str, dict] = {}
        cooccur: set[frozenset] = set()
        for ev in sorted(events, key=lambda e: e.timestamp_sec):
            pids = [p.person_id for p in ev.people if _trackable(p.person_id)]
            for p in ev.people:
                if not _trackable(p.person_id):
                    continue
                d = info.setdefault(p.person_id, {"first": ev.timestamp_sec, "app": ""})
                if len(p.appearance) > len(d["app"]):
                    d["app"] = p.appearance
            for a in pids:
                for b in pids:
                    if a < b:
                        cooccur.add(frozenset((a, b)))
        ids = sorted(info, key=lambda k: info[k]["first"])
        if not ids:
            return events

        groups = None
        if multi_batch and len(ids) > 1:
            groups = self._llm_group_people(ids, info, cooccur)
        if groups is None:
            groups = _heuristic_group_people(ids, info, cooccur) if multi_batch else [[i] for i in ids]

        # Never merge two IDs seen in the same frame; every id exactly once
        clean, seen = [], set()
        for g in groups:
            bucket: list[str] = []
            for pid in g:
                if pid in info and pid not in seen and not any(frozenset((pid, o)) in cooccur for o in bucket):
                    bucket.append(pid); seen.add(pid)
                elif pid in info and pid not in seen:
                    clean.append([pid]); seen.add(pid)
            if bucket:
                clean.append(bucket)
        clean += [[pid] for pid in ids if pid not in seen]
        clean.sort(key=lambda g: min(info[p]["first"] for p in g))
        mapping = {pid: f"P{n}" for n, g in enumerate(clean, 1) for pid in g}
        if len(mapping) != len(set(mapping.values())):
            logger.info(f"People reconciled: {len(ids)} IDs → {len(clean)} individual(s)")

        for ev in events:
            renamed = []
            for p in ev.people:
                q = p.model_copy(deep=True)
                q.person_id = mapping.get(p.person_id, p.person_id)
                renamed.append(q)
            ev.people = _union_people([], renamed)
        return events

    def _llm_group_people(self, ids, info, cooccur):
        """One short text request that groups IDs which are the same person."""
        people = "\n".join(f"- {pid} (first seen {format_human_time(info[pid]['first'])}): "
                           f"{info[pid]['app'] or 'no description'}" for pid in ids)
        co = ", ".join("/".join(sorted(p)) for p in sorted(cooccur, key=sorted)) or "none"
        try:
            resp = self.client.chat(prompt=RECONCILE_PROMPT.format(people=people, cooccur=co),
                                    temperature=0.0, max_tokens=300 + 20 * len(ids))
            data = parse_llm_json(resp.text)
            groups = data.get("groups") if isinstance(data, dict) else None
            if isinstance(groups, list) and all(isinstance(g, list) for g in groups):
                return [[str(x).strip().upper() for x in g] for g in groups]
        except FatalLLMError:
            pass
        except Exception as e:
            logger.warning(f"People reconciliation request failed ({e}) — using appearance matching")
        return None

    def _analyse_frame(
        self,
        frame: dict,
        manual_ctx: str,
        prev_event: DetectedEvent | None,
    ) -> DetectedEvent | None:
        ev = self._analyse_frame_with(self.client, frame, manual_ctx, prev_event,
                                      getattr(self, "_known_people", {}) or {})
        if ev is None:
            self._failed_frames += 1
        return ev

    def _analyse_frame_with(self, client, frame: dict, manual_ctx: str,
                            prev_event: DetectedEvent | None, known: dict[str, str]) -> DetectedEvent | None:
        """One-frame request (used for single images and as the batch fallback)."""
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
            known_people   = _known_txt(known),
            issue_keys     = ", ".join(ISSUE_KEYS),
            timestamp      = frame["timestamp_sec"],
        )
        try:
            resp = client.chat(prompt=prompt, images=[frame["base64"]], temperature=0.05, max_tokens=900)
            data = parse_llm_json(resp.text)
            if not data:
                logger.warning(f"Unparseable VLM reply @ {frame['timestamp_sec']}s — frame skipped")
                return None
            return _frame_data_to_event(frame, data, known)
        except FatalLLMError:
            raise   # all keys exhausted / project blocked — abort the run, don't fake a clean report
        except Exception as e:
            logger.warning(f"Frame analysis failed @ {frame['timestamp_sec']}s: {e}")
            return None

    def _synthesise_timeline(self, events: list[DetectedEvent], duration: float) -> str:
        """Plain summary built locally — the report's executive summary (one LLM call) replaces it."""
        if not events:
            return "No significant events detected."
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

    people = cleaned.get("people") or []
    if not isinstance(people, list):
        people = []
    if people_count == 0:
        people = []
        cleaned["people"] = []
    person_issues = [i for pp in people if isinstance(pp, dict)
                     for i in (pp.get("issues") or []) if str(i).strip() and str(i).lower() != "none"]
    scene_issues = [i for i in (cleaned.get("scene_issues") or []) if str(i).strip() and str(i).lower() != "none"]

    if people_count == 0:
        # Nobody present → PPE can't be "missing"; keep equipment/hazard findings only
        missing_ppe = []
        findings = [f for f in findings if not _mentions_ppe(f)]
        if _mentions_ppe(hint):
            hint = ""

    def _certain(text: str) -> bool:
        return bool(text) and not any(p in text.lower() for p in _UNCERTAIN_PHRASES)

    has_evidence = (bool(missing_ppe) or bool(person_issues) or bool(scene_issues)
                    or any(_certain(f) for f in findings) or _certain(hint))

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


def _issue_keys(e: DetectedEvent) -> set[tuple[str, str]]:
    keys = {(p.person_id, i) for p in e.people if _trackable(p.person_id) for i in p.issues if i != "other"}
    keys |= {("scene", i) for i in e.scene_issues if i != "other"}
    return keys


def _same_issue(a: DetectedEvent, b: DetectedEvent) -> bool:
    """True when two flagged events describe the same violation (not merely the same lab)."""
    ka, kb = _issue_keys(a), _issue_keys(b)
    if ka and kb:
        # Continuation of the same situation only if the later frame adds nothing new;
        # a frame with a NEW issue stays separate so that issue gets its own violation
        return kb <= ka
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
        merged.people = _union_people(last.people, event.people)
        merged.scene_issues = list(dict.fromkeys(last.scene_issues + event.scene_issues))
        collapsed[merged_into] = merged
        last_seen[merged_into] = event.timestamp_sec

    return collapsed


_PID_RE = re.compile(r"^P\d{1,5}$")


def _known_txt(known: dict[str, str]) -> str:
    return "; ".join(f"{pid}: {app[:90]}" for pid, app in known.items()) or "none yet"


def _frame_data_to_event(frame: dict, data: dict, known: dict[str, str]) -> DetectedEvent | None:
    """Turn one frame's JSON result into a DetectedEvent (shared by single and batch paths)."""
    if not isinstance(data, dict) or not data:
        return None
    data = _sanitize_frame_result(data)
    lab_type   = str(data.get("lab_type", "general") or "general")
    objects    = data.get("objects_involved", []) or []
    chk        = data.get("checklist_findings", []) or []
    ppe_status = data.get("ppe_status", {}) or {}
    if not isinstance(ppe_status, dict):
        ppe_status = {}

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

    people, scene_issues = _parse_people(data, missing_ppe, known)
    try:
        confidence = float(data.get("confidence", 0.5) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.5

    return DetectedEvent(
        event_id            = str(uuid.uuid4()),
        timestamp_sec       = frame["timestamp_sec"],
        frame_index         = frame["frame_index"],
        description         = f"[{lab_type.upper()} LAB] {data.get('description', '')}",
        objects_involved    = [str(o) for o in objects] + [str(c) for c in chk] + [f"lab_type:{lab_type}"],
        actions_observed    = [str(a) for a in data.get("actions_observed", []) or []],
        potential_violation = is_violation,
        severity            = severity,
        confidence          = max(0.0, min(1.0, confidence)),
        frame_path          = frame["frame_path"],
        people              = people,
        scene_issues        = scene_issues if is_violation else [],
    )


_APPEARANCE_STOP = {"the", "and", "with", "wearing", "person", "worker", "researcher", "near", "standing",
                    "left", "right", "side", "bench", "lab", "a", "an", "of", "in", "on", "at"}


def _heuristic_group_people(ids, info, cooccur) -> list[list[str]]:
    """Fallback identity matching: appearance-word overlap, never merging people seen together."""
    toks = {pid: {w for w in re.findall(r"[a-z]+", info[pid]["app"].lower())
                  if len(w) > 2 and w not in _APPEARANCE_STOP} for pid in ids}
    groups: list[list[str]] = []
    for pid in ids:
        best, best_score = None, 0.0
        for g in groups:
            if any(frozenset((pid, o)) in cooccur for o in g):
                continue
            rep_t = set().union(*(toks[o] for o in g))
            score = len(toks[pid] & rep_t) / len(toks[pid] | rep_t) if toks[pid] and rep_t else 0.0
            if score > best_score:
                best, best_score = g, score
        if best is not None and best_score >= 0.5:
            best.append(pid)
        else:
            groups.append([pid])
    return groups


def _trackable(person_id: str) -> bool:
    return bool(_PID_RE.match(str(person_id or "")))


def _norm_issue(value) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not key or key in ("none", "n_a", "na", "no_issue", "no_issues"):
        return ""
    if key in ISSUE_KEYS:
        return key
    # Map clear synonyms onto the standard keys so the same problem always matches.
    # PPE synonyms only count when the key says the item is absent ("no_goggles"), so e.g.
    # "eyewash_station_blocked" is NOT mistaken for missing eye protection.
    absent = any(w in key for w in ("missing", "no_", "without", "not_wearing", "lack", "absent", "removed"))
    if absent:
        for frags, mapped in ((("goggle", "eye_protect", "safety_glass", "eyewear", "face_shield"), "missing_eye_protection"),
                              (("glove",), "missing_gloves"),
                              (("lab_coat", "labcoat", "coat", "gown"), "missing_lab_coat")):
            if any(f in key for f in frags):
                return mapped
    for frags, mapped in ((("eating", "drinking", "food", "beverage"), "eating_drinking"),
                          (("phone",), "phone_use"),
                          (("spill",), "spill"),
                          (("unlabel", "no_label", "not_labelled", "not_labeled"), "unlabelled_container"),
                          (("sash",), "fume_hood_misuse")):
        if any(f in key for f in frags):
            return mapped
    return key[:40]


def _parse_people(data: dict, missing_ppe: list[str], known: dict[str, str], next_start: int | None = None):
    """Normalise the model's people / scene_issues output into PersonObservation objects."""
    people: list[PersonObservation] = []
    used = set()
    next_n = next_start or (max([int(k[1:]) for k in known if _trackable(k)] + [0]) + 1)
    raw = data.get("people") or []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("person_id") or "").strip().upper()
            if not _trackable(pid) or pid in used:
                pid = f"P{next_n}"
                next_n += 1
            elif pid not in known:
                next_n = max(next_n, int(pid[1:]) + 1)
            used.add(pid)
            issues = list(dict.fromkeys(filter(None, (_norm_issue(i) for i in (item.get("issues") or [])))))
            people.append(PersonObservation(person_id=pid, appearance=str(item.get("appearance") or "")[:160],
                                            issues=issues))

    # Model reported missing PPE but no per-person issues: attribute to the only person present
    ppe_issues = [_PPE_ISSUE[k] for k in missing_ppe if k in _PPE_ISSUE]
    if ppe_issues and not any(p.issues for p in people):
        if len(people) == 1:
            people[0].issues = list(dict.fromkeys(people[0].issues + ppe_issues))
        elif not people and int(data.get("people_count", 0) or 0) == 1:
            people.append(PersonObservation(person_id="P?", issues=ppe_issues))   # untracked

    scene = list(dict.fromkeys(filter(None, (_norm_issue(i) for i in (data.get("scene_issues") or [])))))
    # Object/scene hazards attributed to a person are moved to the scene, so the same
    # unlabelled bottle isn't "new" just because the model filed it differently
    for person in people:
        moved = [i for i in person.issues if i in _SCENE_ONLY_ISSUES]
        if moved:
            person.issues = [i for i in person.issues if i not in _SCENE_ONLY_ISSUES]
            scene = list(dict.fromkeys(scene + moved))
    # Don't double-count: drop scene issues already attributed to a person; person-type issues
    # listed under the scene go to the only person present (or are dropped if nobody is tracked)
    person_issue_set = {i for p in people for i in p.issues}
    cleaned_scene = []
    for issue in scene:
        if issue in person_issue_set:
            continue
        if issue in _PERSON_ISSUES:
            if len(people) == 1:
                people[0].issues.append(issue)
                continue
            if people:
                continue
        cleaned_scene.append(issue)
    return people, cleaned_scene


def _union_people(a: list[PersonObservation], b: list[PersonObservation]) -> list[PersonObservation]:
    out: dict[str, PersonObservation] = {}
    for p in list(a) + list(b):
        if p.person_id in out:
            cur = out[p.person_id]
            cur.issues = list(dict.fromkeys(cur.issues + p.issues))
            cur.appearance = cur.appearance or p.appearance
        else:
            out[p.person_id] = p.model_copy(deep=True)
    return list(out.values())


def _merge_repeat_sightings(
    events: list[DetectedEvent], scene_window: float = SCENE_REPEAT_WINDOW_SEC
) -> tuple[list[DetectedEvent], int]:
    """
    Once a person has been flagged for an issue, later sightings of the SAME person with
    the SAME issue are not flagged again — they're recorded on the first event as
    repeat_timestamps ("also seen at"). Scene hazards (e.g. the same spill) are treated
    the same way while they keep being seen within `scene_window` seconds.
    An event is kept as a new flag if it has at least one new person/issue pair, or if
    it can't be tracked (no person id / issue key).
    Returns (events, number_of_merged_events).
    """
    out: list[DetectedEvent] = []
    first_for_person: dict[tuple[str, str], int] = {}
    scene_seen: dict[str, tuple[int, float]] = {}
    merged = 0

    for e in sorted(events, key=lambda x: (x.timestamp_sec, x.frame_index)):
        if not e.potential_violation:
            out.append(e)
            continue

        person_keys = [(p.person_id, i) for p in e.people if _trackable(p.person_id)
                       for i in p.issues if i != "other"]
        scene_keys = [i for i in e.scene_issues if i != "other"]
        untracked = (not person_keys and not scene_keys) or any(
            (not _trackable(p.person_id) and p.issues) or "other" in p.issues for p in e.people
        ) or "other" in e.scene_issues

        new_person = [k for k in person_keys if k not in first_for_person]
        new_scene = [k for k in scene_keys
                     if k not in scene_seen or e.timestamp_sec - scene_seen[k][1] > scene_window]
        repeat_targets = {first_for_person[k] for k in person_keys if k in first_for_person}
        repeat_targets |= {scene_seen[k][0] for k in scene_keys if k not in new_scene}

        if new_person or new_scene or untracked:
            out.append(e)
            idx = len(out) - 1
            for k in new_person:
                first_for_person[k] = idx
            for k in new_scene:
                scene_seen[k] = (idx, e.timestamp_sec)
        else:
            merged += 1

        # Kept because of NEW issues: tell the reasoner which ones were already reported
        if out and out[-1] is e:
            e.already_reported = [f"{pid}: {iss.replace('_', ' ')}" for pid, iss in person_keys if (pid, iss) not in new_person]
            e.already_reported += [f"scene: {iss.replace('_', ' ')}" for iss in scene_keys if iss not in new_scene]

        # Record this sighting on the earlier flag(s) it repeats (per person+issue too)
        for idx in repeat_targets:
            if out[idx] is not e and e.timestamp_sec not in out[idx].repeat_timestamps:
                out[idx].repeat_timestamps.append(e.timestamp_sec)
        for k in person_keys:
            idx = first_for_person.get(k)
            if idx is not None and out[idx] is not e:
                out[idx].repeat_sightings.setdefault(f"{k[0]}:{k[1]}", []).append(e.timestamp_sec)
        for k in scene_keys:
            if k in scene_seen and out[scene_seen[k][0]] is not e:
                out[scene_seen[k][0]].repeat_sightings.setdefault(f"scene:{k}", []).append(e.timestamp_sec)
        for k in scene_keys:
            if k in scene_seen:
                scene_seen[k] = (scene_seen[k][0], e.timestamp_sec)

    return out, merged


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
