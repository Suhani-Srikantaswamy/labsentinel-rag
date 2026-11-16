"""
Pydantic data models shared across all LabSentinel-RAG modules.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


# ── Video Analysis Models ─────────────────────────────────────────────────────

class PersonObservation(BaseModel):
    """One person seen in a frame, tracked across frames by a stable ID (P1, P2, …)."""
    person_id: str
    appearance: str = ""                                 # clothing / hair / position — used to re-identify
    issues: list[str] = Field(default_factory=list)      # normalised issue keys, e.g. "missing_eye_protection"


class DetectedEvent(BaseModel):
    """A single event/observation detected in the incident video."""
    event_id: str
    timestamp_sec: float = Field(..., description="Time offset in the video (seconds)")
    frame_index: int
    description: str = Field(..., description="Natural language description of what is happening")
    objects_involved: list[str] = Field(default_factory=list)
    actions_observed: list[str] = Field(default_factory=list)
    potential_violation: bool = False
    severity: str = Field(default="unknown", description="low | medium | high | critical")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    frame_path: Optional[str] = None   # path to extracted frame image
    people: list[PersonObservation] = Field(default_factory=list)
    scene_issues: list[str] = Field(default_factory=list)       # hazards not tied to a person (spill, blocked exit…)
    repeat_timestamps: list[float] = Field(default_factory=list)  # later sightings of the SAME person+issue
    repeat_sightings: dict[str, list[float]] = Field(default_factory=dict)  # "P1:missing_gloves" → [ts, …]
    already_reported: list[str] = Field(default_factory=list)   # issues in this frame flagged earlier ("P1: missing gloves")


class VideoAnalysisResult(BaseModel):
    """Full result of VLM-based video analysis."""
    video_path: str
    duration_sec: float
    total_frames_sampled: int
    events: list[DetectedEvent] = Field(default_factory=list)
    summary: str = ""
    failed_frames: int = 0     # frames the VLM could not analyse (API error / unparseable reply)
    repeats_merged: int = 0    # flagged frames folded into an earlier violation (same person + same issue)
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Knowledge Base Models ─────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A chunk of text from a knowledge-base document."""
    chunk_id: str
    source_file: str
    source_type: str = Field(..., description="sop | manual | incident")
    title: str = ""
    content: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    """A retrieved knowledge-base chunk with its relevance score."""
    chunk: DocumentChunk
    score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    query_used: str = ""


# ── Violation Models ──────────────────────────────────────────────────────────

class Violation(BaseModel):
    """A confirmed safety violation identified after RAG + LLM reasoning."""
    violation_id: str
    event: DetectedEvent
    violation_type: str
    rule_violated: str
    explanation: str
    severity: str
    sop_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    similar_incidents: list[RetrievedEvidence] = Field(default_factory=list)
    equipment_notes: list[RetrievedEvidence] = Field(default_factory=list)
    recommended_action: str = ""
    immediate_action: str = ""
    preventive_measure: str = ""
    issues: list[str] = Field(default_factory=list)       # issue keys this violation covers
    person_ids: list[str] = Field(default_factory=list)   # people this violation is about (P1, …)
    def repeat_times(self) -> list[float]:
        """Later sightings of THIS violation's person+issue (falls back to the whole event)."""
        rs = self.event.repeat_sightings
        if self.issues and rs:
            owners = self.person_ids or ["scene"]
            keys = {f"{o}:{i}" for o in owners for i in self.issues}
            return sorted({t for k in keys for t in rs.get(k, [])})
        return sorted(self.event.repeat_timestamps)

    # Human review — set from the web UI ("Confirm" / "False alarm" buttons)
    review_status: str = Field(default="pending", description="pending | confirmed | false_alarm")
    reviewed_at: Optional[datetime] = None


# ── Final Report Model ────────────────────────────────────────────────────────

class InvestigationReport(BaseModel):
    """Complete investigation report — the final output of LabSentinel-RAG."""
    report_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    video_path: str
    incident_summary: str
    timeline: list[DetectedEvent]
    violations: list[Violation]
    overall_risk_level: str = Field(..., description="low | medium | high | critical")
    total_violations: int = 0
    key_findings: list[str] = Field(default_factory=list)
    immediate_actions_required: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def model_post_init(self, __context):
        self.total_violations = len(self.violations)

    def review_counts(self) -> dict[str, int]:
        counts = {"pending": 0, "confirmed": 0, "false_alarm": 0}
        for v in self.violations:
            counts[v.review_status if v.review_status in counts else "pending"] += 1
        return counts
