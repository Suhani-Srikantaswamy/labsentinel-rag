"""
Pydantic data models shared across all LabSentinel-RAG modules.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Video Analysis Models ─────────────────────────────────────────────────────

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


class VideoAnalysisResult(BaseModel):
    """Full result of VLM-based video analysis."""
    video_path: str
    duration_sec: float
    total_frames_sampled: int
    events: list[DetectedEvent] = Field(default_factory=list)
    summary: str = ""
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


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


# ── Final Report Model ────────────────────────────────────────────────────────

class InvestigationReport(BaseModel):
    """Complete investigation report — the final output of LabSentinel-RAG."""
    report_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
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
