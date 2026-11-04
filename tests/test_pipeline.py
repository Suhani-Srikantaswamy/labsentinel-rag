"""
LabSentinel-RAG — Test Suite
Run: pytest tests/ -v
"""

import json
import uuid
from pathlib import Path

import pytest

from src.utils.models import (
    DetectedEvent,
    VideoAnalysisResult,
    DocumentChunk,
    RetrievedEvidence,
    Violation,
    InvestigationReport,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def event() -> DetectedEvent:
    return DetectedEvent(
        event_id            = str(uuid.uuid4()),
        timestamp_sec       = 12.5,
        frame_index         = 375,
        description         = "Researcher pours acid without splash goggles.",
        objects_involved    = ["hydrochloric acid", "beaker", "pipette"],
        actions_observed    = ["pouring liquid", "no splash goggles"],
        potential_violation = True,
        severity            = "high",
        confidence          = 0.91,
    )


@pytest.fixture
def chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id    = str(uuid.uuid4()),
        source_file = "data/sops/lab_safety_sop_001.txt",
        source_type = "sop",
        title       = "Lab Safety SOP 001",
        content     = "Chemical splash goggles MUST be worn when pouring any liquid chemical.",
        section     = "Section 1.3",
        metadata    = {"filename": "lab_safety_sop_001.txt"},
    )


@pytest.fixture
def evidence(chunk) -> RetrievedEvidence:
    return RetrievedEvidence(chunk=chunk, score=0.87, query_used="acid no goggles")


@pytest.fixture
def violation(event, evidence) -> Violation:
    return Violation(
        violation_id       = str(uuid.uuid4()),
        event              = event,
        violation_type     = "PPE Failure — Eye Protection",
        rule_violated      = "SOP-LAB-001 Section 1.3",
        explanation        = "Researcher poured acid without splash goggles.",
        severity           = "high",
        sop_evidence       = [evidence],
        similar_incidents  = [],
        equipment_notes    = [],
        immediate_action   = "Stop work immediately. Don splash goggles.",
        recommended_action = "Retrain researcher on PPE requirements.",
        preventive_measure = "Install PPE checkpoint at lab entrance.",
    )


@pytest.fixture
def report(event, violation) -> InvestigationReport:
    return InvestigationReport(
        report_id                  = str(uuid.uuid4()),
        video_path                 = "data/videos/test.mp4",
        incident_summary           = "Researcher handled acid without PPE.",
        timeline                   = [event],
        violations                 = [violation],
        overall_risk_level         = "high",
        key_findings               = ["PPE non-compliance during acid handling"],
        immediate_actions_required = ["Stop work", "Enforce PPE"],
    )


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestModels:
    def test_event_fields(self, event):
        assert event.potential_violation is True
        assert event.severity == "high"
        assert 0.0 <= event.confidence <= 1.0

    def test_video_result(self, event):
        r = VideoAnalysisResult(
            video_path           = "test.mp4",
            duration_sec         = 60.0,
            total_frames_sampled = 15,
            events               = [event],
            summary              = "Test",
        )
        assert len(r.events) == 1

    def test_violation_fields(self, violation):
        assert violation.severity == "high"
        assert len(violation.sop_evidence) == 1

    def test_report_violation_count(self, report):
        assert report.total_violations == 1

    def test_report_json_round_trip(self, report):
        data = json.loads(json.dumps(report.model_dump(mode="json"), default=str))
        assert data["total_violations"] == 1
        assert data["overall_risk_level"] == "high"


# ── Document Loader Tests ─────────────────────────────────────────────────────

class TestDocumentLoader:
    def _loader(self):
        from src.ingestion.document_loader import DocumentLoader, _chunk_text
        import types
        m = types.SimpleNamespace(DocumentLoader=DocumentLoader, _chunk_text=_chunk_text)
        return m

    def test_chunk_text_word_count(self):
        mod = self._loader()
        words = " ".join([f"w{i}" for i in range(200)])
        chunks = mod._chunk_text(words, chunk_size=50, overlap=10)
        assert len(chunks) > 0
        for c in chunks:
            assert len(c.split()) <= 50

    def test_chunk_overlap_present(self):
        mod = self._loader()
        words = " ".join([str(i) for i in range(100)])
        chunks = mod._chunk_text(words, chunk_size=20, overlap=5)
        if len(chunks) > 1:
            tail = set(chunks[0].split()[-5:])
            head = set(chunks[1].split()[:5])
            assert tail & head  # overlapping words exist

    def test_load_txt_file(self, tmp_path):
        mod    = self._loader()
        loader = mod.DocumentLoader(chunk_size=20, chunk_overlap=5)
        f = tmp_path / "sop.txt"
        f.write_text("Always wear PPE in the lab. " * 50)
        chunks = loader.load_file(f, "sop")
        assert len(chunks) > 0
        assert all(c.source_type == "sop" for c in chunks)

    def test_load_json_file(self, tmp_path):
        mod    = self._loader()
        loader = mod.DocumentLoader()
        f = tmp_path / "inc.json"
        f.write_text(json.dumps({"content": "Acid spill. No PPE worn."}))
        chunks = loader.load_file(f, "incident")
        assert len(chunks) > 0
        assert "Acid spill" in chunks[0].content

    def test_load_json_with_title(self, tmp_path):
        mod    = self._loader()
        loader = mod.DocumentLoader()
        f = tmp_path / "inc2.json"
        f.write_text(json.dumps({
            "title": "HCl Burn 2024",
            "content": "Researcher burned by acid. No splash goggles worn."
        }))
        chunks = loader.load_file(f, "incident")
        assert any("HCl Burn" in c.content for c in chunks)


# ── Report Generator Tests ────────────────────────────────────────────────────

class TestReportGenerator:
    def test_human_readable_timestamps(self):
        from src.utils.time_utils import format_human_time
        assert format_human_time(0) == "00:00"
        assert format_human_time(12.5) == "00:12"
        assert format_human_time(3661) == "1:01:01"

    def test_json_output(self, report, tmp_path):
        from src.report.report_generator import ReportGenerator
        gen  = ReportGenerator(output_dir=str(tmp_path))
        path = gen.to_json(report)
        data = json.loads(Path(path).read_text())
        assert data["total_violations"] == 1
        assert data["overall_risk_level"] == "high"

    def test_html_output(self, report, tmp_path):
        from src.report.report_generator import ReportGenerator
        gen  = ReportGenerator(output_dir=str(tmp_path))
        path = gen.to_html(report)
        html = Path(path).read_text()
        assert "LabSentinel-RAG" in html
        assert report.report_id in html

    def test_html_contains_violation(self, report, tmp_path):
        from src.report.report_generator import ReportGenerator
        gen  = ReportGenerator(output_dir=str(tmp_path))
        path = gen.to_html(report)
        html = Path(path).read_text()
        assert "PPE Failure" in html
        assert "SOP-LAB-001" in html


# ── Retriever Query Builder Tests ─────────────────────────────────────────────

class TestQueryBuilder:
    def _q(self, event):
        from src.utils.json_utils import build_event_query
        return build_event_query(
            event.description,
            event.objects_involved,
            event.actions_observed,
            event.potential_violation,
            event.severity,
        )

    def test_query_includes_description(self, event):
        assert "acid" in self._q(event).lower()
        assert len(self._q(event)) > 20

    def test_query_includes_objects(self, event):
        assert "hydrochloric acid" in self._q(event)

    def test_query_flags_violation(self, event):
        assert "violation" in self._q(event).lower()


# ── VLM JSON Parser Tests ─────────────────────────────────────────────────────

class TestJsonParser:
    def _parse(self, raw):
        from src.utils.json_utils import parse_llm_json
        return parse_llm_json(raw)

    def test_clean_json(self):
        raw  = '{"severity": "high", "confidence": 0.9, "potential_violation": true}'
        data = self._parse(raw)
        assert data["severity"] == "high"
        assert data["confidence"] == 0.9

    def test_strips_code_fences(self):
        raw  = '```json\n{"severity": "low"}\n```'
        data = self._parse(raw)
        assert data["severity"] == "low"

    def test_malformed_returns_empty(self):
        assert self._parse("not json at all") == {}

    def test_empty_string(self):
        assert self._parse("") == {}


class TestFalsePositiveGuardrails:
    def test_uncertain_ppe_is_not_promoted_to_violation(self):
        from src.vlm.video_analyzer import _sanitize_frame_result

        data = {
            "description": "No clear evidence of lab work in this frame.",
            "people_count": 0,
            "objects_involved": [],
            "actions_observed": [],
            "ppe_status": {
                "lab_coat": "missing",
                "gloves": "missing",
                "eye_protection": "missing",
                "closed_toe_shoes": "cannot_determine",
            },
            "checklist_findings": ["No PPE visible"],
            "potential_violation": True,
            "violation_hint": "No PPE visible",
            "severity": "medium",
            "confidence": 0.58,
        }

        cleaned = _sanitize_frame_result(data)
        assert cleaned["potential_violation"] is False
        assert cleaned["severity"] == "none"

    def test_direct_ppe_observation_still_counts(self):
        from src.vlm.video_analyzer import _sanitize_frame_result

        data = {
            "description": "Researcher is handling chemicals without safety goggles.",
            "people_count": 1,
            "objects_involved": ["beaker", "chemical bottle"],
            "actions_observed": ["pouring liquid"],
            "ppe_status": {
                "lab_coat": "worn",
                "gloves": "worn",
                "eye_protection": "missing",
                "closed_toe_shoes": "worn",
            },
            "checklist_findings": ["Researcher missing eye protection while pouring liquid"],
            "potential_violation": True,
            "violation_hint": "Missing goggles while pouring chemical",
            "severity": "medium",
            "confidence": 0.87,
        }

        cleaned = _sanitize_frame_result(data)
        assert cleaned["potential_violation"] is True
        assert cleaned["severity"] in {"medium", "high"}

    def test_clear_equipment_violation_is_not_rejected(self):
        from src.vlm.video_analyzer import _sanitize_frame_result

        data = {
            "description": "Centrifuge lid is open while rotor is running.",
            "people_count": 0,
            "objects_involved": ["centrifuge", "rotor", "sample tubes"],
            "actions_observed": ["centrifuge running"],
            "ppe_status": {
                "lab_coat": "cannot_determine",
                "gloves": "cannot_determine",
                "eye_protection": "cannot_determine",
                "closed_toe_shoes": "cannot_determine",
            },
            "checklist_findings": ["Centrifuge lid open while in operation"],
            "potential_violation": True,
            "violation_hint": "Centrifuge lid open while running",
            "severity": "high",
            "confidence": 0.91,
        }

        cleaned = _sanitize_frame_result(data)
        assert cleaned["potential_violation"] is True
        assert cleaned["severity"] == "high"


class TestDetectionQuality:
    def test_collapse_adjacent_duplicate_events(self):
        from src.vlm.video_analyzer import _collapse_adjacent_events

        events = [
            DetectedEvent(event_id="1", timestamp_sec=12.0, frame_index=1, description="Researcher without goggles", objects_involved=["goggles"], actions_observed=["pouring"], potential_violation=True, severity="medium", confidence=0.82),
            DetectedEvent(event_id="2", timestamp_sec=13.0, frame_index=2, description="Researcher without goggles", objects_involved=["goggles"], actions_observed=["pouring"], potential_violation=True, severity="medium", confidence=0.84),
            DetectedEvent(event_id="3", timestamp_sec=25.0, frame_index=3, description="Researcher wearing goggles again", objects_involved=["goggles"], actions_observed=["standing"], potential_violation=False, severity="none", confidence=0.5),
            DetectedEvent(event_id="4", timestamp_sec=30.0, frame_index=4, description="Researcher without goggles", objects_involved=["goggles"], actions_observed=["pouring"], potential_violation=True, severity="medium", confidence=0.81),
        ]

        collapsed = _collapse_adjacent_events(events)
        assert len(collapsed) == 2
        assert collapsed[0].timestamp_sec == 12.0
        assert collapsed[1].timestamp_sec == 30.0

    def test_normalize_manual_files_handles_single_and_nested_lists(self):
        from app import _normalize_manual_files

        files = [["/tmp/a.txt"], "/tmp/b.txt", None, ["/tmp/c.txt"]]
        normalized = _normalize_manual_files(files)
        assert normalized == ["/tmp/a.txt", "/tmp/b.txt", "/tmp/c.txt"]
