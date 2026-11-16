"""
End-to-end test: real pipeline (frame sampling, merging, manual indexing,
retrieval, reasoning flow, JSON/HTML/PDF reports, review flow) with a fake LLM.

Catches regressions like "0 violations because every frame was silently
dropped" without spending API quota.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.llm_client import LLMResponse

MANUAL = (
    "=== test_manual.txt ===\n"
    "Section 4.2 Eye protection: Chemical splash goggles MUST be worn when handling acids.\n"
    "Section 7.1 Spills: Any chemical spill must be cleaned immediately using the spill kit.\n"
    "Section 9.0 Food: No food or drink in the laboratory.\n"
)


class FakeLLM:
    """Deterministic stand-in for Gemini. Scenes: 0–10s ok, 10–20s no goggles,
    20–30s ok (24s returns garbage), 30–40s spill."""

    def __init__(self):
        self.calls = {"frame": 0, "timeline": 0, "violation": 0, "summary": 0}

    def chat(self, prompt, images=None, temperature=0.1, max_tokens=900):
        if images:
            self.calls["frame"] += 1
            ts = float(re.search(r"Analyse this frame at timestamp ([\d.]+)s", prompt).group(1))
            if 24 <= ts < 25:
                return LLMResponse('{"lab_type": "chemistry", "description": "cut off')   # unparseable
            if 10 <= ts < 20:
                return LLMResponse(json.dumps({
                    "lab_type": "chemistry", "description": "Researcher handling acid without goggles.",
                    "people_count": 1, "objects_involved": ["acid bottle", "beaker"],
                    "actions_observed": ["pouring acid"],
                    "ppe_status": {"lab_coat": "worn", "gloves": "worn", "eye_protection": "missing"},
                    "checklist_findings": ["Researcher handling acid without goggles"],
                    "potential_violation": True, "violation_hint": "No goggles while pouring acid",
                    "severity": "high", "confidence": 0.92}))
            if 30 <= ts < 40:
                return LLMResponse(json.dumps({
                    "lab_type": "chemistry", "description": "Acid spill left uncleaned on the bench.",
                    "people_count": 0, "objects_involved": ["spill", "bench"], "actions_observed": [],
                    "ppe_status": {"lab_coat": "cannot_determine"},
                    "checklist_findings": ["Chemical spill not cleaned up"],
                    "potential_violation": True, "violation_hint": "Uncleaned acid spill",
                    "severity": "medium", "confidence": 0.85}))
            return LLMResponse(json.dumps({
                "lab_type": "chemistry", "description": "Researcher working with full PPE.",
                "people_count": 1, "objects_involved": ["beaker"], "actions_observed": ["writing"],
                "ppe_status": {"lab_coat": "worn", "gloves": "worn", "eye_protection": "worn"},
                "checklist_findings": [], "potential_violation": False,
                "violation_hint": "None", "severity": "none", "confidence": 0.95}))
        if "CROSS-FRAME" in prompt:
            self.calls["timeline"] += 1
            return LLMResponse('{"incident_summary": "Goggles missing, then a spill.", "persistent_violations": []}')
        if "DETECTED EVENT" in prompt:
            self.calls["violation"] += 1
            spill = "spill" in prompt.lower().split("detected event")[1][:400]
            return LLMResponse(json.dumps({
                "is_violation": True,
                "violation_type": "Chemical Handling" if spill else "PPE",
                "rule_violated": "Section 7.1 Spills" if spill else "Section 4.2 Eye protection",
                "explanation": "Observed in footage.", "severity": "medium" if spill else "high",
                "immediate_action": "Stop work.", "recommended_action": "Retrain.",
                "preventive_measure": "Checklist."}))
        if "executive_summary" in prompt:
            self.calls["summary"] += 1
            return LLMResponse(json.dumps({
                "executive_summary": "Two violations found.", "overall_risk_level": "low",   # LLM under-rates
                "key_findings": ["PPE", "Spill"], "immediate_actions": ["Clean spill"]}))
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


def _make_video(path: Path, seconds=40, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    colours = [(40, 160, 40), (40, 40, 200), (200, 160, 40), (160, 40, 160)]   # new scene every 10s
    for i in range(seconds * fps):
        frame = np.full((240, 320, 3), colours[(i // (10 * fps)) % 4], np.uint8)
        cv2.putText(frame, f"{i / fps:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        w.write(frame)
    w.release()
    return path


@pytest.fixture
def e2e_env(tmp_path, monkeypatch):
    import importlib
    cfg = importlib.import_module("config.settings")
    import src.vlm.video_chunker as chunker
    import src.vlm.video_analyzer as analyzer
    import src.reasoning.violation_reasoner as reasoner
    import src.report.report_generator as rg

    out = tmp_path / "outputs"; out.mkdir()
    monkeypatch.setattr(cfg.settings.rag, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(cfg, "OUTPUT_DIR", out)
    monkeypatch.setattr(chunker, "OUTPUT_DIR", out)
    monkeypatch.setattr(rg, "OUTPUT_DIR", out)
    monkeypatch.setattr(cfg.settings.report, "output_format", ["json", "html", "pdf"])

    fake = FakeLLM()
    monkeypatch.setattr(analyzer, "get_client", lambda *a, **k: fake)
    monkeypatch.setattr(reasoner, "get_client", lambda *a, **k: fake)
    video = _make_video(tmp_path / "incident.mp4")
    return {"video": video, "out": out, "fake": fake}


@pytest.fixture
def e2e_report(e2e_env):
    from src.pipeline import LabSentinelPipeline
    report = LabSentinelPipeline().run(
        video_path=str(e2e_env["video"]), manual_text=MANUAL,
        report_id="e2e-test", output_dir=str(e2e_env["out"]), verbose=False)
    return report, e2e_env


class TestEndToEnd:
    def test_detects_both_violations_with_correct_timestamps(self, e2e_report):
        report, _ = e2e_report
        assert report.total_violations == 2
        ts = sorted(v.event.timestamp_sec for v in report.violations)
        assert 10 <= ts[0] < 13 and 30 <= ts[1] < 33
        assert {v.violation_type for v in report.violations} == {"PPE", "Chemical Handling"}

    def test_frames_cover_whole_video(self, e2e_report):
        report, env = e2e_report
        assert report.metadata["frames_sampled"] >= 7
        assert env["fake"].calls["frame"] == report.metadata["frames_sampled"]

    def test_failed_frame_is_reported_not_silent(self, e2e_report):
        report, _ = e2e_report
        assert report.metadata["failed_frames"] == 1
        assert any("could not be analysed" in w for w in report.metadata["warnings"])

    def test_risk_not_lower_than_worst_violation(self, e2e_report):
        report, _ = e2e_report
        assert report.overall_risk_level == "high"     # LLM said "low"; worst violation is high

    def test_uploaded_manual_is_retrieved_as_evidence(self, e2e_report):
        report, _ = e2e_report
        for v in report.violations:
            assert v.sop_evidence and "(uploaded manual)" in v.sop_evidence[0].chunk.title

    def test_report_files_and_features(self, e2e_report):
        report, env = e2e_report
        out = env["out"]
        for ext in ("json", "html", "pdf"):
            assert (out / f"report_e2e-test.{ext}").is_file()
        html = (out / "report_e2e-test.html").read_text()
        assert html.count("data:image/jpeg;base64,") == 2           # screenshot per violation
        assert "<svg" in html and html.count('class="marker seek"') == 2   # timeline chart
        assert 'data-seek="' in html and "<video" in html           # clickable timestamps + player
        assert "data-review" not in html                            # no buttons in standalone file
        saved = json.loads((out / "report_e2e-test.json").read_text())
        assert saved["total_violations"] == 2

    def test_review_flow_persists(self, e2e_report):
        import app
        report, env = e2e_report
        vid = report.violations[0].violation_id
        ui_html = app.review_violation(f"e2e-test|{vid}|false_alarm")[0]
        assert "Marked false alarm" in ui_html and "data-review" in ui_html
        saved = json.loads((env["out"] / "report_e2e-test.json").read_text())
        v = next(x for x in saved["violations"] if x["violation_id"] == vid)
        assert v["review_status"] == "false_alarm" and v["reviewed_at"]
        app.review_violation(f"e2e-test|{vid}|pending")
        saved = json.loads((env["out"] / "report_e2e-test.json").read_text())
        assert next(x for x in saved["violations"] if x["violation_id"] == vid)["review_status"] == "pending"

    def test_review_rejects_bad_input(self, e2e_report):
        import app
        assert "Invalid" in app.review_violation("garbage")[3]
        assert "Invalid" in app.review_violation("e2e-test|x|delete_everything")[3]
        assert "not found" in app.review_violation("../../etc|x|confirmed")[3]
