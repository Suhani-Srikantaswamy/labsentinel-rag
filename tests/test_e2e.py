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
    """Deterministic stand-in for Gemini. Scenes: 0–10s ok, 10–20s P1 no goggles,
    20–30s ok (24s unreadable), 30–40s spill, 40–50s P1 no goggles AGAIN (repeat).
    Understands single-frame, batched-frame, batched-reasoning and reconciliation prompts."""

    P1 = "blue lab coat, short dark hair, left bench"

    def __init__(self):
        self.calls = {"frame": 0, "frame_requests": 0, "violation": 0, "summary": 0, "reconcile": 0}
        self.known_people_prompts = []
        self.lock = __import__("threading").Lock()

    def _frame(self, ts, pid="P1"):
        p_ok = [{"person_id": pid, "appearance": self.P1, "issues": []}]
        p_bad = [{"person_id": pid, "appearance": self.P1, "issues": ["missing_eye_protection"]}]
        if 10 <= ts < 20:
            return {"lab_type": "chemistry", "description": "Researcher handling acid without goggles.",
                    "people_count": 1, "people": p_bad, "scene_issues": [],
                    "objects_involved": ["acid bottle", "beaker"], "actions_observed": ["pouring acid"],
                    "ppe_status": {"lab_coat": "worn", "gloves": "worn", "eye_protection": "missing"},
                    "checklist_findings": ["Researcher handling acid without goggles"],
                    "potential_violation": True, "violation_hint": "No goggles while pouring acid",
                    "severity": "high", "confidence": 0.92}
        if 30 <= ts < 40:
            return {"lab_type": "chemistry", "description": "Acid spill left uncleaned on the bench.",
                    "people_count": 0, "people": [], "scene_issues": ["spill"],
                    "objects_involved": ["spill", "bench"], "actions_observed": [],
                    "ppe_status": {"lab_coat": "cannot_determine"},
                    "checklist_findings": ["Chemical spill not cleaned up"],
                    "potential_violation": True, "violation_hint": "Uncleaned acid spill",
                    "severity": "medium", "confidence": 0.85}
        if 40 <= ts < 50:   # same person, same issue, worded differently → must NOT be a new violation
            return {"lab_type": "chemistry",
                    "description": "The same worker is back at the fume hood, eyes unprotected while decanting.",
                    "people_count": 1, "people": p_bad, "scene_issues": [],
                    "objects_involved": ["fume hood", "flask"], "actions_observed": ["decanting"],
                    "ppe_status": {"lab_coat": "worn", "gloves": "worn", "eye_protection": "missing"},
                    "checklist_findings": ["No eye protection while decanting"],
                    "potential_violation": True, "violation_hint": "Eye protection missing",
                    "severity": "high", "confidence": 0.9}
        return {"lab_type": "chemistry", "description": "Researcher working with full PPE.",
                "people_count": 1, "people": p_ok, "scene_issues": [],
                "objects_involved": ["beaker"], "actions_observed": ["writing"],
                "ppe_status": {"lab_coat": "worn", "gloves": "worn", "eye_protection": "worn"},
                "checklist_findings": [], "potential_violation": False,
                "violation_hint": "None", "severity": "none", "confidence": 0.95}

    @staticmethod
    def _violation_for(block):
        desc = re.search(r"description\s*:\s*(.*)", block, re.I)
        spill = "spill" in (desc.group(1) if desc else block).lower()
        return {"violation_type": "Chemical Handling" if spill else "PPE",
                "issues": ["spill"] if spill else ["missing_eye_protection"],
                "person_ids": [] if spill else re.findall(r"\b(P\d+)\b", block)[:1],
                "rule_violated": "Section 7.1 Spills" if spill else "Section 4.2 Eye protection",
                "explanation": "Observed in footage.", "severity": "medium" if spill else "high",
                "immediate_action": "Stop work.", "recommended_action": "Retrain.",
                "preventive_measure": "Checklist."}

    def chat(self, prompt, images=None, temperature=0.1, max_tokens=900):
        with self.lock:
            if images:
                self.calls["frame_requests"] += 1
                self.calls["frame"] += len(images)
                self.known_people_prompts.append(
                    (prompt, re.search(r"KNOWN PEOPLE[^:]*: (.*)", prompt).group(1)))
                if len(images) == 1 and "Analyse this frame at timestamp" in prompt:
                    ts = float(re.search(r"Analyse this frame at timestamp ([\d.]+)s", prompt).group(1))
                    if 24 <= ts < 25:
                        return LLMResponse('{"lab_type": "chemistry", "description": "cut off')   # unparseable
                    return LLMResponse(json.dumps(self._frame(ts)))
                tss = [float(x) for x in re.findall(r"Image \d+ → t=([\d.]+)s", prompt)]
                assert len(tss) == len(images)
                # Batch reply leaves out the 24s frame → analyzer must retry it on its own
                frames = [dict(self._frame(t), image=i + 1) for i, t in enumerate(tss) if not (24 <= t < 25)]
                return LLMResponse(json.dumps({"frames": frames}))
            if "People were detected in different parts" in prompt:
                self.calls["reconcile"] += 1
                ids = re.findall(r"^- (P\d+) ", prompt, re.M)
                return LLMResponse(json.dumps({"groups": [ids]}))     # all the same researcher
            if "flagged moments from ONE incident video" in prompt:
                self.calls["violation"] += 1
                blocks = re.split(r"━━ EVENT (E\d+) ━━", prompt)[1:]
                events = [{"event": blocks[i], "violations": [self._violation_for(blocks[i + 1].split("━━━━")[0])]}
                          for i in range(0, len(blocks), 2)]
                return LLMResponse(json.dumps({"events": events}))
            if "DETECTED EVENT" in prompt:
                self.calls["violation"] += 1
                block = prompt.split("DETECTED EVENT")[1][:800]
                return LLMResponse(json.dumps({"violations": [self._violation_for(block)]}))
            if "executive_summary" in prompt:
                self.calls["summary"] += 1
                return LLMResponse(json.dumps({
                    "executive_summary": "Two violations found.", "overall_risk_level": "low",   # LLM under-rates
                    "key_findings": ["PPE", "Spill"], "immediate_actions": ["Clean spill"]}))
            raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


def _make_video(path: Path, seconds=50, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    colours = [(40, 160, 40), (40, 40, 200), (200, 160, 40), (160, 40, 160), (40, 200, 200)]   # new scene every 10s
    for i in range(seconds * fps):
        frame = np.full((240, 320, 3), colours[(i // (10 * fps)) % 5], np.uint8)
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
        fake = env["fake"]
        # every frame analysed once in a batch, plus the one retried on its own
        assert fake.calls["frame"] == report.metadata["frames_sampled"] + 1
        assert fake.calls["frame_requests"] < report.metadata["frames_sampled"] / 2   # batching cut requests

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
        assert "Invalid" in app.review_violation("garbage")[4]
        assert "Invalid" in app.review_violation("e2e-test|x|delete_everything")[4]
        assert "not found" in app.review_violation("../../etc|x|confirmed")[4]


class TestPerPersonDeduplication:
    def test_repeat_by_same_person_not_flagged_again(self, e2e_report):
        report, env = e2e_report
        assert report.total_violations == 2                       # not 3
        ppe = next(v for v in report.violations if v.violation_type == "PPE")
        assert ppe.event.people[0].person_id == "P1"
        assert ppe.repeat_times() and all(40 <= t < 50 for t in ppe.repeat_times())
        spill = next(v for v in report.violations if v.violation_type == "Chemical Handling")
        assert spill.repeat_times() == []                          # P1's repeat doesn't leak onto the spill
        assert report.metadata["repeats_merged"] >= 1
        assert env["fake"].calls["violation"] == 1                # both events verified in ONE request

    def test_people_reconciled_to_one_person(self, e2e_report):
        report, env = e2e_report
        ids = {p.person_id for e in report.timeline for p in e.people}
        assert ids == {"P1"}                                       # one researcher → one ID

    def test_report_lists_repeat_sightings(self, e2e_report):
        report, env = e2e_report
        html = (env["out"] / "report_e2e-test.html").read_text()
        assert "Also seen at" in html and 'class="repeat-chip seek"' in html
        assert 'class="marker-repeat seek"' in html and "👤 P1" in html
