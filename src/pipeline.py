"""
LabSentinel-RAG — Pipeline Orchestrator

Stage 1: Validate inputs
Stage 2: Video Analysis  — the vision model (Gemini / GPT-4o) reads sampled frames against the uploaded manual
Stage 3: RAG Retrieval   — retrieve relevant passages from the manual for each event
Stage 4: LLM Reasoning   — confirm violations, cite exact rule, give recommendations
Stage 5: Output          — generate structured report (JSON + HTML + PDF)

The uploaded lab manual drives everything — no hardcoded rules anywhere.
"""

from __future__ import annotations
import time
from pathlib import Path

from loguru import logger
from rich.console import Console

from config.settings import settings
from src.vlm.video_analyzer import VideoAnalyzer
from src.retrieval.retriever import EvidenceRetriever
from src.reasoning.violation_reasoner import ViolationReasoner
from src.report.report_generator import ReportGenerator
from src.utils.models import InvestigationReport, Violation

console = Console()


class LabSentinelPipeline:
    """End-to-end incident investigation pipeline."""

    def __init__(self):
        logger.info("Initialising pipeline …")
        self.analyzer  = VideoAnalyzer()
        self.retriever = EvidenceRetriever()
        self.reasoner  = ViolationReasoner()
        self.reporter  = ReportGenerator()
        logger.success("Pipeline ready")

    def run(
        self,
        video_path: str,
        manual_text: str = "",          # full text extracted from uploaded manual(s)
        report_id: str | None = None,
        output_dir: str | None = None,
        sample_rate: int | None = None,
        max_frames: int | None = None,
        chunk_duration_sec: int | None = None,
        on_progress=None,               # callback(pct: float, msg: str) for UI
        verbose: bool = True,
    ) -> InvestigationReport:
        """
        Run the full pipeline.

        Args:
            video_path:   Path to the incident video
            manual_text:  Full text of the uploaded lab manual.
                          If empty, universal safety rules are applied.
            All other args are optional overrides / output config.
        """
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        def _prog(pct, msg):
            if on_progress:
                on_progress(pct, msg)
            if verbose:
                console.print(f"  [dim]{pct*100:.0f}%[/dim] {msg}")

        from src.utils.llm_client import request_count
        t0, req0 = time.time(), request_count()
        _prog(0.01, "Starting investigation …")

        # ── Stage 2: Video Analysis ───────────────────────────────────────────
        _prog(0.02, "Stage 2 — Analysing video frames …")
        video_result = self.analyzer.analyse_video(
            video_path         = video_path,
            manual_text        = manual_text,
            sample_rate        = sample_rate,
            max_frames         = max_frames,
            chunk_duration_sec = chunk_duration_sec,
            on_progress        = lambda p, m: _prog(0.02 + p * 0.53, m),
        )
        violation_events = [e for e in video_result.events if e.potential_violation]
        _prog(0.56, f"{len(video_result.events)} events detected, "
                    f"{len(violation_events)} flagged as potential violations")

        # ── Stage 3: RAG Retrieval ────────────────────────────────────────────
        _prog(0.57, "Stage 3 — Retrieving relevant manual passages …")
        if violation_events and manual_text.strip():
            try:
                n_passages = self.retriever.index_session_manual(manual_text)
                _prog(0.575, f"Indexed uploaded manual — {n_passages} passages searchable")
            except Exception as e:
                logger.warning(f"Could not index uploaded manual for retrieval: {e}")
        all_evidence: dict[str, dict] = {}
        n = len(violation_events)
        for i, event in enumerate(violation_events):
            evidence = self.retriever.retrieve_for_event(event)
            all_evidence[event.event_id] = evidence
            _prog(0.57 + (i + 1) / max(n, 1) * 0.10,
                  f"Retrieved evidence for event @ {event.timestamp_sec:.1f}s")

        # ── Stage 4: LLM Reasoning ────────────────────────────────────────────
        _prog(0.68, "Stage 4 — Reasoning over violations …")
        self.retriever.clear_session_manual()
        self.reasoner.failed_events = 0
        violations: list[Violation] = self.reasoner.analyse_events(
            violation_events, all_evidence, manual_text,
            on_progress=lambda p, m: _prog(0.68 + p * 0.18, m),
        )
        from src.reasoning.violation_reasoner import consolidate_by_person
        violations, folded = consolidate_by_person(violations)
        if folded:
            logger.info(f"Folded {folded} repeat violation(s) into the same person's earlier violation")

        # ── Stage 5: Report ───────────────────────────────────────────────────
        _prog(0.87, "Stage 5 — Building report …")
        report = self.reasoner.build_report(
            video_result, violations, report_id, output_dir
        )
        warnings = []
        if video_result.failed_frames:
            warnings.append(
                f"{video_result.failed_frames} of {video_result.total_frames_sampled} frames could not be "
                "analysed (API errors or unreadable replies) — violations in those moments may be missing.")
        if self.reasoner.failed_events:
            warnings.append(
                f"{self.reasoner.failed_events} flagged event(s) could not be verified because the reasoning "
                "step failed — they are listed as unverified violations.")
        report.metadata["warnings"] = warnings
        report.metadata["failed_frames"] = video_result.failed_frames
        report.metadata["repeats_merged"] = video_result.repeats_merged + folded
        report.metadata["analysis_seconds"] = round(time.time() - t0, 1)
        report.metadata["api_requests"] = request_count() - req0
        output_paths = ReportGenerator(output_dir=output_dir).generate_all(report)
        _prog(1.0, f"Done — {report.total_violations} violations, "
                   f"risk: {report.overall_risk_level.upper()}")

        if verbose:
            console.rule("[bold green]Investigation Complete[/bold green]")
            console.print(f"  Violations : [red]{report.total_violations}[/red]")
            console.print(f"  Risk level : [bold]{report.overall_risk_level.upper()}[/bold]")
            for fmt, path in output_paths.items():
                console.print(f"  [cyan]{fmt.upper():5}[/cyan] → {path}")

        return report
