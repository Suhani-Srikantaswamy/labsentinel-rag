"""
Report Generator — Stage 5 of LabSentinel-RAG pipeline.

Generates the final investigation report in three formats:
  1. JSON  — machine-readable, full fidelity (includes reviewer decisions)
  2. HTML  — rendered from templates/report.html.j2 (Jinja2, auto-escaped):
             evidence screenshots, clickable timestamps, violation timeline chart,
             and — in the web UI — Confirm / False-alarm review buttons
  3. PDF   — printable formal report (reportlab) with screenshots and timeline

The same template serves both the standalone HTML file and the in-app view;
`interactive=True` adds review buttons, `include_player=True` embeds a video
player that the timestamps seek.
"""

from __future__ import annotations
import base64
import html as _html
import io
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger
from markupsafe import Markup

from config.settings import settings, OUTPUT_DIR
from src.utils.models import InvestigationReport
from src.utils.time_utils import format_human_time
from src.report.theme import (
    SEVERITY_ORDER, SEVERITY_COLORS, FALSE_ALARM_COLOR,
    REVIEW_LABELS, REVIEW_COLORS, severity_color,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _e(value) -> str:
    """Escape model/user text for reportlab's mini-markup (and any raw HTML)."""
    return _html.escape(str(value if value is not None else ""), quote=True)


def _file_stem(report_id: str) -> str:
    """Safe, short filename stem (keeps the historical 8-char uuid prefix)."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(report_id))[:40] or "report"
    return f"report_{safe[:8] if len(safe) == 36 else safe}"


def report_json_path(report_id: str, output_dir: str | Path | None = None) -> Path:
    return Path(output_dir or OUTPUT_DIR) / f"{_file_stem(report_id)}.json"


def _fmt_ts(seconds: float) -> str:
    """Backwards-compatible timestamp formatter for report HTML and UI output."""
    return format_human_time(seconds)


def _rule_source(rule: str) -> str:
    r = str(rule).lower()
    if "not verified" in r:
        return "⚠ Unverified"
    return "🌐 Universal Rule" if "universal" in r else "📋 From Lab Manual"


def _people_labels(event, violation=None) -> list[dict]:
    """[{label, issues}] for people with issues in this event (only this violation's, if given)."""
    out = []
    only_ids = set(violation.person_ids) if violation is not None and violation.person_ids else None
    only_issues = set(violation.issues) if violation is not None and violation.issues else None
    for p in getattr(event, "people", []) or []:
        if not p.issues or (only_ids is not None and p.person_id not in only_ids):
            continue
        issues = [i for i in p.issues if only_issues is None or i in only_issues] or p.issues
        who = p.person_id if p.person_id != "P?" else "Person"
        label = f"{who} · {p.appearance[:60]}" if p.appearance else who
        out.append({"label": label, "issues": ", ".join(i.replace("_", " ") for i in issues)})
    return out


def _duration(report: InvestigationReport) -> float:
    try:
        d = float(report.metadata.get("duration_sec") or 0)
    except (TypeError, ValueError):
        d = 0.0
    latest = max([e.timestamp_sec for e in report.timeline] +
                 [v.event.timestamp_sec for v in report.violations] +
                 [t for v in report.violations for t in v.repeat_times()] + [0.0])
    return max(d, latest + 1.0, 1.0)


def _nice_tick_step(duration: float) -> float:
    for step in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200):
        if duration / step <= 8:
            return step
    return 14400


def timeline_svg(report: InvestigationReport) -> str:
    """
    SVG strip across the video's length with one marker per violation, coloured
    by severity (false alarms grey). Markers carry data-seek for click-to-play.
    """
    W, H, PAD = 640, 96, 26
    axis_y = 62
    dur = _duration(report)
    x = lambda t: PAD + (W - 2 * PAD) * min(max(t, 0.0), dur) / dur

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="Violation timeline">',
             f'<rect x="{PAD}" y="{axis_y - 3}" width="{W - 2 * PAD}" height="6" rx="3" fill="#334155"/>']

    step = _nice_tick_step(dur)
    t = 0.0
    while t <= dur + 1e-6:
        xx = x(t)
        parts.append(f'<line x1="{xx:.1f}" y1="{axis_y + 5}" x2="{xx:.1f}" y2="{axis_y + 11}" stroke="#475569"/>')
        parts.append(f'<text x="{xx:.1f}" y="{axis_y + 26}" font-size="14" fill="#94a3b8" '
                     f'text-anchor="middle" font-family="monospace">{_e(format_human_time(t))}</text>')
        t += step

    # Repeat sightings: small hollow dots on the axis, same colour, clickable
    for v in report.violations:
        col = FALSE_ALARM_COLOR if v.review_status == "false_alarm" else severity_color(v.severity)
        for rt in v.repeat_times():
            rx = x(rt)
            parts.append(
                f'<g class="marker-repeat seek" data-seek="{rt:.2f}" style="cursor:pointer">'
                f'<title>{_e(format_human_time(rt))} — repeat of {_e(v.violation_type)} (not flagged again)</title>'
                f'<circle cx="{rx:.1f}" cy="{axis_y}" r="5" fill="#0f172a" stroke="{col}" stroke-width="2"/></g>')

    # Stagger markers that would overlap
    rows_last_x = [-1e9, -1e9, -1e9]
    for v in sorted(report.violations, key=lambda v: v.event.timestamp_sec):
        ts = v.event.timestamp_sec
        xx = x(ts)
        row = next((i for i, lx in enumerate(rows_last_x) if xx - lx >= 20), 0)
        rows_last_x[row] = xx
        cy = axis_y - 17 - row * 17
        fa = v.review_status == "false_alarm"
        col = FALSE_ALARM_COLOR if fa else severity_color(v.severity)
        fill = "none" if fa else col
        title = f"{format_human_time(ts)} — {v.severity.upper()} — {v.violation_type}"
        if v.review_status != "pending":
            title += f" ({REVIEW_LABELS.get(v.review_status, v.review_status)})"
        parts.append(
            f'<g class="marker seek" data-seek="{ts:.2f}" style="cursor:pointer">'
            f'<title>{_e(title)}</title>'
            f'<line x1="{xx:.1f}" y1="{cy:.1f}" x2="{xx:.1f}" y2="{axis_y}" stroke="{col}" stroke-width="1.5" opacity="0.6"/>'
            f'<circle cx="{xx:.1f}" cy="{cy:.1f}" r="8" fill="{fill}" stroke="{col}" stroke-width="2.5"/></g>')

    if not report.violations:
        parts.append(f'<text x="{W / 2}" y="{axis_y - 20}" font-size="13" fill="#64748b" '
                     f'text-anchor="middle">No violations on the timeline</text>')
    parts.append("</svg>")
    return "".join(parts)


class ReportGenerator:
    """Renders InvestigationReport into JSON, HTML, and PDF outputs."""

    THUMB_WIDTH = 480

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or str(OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thumb_cache: dict[str, str | None] = {}

    def generate_all(self, report: InvestigationReport) -> dict[str, str]:
        """Generate all configured output formats. Returns dict of format → path."""
        paths = {}
        formats = settings.report.output_format

        if "json" in formats:
            paths["json"] = self.to_json(report)
        if "html" in formats:
            paths["html"] = self.to_html(report)
        if "pdf" in formats:
            try:
                paths["pdf"] = self.to_pdf(report)
            except ImportError:
                logger.warning("reportlab not installed — skipping PDF generation")

        logger.success(f"Report generated: {list(paths.values())}")
        return paths

    # ── JSON ──────────────────────────────────────────────────────────────────

    def to_json(self, report: InvestigationReport) -> str:
        path = self.output_dir / f"{_file_stem(report.report_id)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
        logger.info(f"JSON report saved: {path}")
        return str(path)

    # ── HTML ──────────────────────────────────────────────────────────────────

    def to_html(self, report: InvestigationReport) -> str:
        """Standalone HTML file: embedded screenshots + its own video player (if the video exists)."""
        path = self.output_dir / f"{_file_stem(report.report_id)}.html"
        html = self.render_html(report, interactive=False, include_player=True)
        path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved: {path}")
        return str(path)

    def render_html(
        self,
        report: InvestigationReport,
        interactive: bool = False,
        include_player: bool = False,
    ) -> str:
        """
        interactive    — show Confirm / False-alarm buttons (web UI only)
        include_player — embed a <video> the timestamps seek (standalone file)
        """
        review_by_event = {v.event.event_id: v.review_status for v in report.violations}
        events = []
        for e in report.timeline:
            if not e.potential_violation:
                continue
            events.append({
                "ts": f"{e.timestamp_sec:.2f}",
                "ts_label": _fmt_ts(e.timestamp_sec),
                "description": e.description,
                "objects": ", ".join(o for o in e.objects_involved[:4] if not str(o).startswith("lab_type:")),
                "severity": e.severity,
                "color": severity_color(e.severity),
                "false_alarm": review_by_event.get(e.event_id) == "false_alarm",
                "people": ", ".join(p["label"].split(" · ")[0] for p in _people_labels(e)),
                "repeats": len(e.repeat_timestamps),
            })

        violations = []
        for v in report.violations:
            violations.append({
                "v": v,
                "ts": f"{v.event.timestamp_sec:.2f}",
                "ts_label": _fmt_ts(v.event.timestamp_sec),
                "color": severity_color(v.severity),
                "rule_source": _rule_source(v.rule_violated),
                "thumb": self._thumbnail_data_uri(v.event.frame_path),
                "people": _people_labels(v.event, v),
                "repeats": [{"ts": f"{t:.2f}", "label": _fmt_ts(t)} for t in v.repeat_times()],
                "review_label": REVIEW_LABELS.get(v.review_status, v.review_status),
                "review_color": REVIEW_COLORS.get(v.review_status, REVIEW_COLORS["pending"]),
            })

        template = _env.get_template("report.html.j2")
        return template.render(
            r=report,
            generated_at=report.generated_at.strftime("%Y-%m-%d %H:%M UTC"),
            risk_color=severity_color(report.overall_risk_level),
            frames_sampled=report.metadata.get("frames_sampled", "-"),
            warnings=report.metadata.get("warnings", []) or [],
            review=report.review_counts(),
            events=events,
            violations=violations,
            repeats_merged=report.metadata.get("repeats_merged", 0) or 0,
            chart_svg=Markup(timeline_svg(report)),
            legend=[(s, SEVERITY_COLORS[s]) for s in SEVERITY_ORDER],
            false_alarm_color=FALSE_ALARM_COLOR,
            interactive=interactive,
            include_player=include_player,
            video_uri=self._video_uri(report.video_path) if include_player else None,
        )

    # Backwards-compatible name
    def _render_html(self, report: InvestigationReport) -> str:
        return self.render_html(report)

    def _thumbnail_data_uri(self, frame_path: str | None) -> str | None:
        """Small JPEG of the evidence frame, inlined so the HTML file is self-contained."""
        if not frame_path:
            return None
        if frame_path in self._thumb_cache:
            return self._thumb_cache[frame_path]
        uri = None
        try:
            if Path(frame_path).is_file():
                from PIL import Image
                with Image.open(frame_path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((self.THUMB_WIDTH, self.THUMB_WIDTH))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=72, optimize=True)
                uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            logger.debug(f"Thumbnail failed for {frame_path}: {e}")
        self._thumb_cache[frame_path] = uri
        return uri

    def _video_uri(self, video_path: str) -> str | None:
        """Relative link from the report folder to the video (works when opened locally)."""
        try:
            p = Path(video_path)
            if not p.is_file():
                return None
            rel = os.path.relpath(p.resolve(), self.output_dir.resolve())
            return quote(rel.replace(os.sep, "/"))
        except Exception:
            return None

    # ── PDF ───────────────────────────────────────────────────────────────────

    def to_pdf(self, report: InvestigationReport) -> str:
        """Generate a formal PDF report using reportlab."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
            Image as RLImage,
        )

        path = self.output_dir / f"{_file_stem(report.report_id)}.pdf"

        doc = SimpleDocTemplate(
            str(path), pagesize=A4,
            rightMargin=2 * cm, leftMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        rc = colors.HexColor(severity_color(report.overall_risk_level))

        story = []
        title_style = ParagraphStyle("Title", parent=styles["Title"], textColor=colors.HexColor("#2c3e50"), fontSize=18)
        h2_style = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#3498db"))
        h3_style = ParagraphStyle("H3", parent=styles["Heading3"], textColor=colors.HexColor("#2c3e50"),
                                  keepWithNext=1)
        body_style = styles["BodyText"]
        small_style = ParagraphStyle("Small", parent=body_style, fontSize=8, textColor=colors.HexColor("#64748b"))

        story.append(Paragraph("LabSentinel-RAG Investigation Report", title_style))
        story.append(Paragraph(
            f"Report ID: {_e(report.report_id)} | Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            styles["Normal"]))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph(
            f"Overall Risk Level: <font color='{rc.hexval()}'><b>{_e(report.overall_risk_level.upper())}</b></font>",
            body_style))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("Executive Summary", h2_style))
        story.append(Paragraph(_e(report.incident_summary), body_style))
        for w in report.metadata.get("warnings", []) or []:
            story.append(Paragraph(f"<font color='#b45309'><b>Warning:</b> {_e(w)}</font>", body_style))
        story.append(Spacer(1, 0.4 * cm))

        rv = report.review_counts()
        data = [
            ["Events Detected", "Violations", "Confirmed / False alarm", "Frames Analysed", "Duration"],
            [
                str(len(report.timeline)),
                str(report.total_violations),
                f"{rv['confirmed']} / {rv['false_alarm']}",
                str(report.metadata.get("frames_sampled", "-")),
                _fmt_ts(report.metadata.get("duration_sec", 0) or 0),
            ],
        ]
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph("Violation Timeline", h2_style))
        story.append(self._pdf_timeline(report, doc.width))
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph(f"Confirmed Violations ({report.total_violations})", h2_style))
        for i, v in enumerate(report.violations, 1):
            block = []
            block.append(Paragraph(
                f"{i}. [{_e(v.severity.upper())}] {_e(v.violation_type)} @ {_fmt_ts(v.event.timestamp_sec)}", h3_style))
            if v.review_status != "pending":
                col = REVIEW_COLORS.get(v.review_status, "#64748b")
                block.append(Paragraph(
                    f"<font color='{col}'><b>Review:</b> {_e(REVIEW_LABELS.get(v.review_status, v.review_status))}</font>",
                    ParagraphStyle("Rev", parent=body_style, keepWithNext=1)))

            text = []
            people = _people_labels(v.event, v)
            if people:
                text.append(Paragraph("<b>Person:</b> " + "; ".join(
                    f"{_e(p['label'])} ({_e(p['issues'])})" for p in people), body_style))
            text += [
                Paragraph(f"<b>Rule Violated:</b> {_e(v.rule_violated)}", body_style),
                Paragraph(f"<b>Explanation:</b> {_e(v.explanation)}", body_style),
                Paragraph(f"<b>Immediate Action:</b> {_e(v.immediate_action)}", body_style),
                Paragraph(f"<b>Recommended Action:</b> {_e(v.recommended_action)}", body_style),
            ]
            if v.preventive_measure:
                text.append(Paragraph(f"<b>Preventive Measure:</b> {_e(v.preventive_measure)}", body_style))
            if v.repeat_times():
                text.append(Paragraph(
                    "<b>Also seen at (not flagged again):</b> "
                    + ", ".join(_fmt_ts(t) for t in v.repeat_times()), body_style))

            img = self._pdf_image(v.event.frame_path, RLImage, max_w=6.2 * cm)
            if img is not None:
                cap = Paragraph(f"Frame @ {_fmt_ts(v.event.timestamp_sec)}", small_style)
                tbl = Table([[text, [img, cap]]], colWidths=[doc.width - 6.6 * cm, 6.6 * cm])
                tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
                block.append(tbl)
            else:
                block.extend(text)

            if v.sop_evidence:
                best = v.sop_evidence[0]
                block.append(Paragraph(
                    f"<b>Top SOP Reference:</b> {_e(best.chunk.title)} (similarity: {best.score:.2f})<br/>"
                    f"<i>{_e(best.chunk.content[:250])}...</i>", body_style))
            block.append(Spacer(1, 0.3 * cm))
            story.extend(block)

        doc.build(story)
        logger.info(f"PDF report saved: {path}")
        return str(path)

    @staticmethod
    def _pdf_image(frame_path, RLImage, max_w):
        try:
            if not frame_path or not Path(frame_path).is_file():
                return None
            from PIL import Image
            with Image.open(frame_path) as im:
                w, h = im.size
            return RLImage(frame_path, width=max_w, height=max_w * h / max(w, 1))
        except Exception:
            return None

    @staticmethod
    def _pdf_timeline(report: InvestigationReport, width: float):
        from reportlab.graphics.shapes import Drawing, Line, Circle, String, Rect
        from reportlab.lib import colors

        H = 60
        pad = 10
        axis_y = 22
        dur = _duration(report)
        d = Drawing(width, H)
        x = lambda t: pad + (width - 2 * pad) * min(max(t, 0.0), dur) / dur

        d.add(Rect(pad, axis_y - 2, width - 2 * pad, 4, fillColor=colors.HexColor("#cbd5e1"), strokeColor=None))
        step = _nice_tick_step(dur)
        t = 0.0
        while t <= dur + 1e-6:
            xx = x(t)
            d.add(Line(xx, axis_y - 4, xx, axis_y - 8, strokeColor=colors.HexColor("#94a3b8")))
            d.add(String(xx, axis_y - 17, format_human_time(t), fontSize=6.5, fillColor=colors.HexColor("#64748b"),
                         textAnchor="middle", fontName="Courier"))
            t += step

        for v in report.violations:
            col = colors.HexColor(FALSE_ALARM_COLOR if v.review_status == "false_alarm" else severity_color(v.severity))
            for rt in v.repeat_times():
                d.add(Circle(x(rt), axis_y, 2.5, fillColor=colors.white, strokeColor=col, strokeWidth=1))

        rows_last_x = [-1e9, -1e9, -1e9]
        for v in sorted(report.violations, key=lambda v: v.event.timestamp_sec):
            xx = x(v.event.timestamp_sec)
            row = next((i for i, lx in enumerate(rows_last_x) if xx - lx >= 9), 0)
            rows_last_x[row] = xx
            cy = axis_y + 10 + row * 9
            fa = v.review_status == "false_alarm"
            col = colors.HexColor(FALSE_ALARM_COLOR if fa else severity_color(v.severity))
            d.add(Line(xx, axis_y + 2, xx, cy, strokeColor=col, strokeWidth=0.8))
            d.add(Circle(xx, cy, 3.5, fillColor=None if fa else col, strokeColor=col, strokeWidth=1))
        if not report.violations:
            d.add(String(width / 2, axis_y + 12, "No violations on the timeline", fontSize=8,
                         fillColor=colors.HexColor("#64748b"), textAnchor="middle"))
        return d
