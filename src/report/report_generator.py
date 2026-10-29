"""
Report Generator — Stage 5 of LabSentinel-RAG pipeline.

Generates the final investigation report in three formats:
  1. JSON  — machine-readable, full fidelity
  2. HTML  — human-readable, evidence-linked
  3. PDF   — printable formal report (via reportlab)
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from config.settings import settings, OUTPUT_DIR
from src.utils.models import InvestigationReport, Violation


class ReportGenerator:
    """Renders InvestigationReport into JSON, HTML, and PDF outputs."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or str(OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        path = self.output_dir / f"report_{report.report_id[:8]}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
        logger.info(f"JSON report saved: {path}")
        return str(path)

    # ── HTML ──────────────────────────────────────────────────────────────────

    def to_html(self, report: InvestigationReport) -> str:
        path = self.output_dir / f"report_{report.report_id[:8]}.html"
        html = self._render_html(report)
        path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved: {path}")
        return str(path)

    def _render_html(self, report: InvestigationReport) -> str:
        risk_colors = {
            "low": "#27ae60", "medium": "#f39c12",
            "high": "#e67e22", "critical": "#e74c3c"
        }
        risk_color = risk_colors.get(report.overall_risk_level, "#7f8c8d")

        # Timeline rows
        timeline_rows = ""
        for e in report.timeline:
            badge_color = risk_colors.get(e.severity, "#475569")
            viol_flag   = " ⚠️" if e.potential_violation else ""
            objects_str = ", ".join(e.objects_involved[:4]) if e.objects_involved else "—"
            timeline_rows += f"""
            <tr>
              <td style="white-space:nowrap;color:#60a5fa;font-weight:600">{e.timestamp_sec:.1f}s</td>
              <td>{e.description}{viol_flag}</td>
              <td style="color:#64748b;font-size:0.85em">{objects_str}</td>
              <td><span class="badge" style="background:{badge_color}">{e.severity.upper()}</span></td>
            </tr>"""

        # Violation cards
        violation_cards = ""
        for v in report.violations:
            sop_quotes = ""
            for ev in v.sop_evidence[:2]:
                sop_quotes += f"""
                <div class="evidence-box">
                  <strong>{ev.chunk.title}</strong> (score: {ev.score:.2f})<br>
                  {ev.chunk.content[:350]}...
                </div>"""

            incident_quotes = ""
            for ev in v.similar_incidents[:2]:
                incident_quotes += f"""
                <div class="evidence-box">
                  <strong>{ev.chunk.title}</strong> (score: {ev.score:.2f})<br>
                  {ev.chunk.content[:300]}...
                </div>"""

            badge_color = risk_colors.get(v.severity, "#7f8c8d")
            violation_cards += f"""
            <div class="violation-card">
              <div class="v-header">
                <span class="badge" style="background:{badge_color}">{v.severity.upper()}</span>
                <h3>{v.violation_type}</h3>
                <span class="v-timestamp">⏱ {v.event.timestamp_sec:.1f}s</span>
              </div>
              <div class="rule-box">📋 <strong>Rule Violated:</strong> {v.rule_violated}</div>
              <p class="explanation">{v.explanation}</p>
              <div class="actions">
                <div class="action-box immediate">
                  <strong>⚡ Immediate</strong><span>{v.immediate_action}</span>
                </div>
                <div class="action-box recommended">
                  <strong>🔧 Recommended</strong><span>{v.recommended_action}</span>
                </div>
                <div class="action-box preventive">
                  <strong>🛡 Preventive</strong><span>{v.preventive_measure}</span>
                </div>
              </div>
              <h4>SOP Evidence</h4>
              {sop_quotes or '<div class="evidence-box">No SOP evidence retrieved.</div>'}
              <h4>Similar Past Incidents</h4>
              {incident_quotes or '<div class="evidence-box">No similar incidents found.</div>'}
            </div>"""

        key_findings_html = "".join(
            f"<li>{f}</li>" for f in report.key_findings
        )
        immediate_actions_html = "".join(
            f"<li>{a}</li>" for a in report.immediate_actions_required
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LabSentinel-RAG Investigation Report</title>
  <style>
    :root {{
      --bg: #0f172a; --surface: #1e293b; --border: #334155;
      --primary: #3b82f6; --text: #e2e8f0; --muted: #94a3b8;
      --red: #ef4444; --orange: #f97316; --yellow: #eab308; --green: #22c55e;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      background: var(--bg); color: var(--text);
      padding: 24px; max-width: 1100px; margin: 0 auto; line-height: 1.6;
    }}
    .header {{
      background: linear-gradient(135deg, #0f172a, #1e3a5f);
      border: 1px solid #1e3a5f; border-radius: 14px;
      padding: 24px 30px; margin-bottom: 24px;
    }}
    .header h1 {{ font-size: 1.8em; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px; }}
    .header p  {{ color: var(--muted); margin-top: 4px; font-size: 0.88em; }}
    h2 {{ color: #60a5fa; font-size: 1.1em; font-weight: 700; margin: 28px 0 12px;
          text-transform: uppercase; letter-spacing: 0.8px; }}
    h3 {{ color: var(--text); font-size: 1.0em; font-weight: 600; }}
    h4 {{ color: var(--muted); font-size: 0.85em; font-weight: 600; margin: 12px 0 6px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .summary-box {{
      background: var(--surface); border-left: 3px solid var(--primary);
      padding: 14px 18px; border-radius: 8px; margin-bottom: 20px;
      color: var(--text); font-size: 0.95em;
    }}
    .stats {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px; margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 16px; text-align: center;
    }}
    .stat-card .num {{
      font-size: 2.2em; font-weight: 800; color: var(--primary); line-height: 1;
    }}
    .stat-card .lbl {{ font-size: 0.75em; color: var(--muted); margin-top: 4px; }}
    .risk-pill {{
      display: inline-block; padding: 6px 20px; border-radius: 20px;
      font-weight: 700; font-size: 0.9em; color: white; background: {risk_color};
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.88em; }}
    th {{
      background: #1a2744; color: #93c5fd; padding: 10px 12px;
      text-align: left; font-weight: 600; font-size: 0.8em;
      text-transform: uppercase; letter-spacing: 0.5px;
    }}
    td {{ padding: 9px 12px; border-bottom: 1px solid var(--border); color: var(--text); }}
    tr:hover td {{ background: rgba(59,130,246,0.05); }}
    .badge {{
      display: inline-block; padding: 2px 10px; border-radius: 10px;
      color: white; font-size: 0.75em; font-weight: 700;
    }}
    .violation-card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 20px; margin: 16px 0;
    }}
    .v-header {{
      display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap;
    }}
    .v-header h3 {{ font-size: 1.05em; }}
    .v-timestamp {{
      background: rgba(59,130,246,0.15); color: #60a5fa;
      padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600;
    }}
    .rule-box {{
      background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2);
      border-radius: 6px; padding: 10px 14px; margin: 10px 0; font-size: 0.88em;
      color: #93c5fd;
    }}
    .explanation {{ color: var(--muted); font-size: 0.9em; margin: 8px 0; }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }}
    .action-box {{ flex: 1; min-width: 180px; padding: 12px 14px; border-radius: 8px; font-size: 0.85em; }}
    .immediate  {{ background: rgba(239,68,68,0.1);  border-left: 3px solid #ef4444; }}
    .recommended{{ background: rgba(249,115,22,0.1); border-left: 3px solid #f97316; }}
    .preventive {{ background: rgba(34,197,94,0.1);  border-left: 3px solid #22c55e; }}
    .action-box strong {{ display: block; margin-bottom: 4px; color: var(--text); }}
    .action-box span {{ color: var(--muted); }}
    .evidence-box {{
      background: rgba(255,255,255,0.03); border: 1px solid var(--border);
      border-radius: 6px; padding: 10px 14px; margin: 6px 0; font-size: 0.83em; color: var(--muted);
    }}
    .evidence-box strong {{ color: #93c5fd; }}
    ul li {{ margin: 6px 0; color: var(--muted); }}
    ul li::marker {{ color: var(--primary); }}
    .footer {{
      text-align: center; color: #475569; margin-top: 40px; font-size: 0.8em;
      border-top: 1px solid var(--border); padding-top: 16px;
    }}
  </style>
</head>
<body>

<div class="header">
  <h1>🔬 LabSentinel-RAG</h1>
  <p>Multimodal Evidence-Grounded Laboratory Incident Investigation Report &nbsp;·&nbsp;
     Report ID: {report.report_id[:16]}… &nbsp;·&nbsp;
     {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}</p>
</div>

<div class="stats">
  <div class="stat-card"><div class="num">{len(report.timeline)}</div><div class="lbl">Events Detected</div></div>
  <div class="stat-card"><div class="num">{report.total_violations}</div><div class="lbl">Violations</div></div>
  <div class="stat-card"><div class="num">{report.metadata.get('frames_sampled','-')}</div><div class="lbl">Frames Analysed</div></div>
  <div class="stat-card">
    <div style="font-size:0.75em;color:var(--muted);margin-bottom:6px">RISK LEVEL</div>
    <span class="risk-pill">{report.overall_risk_level.upper()}</span>
  </div>
</div>

<h2>Executive Summary</h2>
<div class="summary-box">{report.incident_summary}</div>

<h2>Key Findings</h2>
<ul>{"".join(f"<li>{f}</li>" for f in report.key_findings) or "<li>See violation details below.</li>"}</ul>

<h2>Immediate Actions Required</h2>
<ul>{"".join(f"<li>{a}</li>" for a in report.immediate_actions_required) or "<li>Review violation details and implement corrective actions.</li>"}</ul>

<h2>Incident Timeline</h2>
<table>
  <thead><tr><th>Time</th><th>Description</th><th>Objects / Materials</th><th>Severity</th></tr></thead>
  <tbody>{timeline_rows}</tbody>
</table>

<h2>Confirmed Violations ({report.total_violations})</h2>
{violation_cards or '<p style="color:var(--muted)">No violations confirmed in this incident.</p>'}

<div class="footer">
  Generated by LabSentinel-RAG &nbsp;·&nbsp; {report.video_path}
</div>
</body>
</html>"""

    # ── PDF ───────────────────────────────────────────────────────────────────

    def to_pdf(self, report: InvestigationReport) -> str:
        """Generate a formal PDF report using reportlab."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )

        path = self.output_dir / f"report_{report.report_id[:8]}.pdf"

        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        risk_color_map = {
            "low": colors.green, "medium": colors.orange,
            "high": colors.darkorange, "critical": colors.red,
        }
        rc = risk_color_map.get(report.overall_risk_level, colors.grey)

        story = []
        title_style = ParagraphStyle("Title", parent=styles["Title"], textColor=colors.HexColor("#2c3e50"), fontSize=18)
        h2_style = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#3498db"))
        h3_style = ParagraphStyle("H3", parent=styles["Heading3"], textColor=colors.HexColor("#2c3e50"))
        body_style = styles["BodyText"]

        # Header
        story.append(Paragraph("LabSentinel-RAG Investigation Report", title_style))
        story.append(Paragraph(
            f"Report ID: {report.report_id} | Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            styles["Normal"]
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 0.4 * cm))

        # Risk level
        story.append(Paragraph(
            f"Overall Risk Level: <font color='{rc.hexval() if hasattr(rc,'hexval') else '#e74c3c'}'><b>{report.overall_risk_level.upper()}</b></font>",
            body_style
        ))
        story.append(Spacer(1, 0.3 * cm))

        # Summary
        story.append(Paragraph("Executive Summary", h2_style))
        story.append(Paragraph(report.incident_summary, body_style))
        story.append(Spacer(1, 0.4 * cm))

        # Stats table
        data = [
            ["Events Detected", "Violations Confirmed", "Frames Analysed", "Duration"],
            [
                str(len(report.timeline)),
                str(report.total_violations),
                str(report.metadata.get("frames_sampled", "-")),
                f"{report.metadata.get('duration_sec', '-')}s",
            ],
        ]
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))

        # Violations
        story.append(Paragraph(f"Confirmed Violations ({report.total_violations})", h2_style))
        for i, v in enumerate(report.violations, 1):
            story.append(Paragraph(f"{i}. [{v.severity.upper()}] {v.violation_type} @ {v.event.timestamp_sec:.1f}s", h3_style))
            story.append(Paragraph(f"<b>Rule Violated:</b> {v.rule_violated}", body_style))
            story.append(Paragraph(f"<b>Explanation:</b> {v.explanation}", body_style))
            story.append(Paragraph(f"<b>Immediate Action:</b> {v.immediate_action}", body_style))
            story.append(Paragraph(f"<b>Recommended Action:</b> {v.recommended_action}", body_style))

            if v.sop_evidence:
                best = v.sop_evidence[0]
                story.append(Paragraph(
                    f"<b>Top SOP Reference:</b> {best.chunk.title} (similarity: {best.score:.2f})<br/>"
                    f"<i>{best.chunk.content[:250]}...</i>",
                    body_style
                ))
            story.append(Spacer(1, 0.3 * cm))

        doc.build(story)
        logger.info(f"PDF report saved: {path}")
        return str(path)
