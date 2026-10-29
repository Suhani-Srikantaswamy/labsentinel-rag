#!/usr/bin/env python3
"""
LabSentinel-RAG — Command Line Interface

Commands:
  build-kb      Build (or update) the ChromaDB knowledge base from your documents
  investigate   Run the full investigation pipeline on an incident video
  stats         Show what is currently indexed in the knowledge base
  reset-kb      Wipe the knowledge base and start fresh
  demo          Launch the Gradio web interface
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    name="labsentinel",
    help="LabSentinel-RAG: Multimodal Evidence-Grounded Lab Incident Investigation",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ── build-kb ──────────────────────────────────────────────────────────────────

@app.command()
def build_kb(
    sop_dir: str = typer.Option(
        None, "--sop-dir", "-s",
        help="Path to your SOP documents folder (PDF/DOCX/TXT). "
             "Defaults to data/sops/",
    ),
    manual_dir: str = typer.Option(
        None, "--manual-dir", "-m",
        help="Path to equipment manuals folder. Defaults to data/manuals/",
    ),
    incident_dir: str = typer.Option(
        None, "--incident-dir", "-i",
        help="Path to historical incident records folder. Defaults to data/incidents/",
    ),
    reset: bool = typer.Option(
        False, "--reset",
        help="Wipe existing index before building.",
    ),
):
    """
    Build the vector knowledge base from your SOP, manual and incident documents.

    Run this ONCE before investigating videos. Re-run whenever you add new documents.

    Example:
      python main.py build-kb
      python main.py build-kb --sop-dir /my/sops --manual-dir /my/manuals
    """
    from src.ingestion.knowledge_base_builder import KnowledgeBaseBuilder
    from src.ingestion.document_loader import DocumentLoader
    import config.settings as cfg

    # Apply directory overrides directly to the module-level variables
    if sop_dir:
        cfg.SOP_DIR = Path(sop_dir)
        console.print(f"  SOP dir   : {cfg.SOP_DIR}")
    if manual_dir:
        cfg.MANUAL_DIR = Path(manual_dir)
        console.print(f"  Manual dir: {cfg.MANUAL_DIR}")
    if incident_dir:
        cfg.INCIDENT_DIR = Path(incident_dir)
        console.print(f"  Incident dir: {cfg.INCIDENT_DIR}")

    builder = KnowledgeBaseBuilder()

    if reset:
        console.print("[yellow]Resetting existing knowledge base…[/yellow]")
        builder.reset()

    console.print("\n[bold blue]Building LabSentinel-RAG Knowledge Base…[/bold blue]")
    loader = DocumentLoader()
    chunks = loader.load_all()

    if not chunks:
        console.print(
            "[red]No documents found.[/red] "
            "Add files to data/sops/, data/manuals/, data/incidents/ "
            "(or use --sop-dir etc.) and try again."
        )
        raise typer.Exit(1)

    counts = builder.build(chunks=chunks)

    table = Table(title="Knowledge Base — Chunks Indexed")
    table.add_column("Collection", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for k, v in counts.items():
        table.add_row(k, str(v))
    console.print(table)
    console.print("[bold green]✓ Knowledge base ready[/bold green]")


# ── investigate ───────────────────────────────────────────────────────────────

@app.command()
def investigate(
    video: str = typer.Argument(..., help="Path to the incident video file"),
    report_id: str = typer.Option(
        None, "--report-id",
        help="Custom report identifier (auto-generated if omitted)",
    ),
    output_dir: str = typer.Option(
        None, "--output-dir", "-o",
        help="Directory for the generated reports. Defaults to outputs/",
    ),
    sample_rate: int = typer.Option(
        None, "--sample-rate",
        help="Analyse 1 frame every N seconds (default from settings)",
    ),
    max_frames: int = typer.Option(
        None, "--max-frames",
        help="Maximum frames to analyse (default from settings)",
    ),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Skip PDF generation"),
    no_html: bool = typer.Option(False, "--no-html", help="Skip HTML generation"),
):
    """
    Run the full LabSentinel-RAG investigation on an incident video.

    Requires:
      1. OPENAI_API_KEY set in your .env file
      2. Knowledge base built (run `python main.py build-kb` first)

    Example:
      python main.py investigate data/videos/incident.mp4
      python main.py investigate /path/to/video.mp4 --output-dir /my/reports
    """
    from src.pipeline import LabSentinelPipeline
    from config.settings import settings

    if not Path(video).exists():
        console.print(f"[red]Video not found:[/red] {video}")
        raise typer.Exit(1)

    if not settings.llm.api_key:
        console.print(
            "[red]OpenAI API key missing.[/red] "
            "Set OPENAI_API_KEY in your .env file."
        )
        raise typer.Exit(1)

    # Apply format overrides
    if no_pdf and "pdf" in settings.report.output_format:
        settings.report.output_format.remove("pdf")
    if no_html and "html" in settings.report.output_format:
        settings.report.output_format.remove("html")

    pipeline = LabSentinelPipeline()
    report = pipeline.run(
        video_path  = video,
        report_id   = report_id,
        output_dir  = output_dir,
        sample_rate = sample_rate,
        max_frames  = max_frames,
        verbose     = True,
    )

    console.print(
        Panel(
            f"[bold]Violations  :[/bold] {report.total_violations}\n"
            f"[bold]Risk Level  :[/bold] {report.overall_risk_level.upper()}\n"
            f"[bold]Events Found:[/bold] {len(report.timeline)}",
            title="Investigation Summary",
            border_style="green",
        )
    )


# ── stats ─────────────────────────────────────────────────────────────────────

@app.command()
def stats():
    """Show how many documents are currently indexed in the knowledge base."""
    from src.ingestion.knowledge_base_builder import KnowledgeBaseBuilder
    builder = KnowledgeBaseBuilder()
    data    = builder.get_stats()
    total   = sum(data.values())

    table = Table(title="LabSentinel-RAG Knowledge Base")
    table.add_column("Collection", style="cyan")
    table.add_column("Chunks Indexed", justify="right", style="green")
    for k, v in data.items():
        table.add_row(k.upper(), str(v))
    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total}[/bold]")
    console.print(table)

    if total == 0:
        console.print(
            "[yellow]Knowledge base is empty.[/yellow] "
            "Run `python main.py build-kb` to index your documents."
        )


# ── reset-kb ──────────────────────────────────────────────────────────────────

@app.command()
def reset_kb():
    """Wipe the entire knowledge base (all indexed documents are removed)."""
    confirm = typer.confirm(
        "This will delete all indexed documents. Continue?", default=False
    )
    if not confirm:
        console.print("Aborted.")
        raise typer.Exit(0)

    from src.ingestion.knowledge_base_builder import KnowledgeBaseBuilder
    builder = KnowledgeBaseBuilder()
    builder.reset()
    console.print("[green]Knowledge base cleared.[/green]")


# ── demo ──────────────────────────────────────────────────────────────────────

@app.command()
def demo(
    port: int = typer.Option(7860, "--port", "-p", help="Port to run the web UI on"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio link"),
):
    """
    Launch the Gradio web interface.

    Open http://localhost:7860 in your browser after running.
    """
    try:
        from app import create_demo
        ui = create_demo()
        ui.launch(server_port=port, share=share, server_name="0.0.0.0")
    except ImportError as e:
        console.print(f"[red]Cannot launch UI:[/red] {e}")
        console.print("Install: pip install gradio")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
