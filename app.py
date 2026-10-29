"""
LabSentinel-RAG — Production Web Application
"""
from __future__ import annotations
import hashlib, json, os, shutil, tempfile, threading, time
from pathlib import Path
from queue import Queue, Empty
import gradio as gr
from loguru import logger

# ── Helpers ───────────────────────────────────────────────────────────────────

def _api_ok():
    from src.utils.llm_client import get_provider_info
    i = get_provider_info()
    return i["configured"], i["status_msg"]

def _extract_text(path: str) -> str:
    p = Path(path); ext = p.suffix.lower()
    try:
        if ext == ".pdf":
            import pdfplumber
            parts = []
            with pdfplumber.open(str(p)) as pdf:
                for pg in pdf.pages:
                    t = pg.extract_text()
                    if t: parts.append(t)
            return "\n".join(parts)
        elif ext in (".docx", ".doc"):
            from docx import Document
            return "\n".join(x.text for x in Document(str(p)).paragraphs if x.text.strip())
        else:
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Text extraction failed {p.name}: {e}"); return ""

def _combine_manuals(files):
    if not files: return ""
    parts = []
    for f in files:
        if f is None: continue
        t = _extract_text(str(f))
        if t.strip(): parts.append(f"=== {Path(str(f)).name} ===\n{t}")
    return "\n\n".join(parts)

def _url_cache_path(url: str) -> Path:
    """Return a stable cache path for a URL so we don't re-download."""
    from config.settings import VIDEO_DIR
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    return VIDEO_DIR / f"cached_{h}.mp4"

def _download_url(url: str, progress_cb=None):
    """
    Download video from URL. Caches to data/videos/ so re-runs are instant.
    Returns (path, tmp_dir_or_None, error_or_None)
    """
    # Check cache first
    cached = _url_cache_path(url)
    if cached.exists() and cached.stat().st_size > 10_000:
        logger.info(f"Using cached video: {cached}")
        if progress_cb: progress_cb(1.0, f"Using cached video ({cached.stat().st_size//1024//1024} MB)")
        return str(cached), None, None

    tmp = tempfile.mkdtemp()
    try:
        import yt_dlp

        class ProgressHook:
            def __call__(self, d):
                if d['status'] == 'downloading' and progress_cb:
                    total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        pct = downloaded / total
                        mb_done = downloaded / 1024 / 1024
                        mb_total = total / 1024 / 1024
                        progress_cb(pct * 0.95, f"Downloading {mb_done:.1f} / {mb_total:.1f} MB …")

        opts = {
            "format": (
                "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]"
                "/bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]"
                "/bestvideo[height<=480]+bestaudio"
                "/best[height<=480]/best"
            ),
            "outtmpl":                       str(Path(tmp) / "video.%(ext)s"),
            "merge_output_format":           "mp4",
            "quiet":                         True,
            "no_warnings":                   True,
            "noplaylist":                    True,
            "ffmpeg_location":               "/opt/homebrew/bin/ffmpeg",
            "concurrent_fragment_downloads": 8,
            "progress_hooks":                [ProgressHook()],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        for ext in ("mp4", "webm", "mkv", "avi", "mov"):
            files = list(Path(tmp).glob(f"*.{ext}"))
            if files:
                # Save to cache for next time
                shutil.copy(str(files[0]), str(cached))
                if progress_cb: progress_cb(1.0, f"Download complete ({cached.stat().st_size//1024//1024} MB)")
                return str(cached), tmp, None

        return None, tmp, "Download completed but no video file found."
    except Exception as e:
        return None, tmp, f"Download failed: {e}"

def _run_pipeline(video_path, manual_text, sample_rate, max_frames, chunk_sec, pq, holder):
    try:
        from src.pipeline import LabSentinelPipeline
        pipeline = LabSentinelPipeline()
        report = pipeline.run(
            video_path         = video_path,
            manual_text        = manual_text,
            sample_rate        = sample_rate or None,
            max_frames         = max_frames  or None,
            chunk_duration_sec = chunk_sec   or None,
            on_progress        = lambda p, m: pq.put((p, m)),
            verbose            = False,
        )
        holder["report"] = report
    except Exception as e:
        logger.exception(e); holder["error"] = str(e)
    finally:
        pq.put(None)

# ── Main investigation callback ───────────────────────────────────────────────

def investigate(video_file, video_url, manual_files,
                sample_rate, max_frames, chunk_sec,
                progress=gr.Progress(track_tqdm=False)):

    ok, msg = _api_ok()
    if not ok:
        return _err("API Key Missing", [
            "FREE option: Get a Google Gemini key at https://aistudio.google.com",
            "Add to .env:  GOOGLE_API_KEY=your-key",
            "Restart the app: python app.py",
        ]), "{}", None, _stat("❌", "No API key"), ""

    # Resolve video
    video_path = None; tmp_dir = None
    if video_url and video_url.strip():
        progress(0.01, desc="Checking download cache …")
        cached = _url_cache_path(video_url.strip())
        if cached.exists() and cached.stat().st_size > 10_000:
            video_path = str(cached)
            progress(0.05, desc=f"✅ Using cached video ({cached.stat().st_size//1024//1024} MB) — skipping download")
        else:
            progress(0.02, desc="Downloading video …")
            video_path, tmp_dir, err = _download_url(
                video_url.strip(),
                progress_cb=lambda p, m: progress(0.02 + p * 0.10, desc=m)
            )
            if err:
                return _err("Download Failed", [err, "Try uploading the video file directly instead."]), "{}", None, _stat("❌", "Download failed"), ""
    elif video_file is not None:
        video_path = str(video_file)
    else:
        return _err("No Video", ["Upload a video file or paste a URL above."]), "{}", None, _stat("⚠️", "No input"), ""

    # Extract manual text
    progress(0.13, desc="Reading lab manual …")
    manual_text = _combine_manuals(manual_files) if manual_files else ""
    manual_label = (f"{len([f for f in (manual_files or []) if f])} manual file(s), {len(manual_text):,} chars"
                    if manual_text else "No manual — universal rules applied")

    # Run pipeline
    progress(0.15, desc="Starting investigation …")
    pq = Queue(); holder = {}
    t  = threading.Thread(target=_run_pipeline,
                          args=(video_path, manual_text, sample_rate, max_frames, chunk_sec, pq, holder),
                          daemon=True)
    t.start()

    while True:
        try:
            item = pq.get(timeout=900)
        except Empty:
            break
        if item is None: break
        pct, msg2 = item
        progress(min(0.15 + pct * 0.80, 0.96), desc=msg2)
        time.sleep(0.02)
    t.join(timeout=15)

    if "error" in holder:
        return _err("Investigation Failed", [holder["error"]]), "{}", None, _stat("❌", holder["error"][:80]), ""

    report = holder.get("report")
    if not report:
        return _err("No Result", ["Pipeline returned no report."]), "{}", None, _stat("❌", "No report"), ""

    progress(0.97, desc="Rendering report …")
    from src.report.report_generator import ReportGenerator
    gen  = ReportGenerator()
    html = gen._render_html(report)
    js   = json.dumps(report.model_dump(mode="json"), indent=2, default=str)
    pdf  = None
    try: pdf = gen.to_pdf(report)
    except Exception as e: logger.warning(f"PDF failed: {e}")

    if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)

    progress(1.0, desc="Done!")
    r   = report.overall_risk_level
    status_html = _stat("✅",
        f"Complete — <b>{report.total_violations} violation(s)</b> | "
        f"Risk: <b style='color:{_risk_col(r)}'>{r.upper()}</b> | "
        f"{len(report.timeline)} events | {report.metadata.get('frames_sampled','?')} frames | {manual_label}")
    return html, js, pdf, status_html, _summary_md(report, manual_label)

# ── KB builder ────────────────────────────────────────────────────────────────

def build_kb(sop_f, man_f, inc_f, sop_d, man_d, inc_d, reset, progress=gr.Progress()):
    progress(0.05, desc="Loading documents …")
    try:
        from src.ingestion.document_loader import DocumentLoader
        from src.ingestion.knowledge_base_builder import KnowledgeBaseBuilder
        import config.settings as cfg
        loader = DocumentLoader(); builder = KnowledgeBaseBuilder()
        if reset: builder.reset()
        chunks = []
        for files, dest, stype in [(sop_f or [], cfg.SOP_DIR, "sop"),
                                    (man_f or [], cfg.MANUAL_DIR, "manual"),
                                    (inc_f or [], cfg.INCIDENT_DIR, "incident")]:
            for f in (files if isinstance(files, list) else [files]):
                if f:
                    dst = dest / Path(str(f)).name
                    shutil.copy(str(f), str(dst))
                    chunks.extend(loader.load_file(dst, stype))
        for dpath, stype in [(sop_d.strip(), "sop"), (man_d.strip(), "manual"), (inc_d.strip(), "incident")]:
            if dpath and Path(dpath).exists():
                chunks.extend(loader.load_directory(Path(dpath), stype))
        if not chunks: return "⚠ No documents found."
        progress(0.6, desc=f"Embedding {len(chunks)} chunks …")
        counts = builder.build(chunks=chunks)
        stats  = builder.get_stats()
        progress(1.0)
        lines = [f"### ✅ Knowledge base updated — {sum(counts.values())} chunks indexed\n",
                 "| Collection | Chunks |", "|---|---|"]
        for k, v in stats.items(): lines.append(f"| {k.title()} | {v} |")
        return "\n".join(lines)
    except Exception as e:
        logger.exception(e); return f"❌ Error: {e}"

def get_status():
    from src.utils.llm_client import get_provider_info
    from config.settings import settings
    pi = get_provider_info()
    try:
        from src.ingestion.knowledge_base_builder import KnowledgeBaseBuilder
        stats = KnowledgeBaseBuilder().get_stats()
    except: stats = {}
    total = sum(stats.values())
    lines = ["| Check | Status |", "|---|---|",
             f"| AI Provider | {pi['status_msg']} |",
             f"| Knowledge Base | {'✅ ' + str(total) + ' chunks' if total else '⚠ Empty'} |"]
    for k, v in stats.items(): lines.append(f"| — {k.title()} | {v} |")
    lines += ["", "### Settings",
              f"| Model | `{pi['model']}` ({'FREE' if pi['is_free'] else 'paid'}) |",
              f"| Frame rate | every `{settings.video.frame_sample_rate}s` |",
              f"| Max frames | `{settings.video.max_frames}` |",
              f"| Embeddings | `{settings.rag.embedding_model}` (local) |"]
    return "\n".join(lines)

# ── Template helpers ──────────────────────────────────────────────────────────

def _risk_col(r): return {"low":"#16a34a","medium":"#d97706","high":"#ea580c","critical":"#dc2626"}.get(r,"#64748b")

def _stat(icon, msg):
    return f'<div style="padding:10px 16px;border-radius:8px;background:#1e293b;color:#e2e8f0;font-size:0.92em">{icon} {msg}</div>'

def _summary_md(report, manual_label):
    r = report.overall_risk_level
    emoji = {"low":"🟢","medium":"🟡","high":"🟠","critical":"🔴"}.get(r,"⚪")
    lines = [f"## {emoji} Investigation Complete\n",
             f"| | |", "|---|---|",
             f"| **Risk Level** | **{r.upper()}** |",
             f"| **Violations** | {report.total_violations} |",
             f"| **Events** | {len(report.timeline)} |",
             f"| **Frames** | {report.metadata.get('frames_sampled','?')} |",
             f"| **Duration** | {report.metadata.get('duration_sec','?')}s |",
             f"| **Manual** | {manual_label} |",
             f"\n**{report.incident_summary}**"]
    if report.key_findings:
        lines.append("\n**Key Findings:**")
        for f in report.key_findings: lines.append(f"- {f}")
    return "\n".join(lines)

def _err(title, tips):
    items = "".join(f"<li style='margin:4px 0'>{t}</li>" for t in tips)
    return f"""
    <div style="background:#1e293b;border:1px solid #ef4444;border-radius:12px;padding:28px 32px;margin:8px 0">
      <h3 style="color:#ef4444;margin:0 0 12px">⚠ {title}</h3>
      <ul style="color:#cbd5e1;margin:0;padding-left:20px;line-height:1.8">{items}</ul>
    </div>"""

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; }
body, .gradio-container { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important; }
footer { display: none !important; }
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; background: #0f172a !important; }

/* Hero */
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2a1e 100%);
  border-radius: 16px; padding: 32px 40px 24px; margin-bottom: 20px;
  border: 1px solid #1e3a5f;
}
.hero h1 { font-size: 2.4em; font-weight: 800; margin: 0 0 6px; color: #f8fafc; letter-spacing: -1px; }
.hero p  { color: #94a3b8; margin: 0 0 14px; font-size: 1.05em; }
.pill {
  display: inline-block; background: rgba(59,130,246,0.15);
  border: 1px solid rgba(59,130,246,0.3); border-radius: 20px;
  padding: 3px 14px; font-size: 0.78em; color: #93c5fd; margin-right: 6px; margin-top: 4px;
}

/* Panels */
.panel {
  background: #1e293b; border: 1px solid #334155; border-radius: 12px;
  padding: 20px 22px; margin-bottom: 14px;
}
.panel-title {
  font-size: 0.7em; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1.2px; color: #64748b; margin: 0 0 12px;
}

/* Run button */
#run-btn {
  background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
  color: #fff !important; font-size: 1.05em !important; font-weight: 700 !important;
  border-radius: 10px !important; border: none !important; padding: 14px !important;
  box-shadow: 0 4px 20px rgba(37,99,235,0.4) !important; transition: all 0.2s !important;
  letter-spacing: 0.3px !important;
}
#run-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(37,99,235,0.5) !important; }

/* PDF button */
.pdf-download button {
  background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
  color: white !important; font-weight: 700 !important; border-radius: 8px !important; border: none !important;
}

/* Tabs */
.tab-nav { border-bottom: 1px solid #334155 !important; }
.tab-nav button { color: #94a3b8 !important; font-weight: 500 !important; }
.tab-nav button.selected { color: #60a5fa !important; border-bottom-color: #3b82f6 !important; }

/* Inputs */
.gradio-textbox textarea, .gradio-textbox input { background: #0f172a !important; border-color: #334155 !important; color: #e2e8f0 !important; border-radius: 8px !important; }
.gradio-number input { background: #0f172a !important; border-color: #334155 !important; color: #e2e8f0 !important; border-radius: 8px !important; }

/* Accordion */
.gradio-accordion { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 8px !important; }

/* Markdown */
.gradio-markdown { color: #e2e8f0 !important; }
.gradio-markdown h2 { color: #f1f5f9 !important; }
.gradio-markdown table { border-collapse: collapse; width: 100%; }
.gradio-markdown th { background: #334155 !important; color: #f1f5f9 !important; padding: 8px 12px !important; }
.gradio-markdown td { border: 1px solid #334155 !important; padding: 7px 12px !important; color: #cbd5e1 !important; }

/* File upload */
.gradio-file { background: #0f172a !important; border: 2px dashed #334155 !important; border-radius: 10px !important; }
.gradio-video { background: #0f172a !important; border: 2px dashed #334155 !important; border-radius: 10px !important; }

/* Placeholder */
.placeholder-box {
  background: #1e293b; border: 2px dashed #334155; border-radius: 16px;
  padding: 60px 40px; text-align: center; color: #64748b;
}
"""

PLACEHOLDER = """
<div class="placeholder-box">
  <div style="font-size:3em;margin-bottom:16px">🔬</div>
  <h2 style="color:#f1f5f9;font-size:1.6em;margin:0 0 10px;font-weight:700">Ready to Investigate</h2>
  <p style="max-width:420px;margin:0 auto 24px;line-height:1.7;color:#94a3b8">
    Upload your <strong style="color:#e2e8f0">lab manual</strong> and provide an
    <strong style="color:#e2e8f0">incident video</strong>, then click
    <em style="color:#60a5fa">Run Investigation</em>.
  </p>
  <div style="display:inline-flex;gap:10px;flex-wrap:wrap;justify-content:center;
              background:#0f172a;border-radius:12px;padding:14px 20px;border:1px solid #334155;font-size:0.85em;color:#94a3b8">
    <span>📋 Upload lab manual</span>
    <span style="color:#334155">│</span>
    <span>🎬 Upload video or paste URL</span>
    <span style="color:#334155">│</span>
    <span>🚀 Click Run</span>
  </div>
  <p style="margin:18px 0 0;font-size:0.8em;color:#475569">
    Works with any lab type · Any video length · No hardcoded rules · 100% dynamic analysis
  </p>
</div>"""

# ── Build UI ──────────────────────────────────────────────────────────────────

def create_app():
    with gr.Blocks(title="LabSentinel-RAG") as app:

        # Hero
        gr.HTML(f"""
        <div class="hero">
          <h1>🔬 LabSentinel-RAG</h1>
          <p>AI-powered laboratory safety incident investigation. Upload any lab manual + any video → instant violation report.</p>
          <span class="pill">Gemini Vision</span>
          <span class="pill">RAG Retrieval</span>
          <span class="pill">Any Lab Type</span>
          <span class="pill">Hours-long Video</span>
          <span class="pill">Exact Timestamps</span>
        </div>
        """)

        with gr.Tabs(elem_classes="tab-nav"):

            # ── INVESTIGATE ────────────────────────────────────────────────────
            with gr.TabItem("🔍  Investigate"):
                with gr.Row(equal_height=False, variant="panel"):

                    # Left panel
                    with gr.Column(scale=5, min_width=360):
                        gr.HTML('<div class="panel"><p class="panel-title">Step 1 — Lab Manual (Optional)</p>')
                        manual_in = gr.File(
                            label="Upload Lab Manual / SOP / Safety Guidelines",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md"],
                        )
                        gr.HTML(
                            '<p style="color:#64748b;font-size:0.8em;margin:6px 0 0">'
                            'PDF, DOCX or TXT. The AI reads it and applies its specific rules to the video. '
                            'Without a manual, universal safety rules are used.</p></div>'
                        )

                        gr.HTML('<div class="panel"><p class="panel-title">Step 2 — Incident Video</p>')
                        video_in = gr.Video(label="Upload Video File", height=200)
                        url_in   = gr.Textbox(
                            label="Or paste a video URL (YouTube, direct MP4, etc.)",
                            placeholder="https://www.youtube.com/watch?v=saXFQR86ziM",
                            lines=1,
                        )
                        gr.HTML(
                            '<p style="color:#64748b;font-size:0.8em;margin:6px 0 0">'
                            '⚡ Downloaded videos are cached — re-running the same URL is instant.</p></div>'
                        )

                        with gr.Accordion("⚙  Advanced Settings", open=False):
                            gr.HTML('<p style="color:#64748b;font-size:0.8em;margin:0 0 10px">Leave at 0 to use defaults from .env</p>')
                            with gr.Row():
                                sr_in = gr.Number(label="Frame rate (s)", value=0, minimum=0, maximum=60, precision=0,
                                                  info="1 frame every N seconds")
                                mf_in = gr.Number(label="Max frames", value=0, minimum=0, maximum=500, precision=0,
                                                  info="Total frame cap")
                            chunk_in = gr.Number(label="Chunk size (s)", value=0, minimum=0, maximum=7200, precision=0,
                                                 info="Seconds per chunk for long videos")

                        run_btn    = gr.Button("🚀  Run Investigation", variant="primary", size="lg", elem_id="run-btn")
                        status_out = gr.HTML(value='<div style="height:36px"></div>')

                    # Right panel
                    with gr.Column(scale=7, min_width=500):
                        summary_out = gr.Markdown(visible=True)
                        report_html = gr.HTML(value=PLACEHOLDER)

                # Bottom row
                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        pdf_out = gr.File(label="📥 Download PDF Report", elem_classes="pdf-download")
                    with gr.Column(scale=4):
                        with gr.Accordion("📄 Raw JSON Report", open=False):
                            json_out = gr.Code(language="json")

                run_btn.click(
                    fn=investigate,
                    inputs=[video_in, url_in, manual_in, sr_in, mf_in, chunk_in],
                    outputs=[report_html, json_out, pdf_out, status_out, summary_out],
                )

            # ── BUILD KB ───────────────────────────────────────────────────────
            with gr.TabItem("📚  Knowledge Base"):
                gr.HTML('<div class="panel"><p style="color:#94a3b8;margin:0"><strong style="color:#e2e8f0">Optional:</strong> Index your SOPs and historical incident records for deeper retrieval. The investigation works without this, but indexing improves rule citation accuracy.</p></div>')
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<p class="panel-title" style="color:#64748b;font-size:0.75em;font-weight:700;letter-spacing:1px;text-transform:uppercase">Upload Documents</p>')
                        kb_sop = gr.File(label="Safety SOPs", file_count="multiple", file_types=[".pdf",".docx",".txt",".md"])
                        kb_man = gr.File(label="Equipment Manuals", file_count="multiple", file_types=[".pdf",".docx",".txt"])
                        kb_inc = gr.File(label="Incident Records", file_count="multiple", file_types=[".json",".txt",".pdf"])
                    with gr.Column():
                        gr.HTML('<p class="panel-title" style="color:#64748b;font-size:0.75em;font-weight:700;letter-spacing:1px;text-transform:uppercase">Or Point to Folders</p>')
                        kb_sd = gr.Textbox(label="SOP folder path", placeholder="/path/to/sops")
                        kb_md = gr.Textbox(label="Manuals folder path", placeholder="/path/to/manuals")
                        kb_id = gr.Textbox(label="Incidents folder path", placeholder="/path/to/incidents")
                        kb_reset = gr.Checkbox(label="Reset index before building", value=False)
                kb_btn = gr.Button("⚙  Build / Update Knowledge Base", variant="secondary", size="lg")
                kb_out = gr.Markdown()
                kb_btn.click(fn=build_kb,
                             inputs=[kb_sop, kb_man, kb_inc, kb_sd, kb_md, kb_id, kb_reset],
                             outputs=[kb_out])

            # ── STATUS ─────────────────────────────────────────────────────────
            with gr.TabItem("📊  Status"):
                gr.HTML('<div class="panel"><p style="color:#94a3b8;margin:0">Live system health and active configuration.</p></div>')
                st_out = gr.Markdown()
                gr.Button("🔄  Refresh", size="sm").click(fn=get_status, outputs=st_out)
                app.load(fn=get_status, outputs=st_out)

            # ── HOW IT WORKS ───────────────────────────────────────────────────
            with gr.TabItem("ℹ  How It Works"):
                gr.Markdown("""
## How LabSentinel-RAG Works

**Fully dynamic** — no hardcoded rules. Every finding comes from your uploaded manual and what the AI sees in your video.

### Five-Stage Pipeline

| Stage | What Happens |
|-------|-------------|
| **1. Input** | Upload lab manual (PDF/DOCX/TXT) + incident video |
| **2. Video Analysis** | Gemini Vision reads every sampled frame against your manual using a structured checklist |
| **3. RAG Retrieval** | Retrieves the most relevant passages from your manual for each detected event |
| **4. LLM Reasoning** | Confirms violations, cites the exact rule, gives corrective actions |
| **5. Report** | Timestamped HTML + PDF report with every violation linked to its video frame |

### Key Features
- **Exact timestamps** — every finding shows the precise second in the video
- **Rule citation** — quotes the exact section from your manual, or states "Universal principle" if not covered
- **Cross-lab** — upload a chemistry manual → chemistry rules applied; biology manual → biology rules
- **Long videos** — processes hours of CCTV footage automatically in chunks
- **Download cache** — YouTube videos cached locally so re-running the same URL is instant

### Supported Inputs
- **Manuals**: PDF, DOCX, TXT — any lab type
- **Videos**: MP4, AVI, MOV, MKV, WebM, YouTube URLs, direct MP4 links
                """)

    return app


if __name__ == "__main__":
    port  = int(os.getenv("GRADIO_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    app   = create_app()
    app.launch(
        server_name   = "0.0.0.0",
        server_port   = port,
        share         = share,
        max_file_size = None,
        show_error    = True,
        css           = CSS,
        theme         = gr.themes.Base(
            primary_hue   = gr.themes.colors.blue,
            secondary_hue = gr.themes.colors.slate,
            neutral_hue   = gr.themes.colors.slate,
            font          = [gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
        ),
    )
