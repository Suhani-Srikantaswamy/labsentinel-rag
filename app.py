"""
LabSentinel-RAG — Production Web Application
ClimbingAddict-inspired design: dark dashboard, orange accents, light/dark toggle.
"""
from __future__ import annotations
import hashlib, html as _html, json, os, re, shutil, socket, tempfile, threading, time
from collections import Counter
from pathlib import Path
from queue import Queue, Empty
import gradio as gr
from loguru import logger

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _resolve_gradio_port(port: int | str | None = None) -> int:
    if port is None:
        port = os.getenv("GRADIO_PORT", "7860")
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 7860

    if port <= 0:
        port = 7860

    if _is_port_free(port):
        return port

    for candidate in range(port + 1, port + 101):
        if _is_port_free(candidate):
            logger.warning(f"Port {port} is occupied; using free port {candidate} instead.")
            return candidate

    raise RuntimeError(f"No free Gradio port found near {port}. Please set GRADIO_PORT to an unused port.")


def _api_ok():
    from src.utils.llm_client import get_provider_info
    i = get_provider_info()
    return i["configured"], i["status_msg"]

def _extract_text(path: str) -> str:
    """Same extractor as the knowledge base (pdfplumber → pypdf fallback, DOCX incl. tables)."""
    from src.ingestion.document_loader import extract_text
    p = Path(path)
    try:
        if p.suffix.lower() in (".pdf", ".docx", ".doc", ".json"):
            text = extract_text(p)
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            logger.warning(f"No text could be extracted from {p.name}")
        return text
    except Exception as e:
        logger.warning(f"Text extraction failed {p.name}: {e}"); return ""

def _normalize_manual_files(manual_files):
    if not manual_files:
        return []
    if isinstance(manual_files, (str, Path)):
        manual_files = [manual_files]

    flattened = []
    for item in manual_files:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            flattened.extend(_normalize_manual_files(item))
            continue
        if isinstance(item, dict):
            if "path" in item and item["path"]:
                flattened.append(item["path"])
            elif "name" in item and item["name"]:
                flattened.append(item["name"])
            continue
        flattened.append(item)

    seen = set()
    ordered = []
    for item in flattened:
        key = str(item)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _coerce_file_path(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            p = _coerce_file_path(item)
            if p:
                return p
        return None
    if isinstance(value, dict):
        for key in ("path", "file", "value"):
            if key in value and value[key]:
                return _coerce_file_path(value[key])
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return str(value)


def _combine_manuals(files):
    files = _normalize_manual_files(files)
    if not files: return ""
    parts = []
    for f in files:
        if f is None: continue
        t = _extract_text(str(f))
        if t.strip(): parts.append(f"=== {Path(str(f)).name} ===\n{t}")
    return "\n\n".join(parts)


LAB_TYPE_ALIASES = {
    "chemistry": {"chemistry", "chem", "chemical", "lab_chemistry", "chemistry_lab"},
    "biology": {"biology", "biological", "biolab", "bio_lab", "microbiology"},
    "physics": {"physics", "physical", "physics_lab", "mechanical"},
    "engineering": {"engineering", "engineer", "materials", "electronics", "electrical"},
    "computer_science": {"computer_science", "computer science", "cs", "it", "software", "cyber"},
    "medical": {"medical", "hospital", "clinical", "healthcare", "pharmacy"},
    "general": {"general", "mixed", "universal", "all", "standard"},
}


def _normalize_lab_type(value):
    if value is None:
        return "general"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return "general"
    for canonical, aliases in LAB_TYPE_ALIASES.items():
        if text == canonical or text in aliases:
            return canonical
        if text.startswith(canonical):
            return canonical
    if "computer" in text and "science" in text:
        return "computer_science"
    if text.startswith("bio"):
        return "biology"
    if text.startswith("chem"):
        return "chemistry"
    if "med" in text:
        return "medical"
    return "general"


# Distinctive vocabulary per lab type (whole-word / phrase matches, counted)
_MANUAL_KEYWORDS = {
    "chemistry": ["chemistry", "fume hood", "fume cupboard", "acid", "acids", "solvent", "solvents", "reagent",
                  "reagents", "titration", "bunsen", "flammable", "corrosive", "msds", "sds", "chemical hygiene"],
    "biology": ["biosafety", "bsl", "bsl-2", "biological safety cabinet", "biohazard", "microbiology", "microbial",
                "pathogen", "pathogens", "culture", "cultures", "petri", "autoclave", "infectious", "bacteria", "virus"],
    "physics": ["laser", "lasers", "optics", "optical table", "oscilloscope", "radiation", "radioactive",
                "high voltage", "cryogenic", "magnet"],
    "engineering": ["machinery", "machine guard", "lathe", "milling", "welding", "grinder", "lockout", "tagout",
                    "workshop", "power tools", "drill press"],
    "computer_science": ["computer lab", "computers", "server room", "servers", "keyboard", "workstation",
                         "network cabling", "software"],
    "medical": ["patient", "patients", "clinical", "hospital", "phlebotomy", "blood sample", "specimen",
                "nursing", "ward", "intravenous"],
}


def _infer_lab_type_from_manual(manual_text: str) -> str:
    """Score each lab type by distinctive-keyword counts; return 'general' unless one clearly dominates."""
    if not manual_text or not manual_text.strip():
        return "general"
    lowered = manual_text.lower()
    scores = {}
    for lab, words in _MANUAL_KEYWORDS.items():
        scores[lab] = sum(len(re.findall(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", lowered)) for w in words)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top, top_score), (_, second_score) = ranked[0], ranked[1]
    if top_score >= 3 and top_score >= 1.5 * max(second_score, 1):
        return top
    return "general"


def _majority_video_lab_type(timeline) -> str:
    """Most common non-general lab type across all analysed events."""
    votes = Counter()
    for event in timeline:
        for entry in event.objects_involved:
            if str(entry).startswith("lab_type:"):
                lt = _normalize_lab_type(str(entry).split(":", 1)[1])
                if lt != "general":
                    votes[lt] += 1
    return votes.most_common(1)[0][0] if votes else "general"


def detect_manual_lab_mismatch(video_lab_type, manual_lab_type):
    """Return True if the uploaded manual clearly targets a different lab type than the video evidence."""
    video_type = _normalize_lab_type(video_lab_type)
    manual_type = _normalize_lab_type(manual_lab_type)
    if not manual_type or manual_type == "general":
        return False
    if not video_type or video_type == "general":
        return False
    return video_type != manual_type


def _url_cache_path(url: str) -> Path:
    from config.settings import VIDEO_DIR
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    return VIDEO_DIR / f"cached_{h}.mp4"

def _download_url(url: str, progress_cb=None):
    cached = _url_cache_path(url)
    if cached.exists() and cached.stat().st_size > 10_000:
        logger.info(f"Using cached video: {cached}")
        if progress_cb: progress_cb(1.0, f"✅ Using cached video ({cached.stat().st_size//1024//1024} MB)")
        return str(cached), None, None
    tmp = tempfile.mkdtemp()
    try:
        import yt_dlp
        class Hook:
            def __call__(self, d):
                if d['status'] == 'downloading' and progress_cb:
                    total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                    done  = d.get('downloaded_bytes', 0)
                    if total > 0:
                        progress_cb(done/total*0.95,
                            f"Downloading {done/1024/1024:.1f} / {total/1024/1024:.1f} MB …")
        opts = {
            "format": ("bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]"
                       "/bestvideo[height<=480]+bestaudio/best[height<=480]/best"),
            "outtmpl": str(Path(tmp)/"video.%(ext)s"),
            "merge_output_format": "mp4", "quiet": True,
            "no_warnings": True, "noplaylist": True,
            "concurrent_fragment_downloads": 8,
            "progress_hooks": [Hook()],
        }
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            opts["ffmpeg_location"] = ffmpeg
        with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])
        for ext in ("mp4","webm","mkv","avi","mov"):
            files = list(Path(tmp).glob(f"*.{ext}"))
            if files:
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
            video_path=video_path, manual_text=manual_text,
            sample_rate=sample_rate or None, max_frames=max_frames or None,
            chunk_duration_sec=chunk_sec or None,
            on_progress=lambda p, m: pq.put((p, m)), verbose=False,
        )
        holder["report"] = report
    except Exception as e:
        logger.exception(e); holder["error"] = str(e)
    finally:
        pq.put(None)

# ── Main investigation ────────────────────────────────────────────────────────

def investigate(video_file, video_url, manual_files,
                sample_rate, max_frames, chunk_sec,
                progress=gr.Progress(track_tqdm=False)):

    manual_files = _normalize_manual_files(manual_files)

    ok, msg = _api_ok()
    if not ok:
        return _err("API Key Missing", [
            "Get a FREE Google Gemini key at https://aistudio.google.com/apikey",
            "Add to .env:  GOOGLE_API_KEY=your-key",
            "Restart the app: python app.py",
        ]), "{}", None, _stat("❌", "No API key"), "", _load_history(5)

    video_path = None; tmp_dir = None
    if video_url and video_url.strip():
        cached = _url_cache_path(video_url.strip())
        if cached.exists() and cached.stat().st_size > 10_000:
            video_path = str(cached)
            progress(0.05, desc=f"✅ Cached video ({cached.stat().st_size//1024//1024} MB)")
        else:
            progress(0.02, desc="Downloading video …")
            video_path, tmp_dir, err = _download_url(
                video_url.strip(),
                progress_cb=lambda p, m: progress(0.02 + p * 0.10, desc=m))
            if err:
                return _err("Download Failed", [err, "Try uploading the video directly."]), "{}", None, _stat("❌", "Download failed"), "", _load_history(5)
    else:
        video_path = _coerce_file_path(video_file)
        if not video_path:
            return _err("No Video", ["Upload a video file or paste a URL."]), "{}", None, _stat("⚠️", "No input"), "", _load_history(5)

    progress(0.13, desc="Reading lab manual …")
    manual_text  = _combine_manuals(manual_files) if manual_files else ""
    manual_label = (f"{len([f for f in (manual_files or []) if f])} file(s), {len(manual_text):,} chars"
                    if manual_text else "No manual — universal rules applied")

    if not manual_text.strip():
        gr.Info("No lab manual uploaded — continuing with universal safety rules.")
    
    progress(0.15, desc="Starting investigation …")
    pq = Queue(); holder = {}
    t  = threading.Thread(target=_run_pipeline,
         args=(video_path, manual_text, sample_rate, max_frames, chunk_sec, pq, holder), daemon=True)
    t.start()

    while True:
        try: item = pq.get(timeout=900)
        except Empty: break
        if item is None: break
        pct, msg2 = item
        progress(min(0.15 + pct * 0.80, 0.96), desc=msg2)
        time.sleep(0.02)
    t.join(timeout=15)

    if "error" in holder:
        err = holder["error"]
        if "Gemini API access is blocked" in err or "free tier is not available" in err.lower() or "billing" in err.lower():
            return _err("Gemini project blocked", [
                "The configured Google project cannot use Gemini right now.",
                "This normally means the project does not have Gemini enabled or billing is not active.",
                "Open https://aistudio.google.com, enable the correct project, turn on billing if needed, and add a valid key to .env.",
                "Then restart the app and try again.",
            ]), "{}", None, _stat("❌", "Gemini project blocked"), "", _load_history(5)
        if "All " in err and "Gemini API key" in err and "exhausted" in err.lower():
            return _err("All API keys are exhausted", [
                "Every configured Gemini API key has reached its daily quota.",
                "Add another key in .env (GOOGLE_API_KEY_2=..., GOOGLE_API_KEY_3=...) or wait for the quota to reset.",
                "You can also reduce Max Frames in Advanced Settings to use fewer requests.",
            ]), "{}", None, _stat("❌", "All API keys exhausted"), "", _load_history(5)
        return _err("Investigation Failed", [err]), "{}", None, _stat("❌", err[:80]), "", _load_history(5)

    report = holder.get("report")
    if not report:
        return _err("No Result", ["Pipeline returned no report."]), "{}", None, _stat("❌", "No report"), "", _load_history(5)

    video_lab_type = _majority_video_lab_type(report.timeline)

    manual_lab_type = _infer_lab_type_from_manual(manual_text)
    mismatch = detect_manual_lab_mismatch(video_lab_type, manual_lab_type)
    warnings = list(report.metadata.get("warnings") or [])   # keep pipeline warnings (failed frames etc.)
    if mismatch:
        gr.Warning(f"Manual lab type mismatch: the uploaded manual looks like {manual_lab_type}, but the video appears to be {video_lab_type}. Review the rule set before relying on the report.")
        warnings.append(
            f"Manual lab type mismatch: uploaded manual ({manual_lab_type}) does not match the video lab type ({video_lab_type})."
        )
    elif not manual_text.strip():
        warnings.append("No manual uploaded — universal safety rules were applied.")
    report.metadata["warnings"] = warnings
    for w in warnings:
        if "could not be" in w:
            gr.Warning(w)
    report.metadata["manual_lab_type"] = manual_lab_type
    report.metadata["video_lab_type"] = video_lab_type
    report.metadata["manual_label"] = manual_label

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

    r = report.overall_risk_level
    status_html = _stat("✅",
        f"Complete — <b>{report.total_violations} violation(s)</b> | "
        f"Risk: <b style='color:{_risk_col(r)}'>{r.upper()}</b> | "
        f"{len(report.timeline)} events | {report.metadata.get('frames_sampled','?')} frames")
    return html, js, pdf, status_html, _summary_md(report, manual_label), _load_history(5)

# ── KB builder ────────────────────────────────────────────────────────────────

def build_kb(sop_f, man_f, inc_f, sop_d, man_d, inc_d, reset, progress=gr.Progress()):
    progress(0.05, desc="Loading …")
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
            for f in _normalize_manual_files(files):
                f = _coerce_file_path(f)
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
        lines = [f"### ✅ Knowledge base updated — {sum(counts.values())} chunks\n",
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
    except:
        stats = {}
    total = sum(stats.values())

    api_ok  = pi.get("configured", False)
    kb_ok   = total > 0
    model   = pi.get("model", "—")
    is_free = pi.get("is_free", True)

    def row(label, value, ok=None):
        dot = ""
        if ok is True:  dot = '<span style="color:#22c55e;font-size:1.1em">✅</span> '
        if ok is False: dot = '<span style="color:#ef4444;font-size:1.1em">❌</span> '
        return (f'<tr>'
                f'<td style="padding:10px 14px;border-bottom:1px solid var(--ls-border);color:var(--ls-text2);font-size:0.85em;font-weight:500;width:180px">{label}</td>'
                f'<td style="padding:10px 14px;border-bottom:1px solid var(--ls-border);color:var(--ls-text);font-size:0.85em">{dot}{value}</td>'
                f'</tr>')

    kb_rows = ""
    for k, v in stats.items():
        kb_rows += row(f"&nbsp;&nbsp;&nbsp;{k.title()}", f"{v} chunks")

    prov_val = f"{pi['status_msg'].replace('✅ ','')}" if api_ok else pi['status_msg'].replace('❌ ','')

    html = f"""
<div style="font-family:Inter,sans-serif;max-width:720px">
  <h3 style="color:var(--ls-text);font-size:1.0em;font-weight:700;margin:0 0 14px">System Status</h3>
  <table style="width:100%;border-collapse:collapse;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:10px;overflow:hidden">
    {row("AI Provider", prov_val, api_ok)}
    {row("Knowledge Base", f"{total} chunks indexed" if kb_ok else "Empty — run Build KB", kb_ok)}
    {kb_rows}
  </table>

  <h3 style="color:var(--ls-text);font-size:1.0em;font-weight:700;margin:22px 0 12px">Active Configuration</h3>
  <table style="width:100%;border-collapse:collapse;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:10px;overflow:hidden">
    {row("Model", f'<code style="background:var(--ls-surface2);padding:2px 6px;border-radius:4px;font-size:0.85em">{model}</code> &nbsp;<span style="font-size:0.78em;color:{"#22c55e" if is_free else "#f59e0b"};font-weight:600">{"FREE" if is_free else "PAID"}</span>')}
    {row("Frame sample rate", f'Every <code style="background:var(--ls-surface2);padding:2px 6px;border-radius:4px;font-size:0.85em">{settings.video.frame_sample_rate}s</code>')}
    {row("Max frames per video", f'<code style="background:var(--ls-surface2);padding:2px 6px;border-radius:4px;font-size:0.85em">{settings.video.max_frames}</code>')}
    {row("Embedding model", f'<code style="background:var(--ls-surface2);padding:2px 6px;border-radius:4px;font-size:0.85em">{settings.rag.embedding_model}</code> &nbsp;<span style="font-size:0.78em;color:var(--ls-text3)">runs locally</span>')}
    {row("Similarity threshold", f'<code style="background:var(--ls-surface2);padding:2px 6px;border-radius:4px;font-size:0.85em">{settings.rag.similarity_threshold}</code>')}
  </table>
</div>"""
    return html

# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_col(r):
    return {"low":"#16a34a","medium":"#d97706","high":"#ea580c","critical":"#dc2626"}.get(r,"#94a3b8")

def _stat(icon, msg):
    return (f'<div style="padding:10px 14px;border-radius:8px;'
            f'background:var(--ls-surface2);border:1px solid var(--ls-border);'
            f'color:var(--ls-text2);font-size:0.88em;font-family:Inter,sans-serif">'
            f'{icon} {msg}</div>')

def _summary_md(report, manual_label):
    from src.report.report_generator import _fmt_ts
    r = report.overall_risk_level
    e = {"low":"🟢","medium":"🟡","high":"🟠","critical":"🔴"}.get(r,"⚪")
    dur = _fmt_ts(report.metadata.get("duration_sec", 0))
    lines = [f"## {e} Investigation Complete\n",
             "| | |", "|---|---|",
             f"| **Risk** | **{r.upper()}** |",
             f"| **Violations** | {report.total_violations} |",
             f"| **Events** | {len(report.timeline)} |",
             f"| **Frames** | {report.metadata.get('frames_sampled','?')} |",
             f"| **Duration** | {dur} |",
             f"| **Manual** | {manual_label} |",
             f"\n**{report.incident_summary}**"]
    if report.key_findings:
        lines.append("\n**Key Findings:**")
        for f in report.key_findings: lines.append(f"- {f}")
    return "\n".join(lines)

def _err(title, tips):
    items = "".join(f"<li style='margin:5px 0;color:var(--ls-text2)'>{t}</li>" for t in tips)
    return (f'<div style="background:var(--ls-surface);border:1px solid #ef444440;'
            f'border-left:3px solid #ef4444;border-radius:10px;padding:24px 28px;margin:8px 0;'
            f'font-family:Inter,sans-serif">'
            f'<h3 style="color:#ef4444;margin:0 0 10px;font-size:1em">⚠ {title}</h3>'
            f'<ul style="margin:0;padding-left:18px;line-height:1.8">{items}</ul></div>')


def _load_history_report(report_path: str):
    if not report_path:
        return PLACEHOLDER, "{}", None, _stat("⚠️", "No history item selected"), "", _load_history(5)

    from config.settings import OUTPUT_DIR
    path = Path(report_path)
    try:
        path.resolve().relative_to(Path(OUTPUT_DIR).resolve())
    except ValueError:
        return PLACEHOLDER, "{}", None, _stat("⚠️", "Invalid history item"), "", _load_history(5)
    if not path.exists():
        return PLACEHOLDER, "{}", None, _stat("⚠️", "History report not found"), "", _load_history(5)

    try:
        import json as _json
        from src.utils.models import InvestigationReport
        from src.report.report_generator import ReportGenerator

        with path.open() as f:
            data = _json.load(f)
        report = InvestigationReport.model_validate(data)
        gen = ReportGenerator()
        html = gen._render_html(report)
        js = json.dumps(report.model_dump(mode="json"), indent=2, default=str)
        pdf = None
        try:
            pdf = gen.to_pdf(report)
        except Exception as e:
            logger.warning(f"History PDF failed: {e}")

        manual_label = report.metadata.get("manual_label") or "Historical report"
        return (
            html,
            js,
            pdf,
            _stat("✅",
                f"Loaded historical report — <b>{report.total_violations} violation(s)</b> | "
                f"Risk: <b style='color:{_risk_col(report.overall_risk_level)}'>{report.overall_risk_level.upper()}</b> | "
                f"{len(report.timeline)} events"),
            _summary_md(report, manual_label),
            _load_history(5),
        )
    except Exception as e:
        logger.exception(e)
        return PLACEHOLDER, "{}", None, _stat("❌", f"Could not load report: {e}"), "", _load_history(5)


def _load_history(n: int = 5) -> str:
    """Load last N investigation reports from outputs/ and render as HTML cards."""
    from config.settings import OUTPUT_DIR
    import glob, json as _json

    reports = sorted(
        glob.glob(str(OUTPUT_DIR / "report_*.json")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True
    )[:n]

    if not reports:
        return (
            '<div style="padding:12px;text-align:center;color:var(--ls-text3);font-size:0.8em">'
            'No past investigations yet</div>'
        )

    risk_colors = {"low":"#16a34a","medium":"#d97706","high":"#ea580c","critical":"#dc2626"}
    risk_bg     = {"low":"#16a34a18","medium":"#d9770618","high":"#ea580c18","critical":"#dc262618"}
    cards = []
    for rp in reports:
        try:
            with open(rp) as f:
                d = _json.load(f)
            r       = d.get("overall_risk_level","low")
            n_viol  = d.get("total_violations", 0)
            n_ev    = len(d.get("timeline", []))
            vid     = _html.escape(Path(d.get("video_path","?")).name[:28])
            gen_at  = d.get("generated_at","")[:16].replace("T"," ")
            dur_s   = d.get("metadata",{}).get("duration_sec", 0)
            from src.report.report_generator import _fmt_ts
            dur     = _fmt_ts(dur_s)
            rc      = risk_colors.get(r,"#94a3b8")
            rb      = risk_bg.get(r,"#94a3b818")
            cards.append(f"""
<div class="ls-history-card" data-history-id="{_html.escape(rp)}" style="padding:10px 12px;border-radius:8px;border:1px solid var(--ls-border);
            background:var(--ls-surface);margin-bottom:6px;cursor:pointer;transition:border-color 0.15s"
     onmouseover="this.style.borderColor='var(--ls-orange)'"
     onmouseout="this.style.borderColor='var(--ls-border)'">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
    <span style="font-size:0.78em;font-weight:600;color:var(--ls-text);
                 overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px"
          title="{vid}">{vid}</span>
    <span style="font-size:0.7em;padding:2px 8px;border-radius:10px;font-weight:700;
                 color:{rc};background:{rb}">{_html.escape(str(r).upper())}</span>
  </div>
  <div style="display:flex;gap:12px;font-size:0.72em;color:var(--ls-text3)">
    <span>⚠ {n_viol} violations</span>
    <span>📋 {n_ev} events</span>
    <span>⏱ {dur}</span>
  </div>
  <div style="font-size:0.68em;color:var(--ls-text3);margin-top:3px">{_html.escape(gen_at)}</div>
</div>""")
        except Exception:
            continue

    if not cards:
        return '<div style="padding:12px;text-align:center;color:var(--ls-text3);font-size:0.8em">No valid history found</div>'

    return "".join(cards)

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ════════════════════════════════════════
   LIGHT MODE — warm, textured, alive
   ════════════════════════════════════════ */
:root, .light {
  --ls-bg:        #FDF8F3;
  --ls-surface:   #FFFFFF;
  --ls-surface2:  #FAF4ED;
  --ls-surface3:  #F5EDE2;
  --ls-border:    #EAE0D5;
  --ls-border2:   #D9CCBE;
  --ls-text:      #1C1714;
  --ls-text2:     #6B5E52;
  --ls-text3:     #A89B8E;
  --ls-orange:    #F5623A;
  --ls-orange2:   #E04E28;
  --ls-oglow:     rgba(245,98,58,0.20);
  --ls-odim:      rgba(245,98,58,0.08);
  --ls-input:     #FFFFFF;
  --ls-accent1:   #FFF4ED;
  --ls-accent2:   #FDE8DB;
}

/* ════════════════════════════════════════
   DARK MODE — charcoal, warm undertone
   ════════════════════════════════════════ */
.dark {
  --ls-bg:        #0E0D0B;
  --ls-surface:   #171613;
  --ls-surface2:  #1F1E1A;
  --ls-surface3:  #272520;
  --ls-border:    #2C2A25;
  --ls-border2:   #3A3730;
  --ls-text:      #F0EBE4;
  --ls-text2:     #9E9488;
  --ls-text3:     #5C574F;
  --ls-orange:    #F5623A;
  --ls-orange2:   #FF7550;
  --ls-oglow:     rgba(245,98,58,0.22);
  --ls-odim:      rgba(245,98,58,0.10);
  --ls-input:     #171613;
  --ls-accent1:   #1F1510;
  --ls-accent2:   #2A1A12;
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box !important; }
footer { display: none !important; }
body, .gradio-container {
  font-family: 'Inter', system-ui, sans-serif !important;
  background: var(--ls-bg) !important;
  color: var(--ls-text) !important;
}
/* Container: fixed-width centred card */
html, body {
  width: 100% !important;
  min-height: 100vh !important;
  margin: 0 !important;
  padding: 0 !important;
}
body {
  background: #1a1a1a !important;
  display: block !important;
  text-align: center !important;
}
.gradio-container {
  width: min(1200px, calc(100vw - 28px)) !important;
  max-width: 1200px !important;
  margin: 20px auto !important;
  padding: 0 !important;
  display: block !important;
  text-align: left !important;
}

.gradio-container > div { width: 100% !important; }

/* ── Topbar ── */
.ls-topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 20px; height: 52px;
  background: var(--ls-surface); border-bottom: 1px solid var(--ls-border);
  position: sticky; top: 0; z-index: 200;
}
.ls-logo {
  display: flex; align-items: center; gap: 10px;
  font-weight: 800; font-size: 1.05em; letter-spacing: -0.4px; color: var(--ls-text);
}
.ls-logo-icon {
  width: 30px; height: 30px; background: var(--ls-orange); border-radius: 7px;
  display: flex; align-items: center; justify-content: center; font-size: 15px;
  box-shadow: 0 0 14px var(--ls-oglow);
}
.ls-logo span { color: var(--ls-orange); }
.ls-topbar-right { display: flex; align-items: center; gap: 10px; }
.ls-chip {
  display: flex; align-items: center; gap: 6px; padding: 5px 12px;
  border-radius: 20px; border: 1px solid var(--ls-border);
  font-size: 0.74em; font-weight: 500; color: var(--ls-text2); background: var(--ls-surface2);
}
.ls-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #22c55e; box-shadow: 0 0 6px #22c55e90; flex-shrink: 0;
}
#theme-toggle {
  width: 34px; height: 34px;
  border: 1px solid var(--ls-border) !important;
  border-radius: 8px !important; background: var(--ls-surface2) !important;
  color: var(--ls-text2) !important; cursor: pointer !important;
  font-size: 15px !important; transition: all 0.2s !important;
  padding: 0 !important; min-width: unset !important; box-shadow: none !important;
  display: flex !important; align-items: center !important; justify-content: center !important;
}
#theme-toggle:hover { border-color: var(--ls-orange) !important; color: var(--ls-orange) !important; }

/* ── Tabs ── */
.tab-nav {
  border-bottom: 1px solid var(--ls-border) !important;
  background: var(--ls-surface) !important; padding: 0 20px !important;
}
.tab-nav button {
  color: var(--ls-text3) !important; font-weight: 500 !important;
  font-size: 0.84em !important; padding: 13px 15px !important;
  border-radius: 0 !important; border-bottom: 2px solid transparent !important;
  transition: all 0.2s !important;
}
.tab-nav button:hover { color: var(--ls-text2) !important; }
.tab-nav button.selected { color: var(--ls-orange) !important; border-bottom-color: var(--ls-orange) !important; }

/* ── Section label ── */
.ls-section-label {
  font-size: 0.62em !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 1.5px !important;
  color: var(--ls-text3) !important; margin-bottom: 8px !important; display: block !important;
}

/* Investigate tab sidebar row */
#investigate-row {
  width: 100% !important;
  min-height: 620px !important;
  gap: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border-top: 1px solid var(--ls-border) !important;
  background: var(--ls-bg) !important;
  flex-wrap: nowrap !important;
  align-items: stretch !important;
}

/* Left sidebar panel */
.ls-left {
  background: var(--ls-surface) !important;
  border-right: 1px solid var(--ls-border) !important;
  padding: 12px 10px 12px 12px !important;
  overflow-y: auto !important;
  flex-shrink: 0 !important;
  min-width: 320px !important;
}

/* Right content panel */
.ls-right {
  background: var(--ls-surface) !important;
  padding: 12px 16px 16px !important;
  overflow-y: auto !important;
  flex: 1 !important;
  min-height: 520px !important;
}

/* ── Upload zones ── */
.gradio-file, .gradio-video {
  background: var(--ls-surface2) !important;
  border: 1.5px dashed var(--ls-border2) !important;
  border-radius: 10px !important; transition: all 0.2s !important;
}
.gradio-file:hover, .gradio-video:hover {
  border-color: var(--ls-orange) !important; background: var(--ls-odim) !important;
}

/* ── Text inputs ── */
.gradio-textbox textarea, .gradio-textbox input, .gradio-number input {
  background: var(--ls-input) !important;
  border: 1.5px solid var(--ls-border2) !important;
  border-radius: 8px !important; color: var(--ls-text) !important;
  font-family: 'Inter', sans-serif !important; font-size: 0.85em !important;
  transition: border-color 0.2s !important;
}
.gradio-textbox textarea:focus, .gradio-textbox input:focus, .gradio-number input:focus {
  border-color: var(--ls-orange) !important;
  box-shadow: 0 0 0 3px var(--ls-odim) !important; outline: none !important;
}

/* ── Labels ── */
.gradio-textbox label, .gradio-file label, .gradio-video label,
.gradio-number label, .gradio-checkbox label {
  color: var(--ls-text2) !important; font-size: 0.8em !important; font-weight: 500 !important;
}

/* ── Accordion ── */
.gradio-accordion {
  background: var(--ls-surface2) !important;
  border: 1px solid var(--ls-border) !important; border-radius: 10px !important;
}
.gradio-accordion .label-wrap span { color: var(--ls-text2) !important; font-size: 0.82em !important; }

/* ── RUN BUTTON ── */
#run-btn {
  background: var(--ls-orange) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-weight: 700 !important; font-size: 0.92em !important; padding: 14px !important;
  box-shadow: 0 4px 18px var(--ls-oglow) !important; transition: all 0.2s !important;
  font-family: 'Inter', sans-serif !important;
}
#run-btn:hover {
  background: var(--ls-orange2) !important; transform: translateY(-1px) !important;
  box-shadow: 0 8px 26px var(--ls-oglow) !important;
}
#run-btn:active { transform: translateY(0) !important; }

/* ── PDF button ── */
.pdf-btn button {
  background: var(--ls-orange) !important; color: white !important;
  font-weight: 600 !important; border-radius: 8px !important; border: none !important;
}

/* ── Code/JSON ── */
.gradio-code {
  background: var(--ls-surface2) !important;
  border: 1px solid var(--ls-border) !important; border-radius: 10px !important;
}

/* ── Markdown ── */
.gradio-markdown { color: var(--ls-text) !important; }
.gradio-markdown h2 { color: var(--ls-text) !important; font-weight: 700 !important; }
.gradio-markdown table { border-collapse: collapse !important; width: 100% !important; }
.gradio-markdown th {
  background: var(--ls-surface3) !important; color: var(--ls-text2) !important;
  padding: 8px 12px !important; border: 1px solid var(--ls-border) !important; font-size: 0.82em !important;
}
.gradio-markdown td {
  padding: 7px 12px !important; border: 1px solid var(--ls-border) !important;
  color: var(--ls-text2) !important; font-size: 0.85em !important;
}
.gradio-markdown strong { color: var(--ls-text) !important; }

/* ════════════════════════════════════════════════════════
   FULL-WIDTH LAYOUT — target elem_ids directly
   ════════════════════════════════════════════════════════ */

/* Gradio container fills viewport / centred content area */
.gradio-container,
#main-tabs,
#main-tabs > div {
  width: 100% !important;
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
}

.gradio-container > div:not(.ls-history-wrap) {
  width: 100% !important;
  max-width: 100% !important;
}

/* Each tab fills full width */
#tab-investigate,
#tab-kb,
#tab-status,
#tab-how,
#tab-investigate > div,
#tab-kb > div,
#tab-status > div,
#tab-how > div {
  width: 100% !important;
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  background: var(--ls-bg) !important;
  min-height: calc(100vh - 112px) !important;
}

/* KB, Status, How It Works inner column fills full width */
#kb-page,
#status-page,
#how-page,
#kb-page > div,
#status-page > div,
#how-page > div {
  width: 100% !important;
  min-width: 100% !important;
  max-width: 100% !important;
  padding: 20px 24px !important;
  background: var(--ls-bg) !important;
  min-height: calc(100vh - 112px) !important;
  box-sizing: border-box !important;
}

/* Investigate sidebar row */
#investigate-row {
  width: 100% !important;
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  gap: 0 !important;
  flex-wrap: nowrap !important;
  min-height: calc(100vh - 112px) !important;
}

/* ls-page fallback (keep for compatibility) */
.ls-page {
  width: 100% !important;
  max-width: 100% !important;
  padding: 24px 28px !important;
  min-height: calc(100vh - 112px) !important;
  background: var(--ls-bg) !important;
  box-sizing: border-box !important;
}

/* ── Notice / info box used on KB, Status tabs ── */
.ls-notice {
  padding: 13px 16px;
  border-radius: 8px;
  background: var(--ls-surface);
  border: 1px solid var(--ls-border);
  color: var(--ls-text2);
  font-size: 0.87em;
  line-height: 1.6;
  margin-bottom: 20px;
}
.ls-notice strong { color: var(--ls-text); }

/* ── Section card — box used throughout KB/Status ── */
.ls-card {
  background: var(--ls-surface);
  border: 1px solid var(--ls-border);
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 16px;
}

/* ── Secondary button ── */
.ls-btn-secondary button {
  background: var(--ls-surface) !important;
  color: var(--ls-text) !important;
  border: 1.5px solid var(--ls-border2) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  transition: all 0.2s !important;
}
.ls-btn-secondary button:hover {
  border-color: var(--ls-orange) !important;
  color: var(--ls-orange) !important;
}

/* ── Refresh/small button ── */
.ls-btn-small button {
  background: var(--ls-surface2) !important;
  color: var(--ls-text2) !important;
  border: 1px solid var(--ls-border) !important;
  border-radius: 6px !important;
  font-size: 0.82em !important;
  font-weight: 500 !important;
}
.ls-btn-small button:hover {
  border-color: var(--ls-orange) !important;
  color: var(--ls-orange) !important;
}

/* ── History wrap ── */
.ls-history-wrap {
  background: var(--ls-surface2);
  border: 1px solid var(--ls-border);
  border-radius: 10px;
  padding: 10px 10px 4px;
  margin-top: 2px;
}
#history-target,
#history-load-btn {
  display: none !important;
}

/* ── Misc ── */
.progress-bar { background: var(--ls-orange) !important; }
.generating { border-color: var(--ls-orange) !important; }
.gradio-checkbox input[type=checkbox]:checked { accent-color: var(--ls-orange) !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--ls-border2); border-radius: 10px; }

/* ── Placeholder ── */
.ls-placeholder {
  display: flex !important; flex-direction: column !important;
  align-items: center !important; justify-content: center !important;
  min-height: 400px !important; gap: 14px !important; text-align: center !important;
}
.ls-placeholder-icon {
  width: 72px; height: 72px; border: 2px dashed var(--ls-border2); border-radius: 20px;
  display: flex; align-items: center; justify-content: center; font-size: 30px; color: var(--ls-text3);
}

/* ── LIGHT MODE extra warmth ── */
.light body          { background: #e8e4de !important; }
.light .ls-topbar      { background: #FFFAF6; border-bottom-color: #EAD8C4; }
.light .tab-nav        { background: #FFFAF6 !important; border-bottom-color: #EAD8C4 !important; }
.light .ls-left        { background: #FFFAF6 !important; border-right-color: #EAD8C4 !important; }
.light .ls-card        { background: #FFFCF9; border-color: #EAD8C4; }
.light .ls-notice      { background: #FFFCF9; border-color: #EAD8C4; }
.light .ls-history-wrap{ background: #FFF8F2; border-color: #EAD8C4; }
.light .gradio-accordion { background: #FFF8F2 !important; border-color: #EAD8C4 !important; }
.light .gradio-file,
.light .gradio-video   { background: #FFF8F3 !important; border-color: #D9C4AE !important; }
.light .gradio-code    { background: #FFF8F3 !important; border-color: #EAD8C4 !important; }
.light .ls-page        { background: #FDF8F3; }
.light #run-btn        { box-shadow: 0 4px 20px rgba(245,98,58,0.28) !important; }
.light .gradio-container { border-color: #D9CCBE !important; }
"""

PLACEHOLDER = """
<div style="
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:380px;gap:16px;text-align:center;padding:32px 24px;
  background:var(--ls-surface);border-radius:12px;border:1px solid var(--ls-border);
  margin:4px 0;
">
  <div style="
    width:64px;height:64px;background:var(--ls-surface2);border-radius:16px;
    display:flex;align-items:center;justify-content:center;font-size:28px;
    border:1.5px dashed var(--ls-border2);
  ">🔬</div>

  <div>
    <h2 style="color:var(--ls-text);font-size:1.2em;font-weight:800;margin:0 0 8px;letter-spacing:-0.3px">
      Ready to Investigate
    </h2>
    <p style="color:var(--ls-text2);font-size:0.84em;margin:0;line-height:1.7;max-width:320px">
      Upload your <strong style="color:var(--ls-text)">lab manual</strong> on the left,
      provide an <strong style="color:var(--ls-text)">incident video</strong>,
      then click <strong style="color:var(--ls-orange)">Run Investigation</strong>.
    </p>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:4px">
    <span style="padding:6px 14px;border-radius:20px;border:1px solid var(--ls-border);font-size:0.75em;color:var(--ls-text2);background:var(--ls-surface2)">
      <span style="color:var(--ls-orange);font-weight:700">1</span> &nbsp;Upload manual
    </span>
    <span style="padding:6px 14px;border-radius:20px;border:1px solid var(--ls-border);font-size:0.75em;color:var(--ls-text2);background:var(--ls-surface2)">
      <span style="color:var(--ls-orange);font-weight:700">2</span> &nbsp;Add video
    </span>
    <span style="padding:6px 14px;border-radius:20px;border:1px solid var(--ls-border);font-size:0.75em;color:var(--ls-text2);background:var(--ls-surface2)">
      <span style="color:var(--ls-orange);font-weight:700">3</span> &nbsp;Run
    </span>
    <span style="padding:6px 14px;border-radius:20px;border:1px solid var(--ls-border);font-size:0.75em;color:var(--ls-text2);background:var(--ls-surface2)">
      <span style="color:var(--ls-orange);font-weight:700">4</span> &nbsp;Download PDF
    </span>
  </div>

  <div style="
    margin-top:8px;padding:14px 20px;background:var(--ls-surface2);
    border-radius:10px;border:1px solid var(--ls-border);width:100%;max-width:400px;
  ">
    <div style="font-size:0.72em;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--ls-text3);margin-bottom:10px">What gets detected</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;text-align:left">
      <div style="font-size:0.78em;color:var(--ls-text2)">⚠️ PPE violations</div>
      <div style="font-size:0.78em;color:var(--ls-text2)">🧪 Chemical handling</div>
      <div style="font-size:0.78em;color:var(--ls-text2)">🔬 Equipment misuse</div>
      <div style="font-size:0.78em;color:var(--ls-text2)">🚨 Fire hazards</div>
      <div style="font-size:0.78em;color:var(--ls-text2)">🗑️ Waste disposal</div>
      <div style="font-size:0.78em;color:var(--ls-text2)">🚪 Emergency exits</div>
    </div>
  </div>

  <p style="font-size:0.72em;color:var(--ls-text3);margin:0">
    Any lab type · Any video length · Exact timestamps · Rule citation from your manual
  </p>
</div>"""

TOPBAR_HTML = """
<div class="ls-topbar" id="ls-topbar">
  <div class="ls-logo">
    <div class="ls-logo-icon">🔬</div>
    Lab<span>Sentinel</span>-RAG
  </div>
  <div class="ls-topbar-right">
    <div class="ls-chip" id="api-chip">
      <div class="ls-dot"></div>
      <span id="api-label">Gemini · auto-rotation ON</span>
    </div>
    <button id="theme-toggle" title="Toggle dark/light mode">🌙</button>
  </div>
</div>
"""

THEME_JS = """
function lsInit() {
  // Apply saved theme on load
  var saved = localStorage.getItem('ls-theme') || 'dark';
  document.documentElement.classList.add(saved);
  document.body.classList.add(saved);
  var btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.textContent = saved === 'dark' ? '🌙' : '☀️';
    btn.onclick = function() {
      var html = document.documentElement;
      var body = document.body;
      if (html.classList.contains('dark')) {
        html.classList.replace('dark','light');
        body.classList.replace('dark','light');
        btn.textContent = '☀️';
        localStorage.setItem('ls-theme','light');
      } else {
        html.classList.replace('light','dark');
        body.classList.replace('light','dark');
        btn.textContent = '🌙';
        localStorage.setItem('ls-theme','dark');
      }
    };
  }
}
"""

HEAD_INJECT = """
<script>
function lsCenter() {
  // Body has inline styles set by Gradio — override them directly
  document.body.style.cssText = [
    'width: 100%',
    'margin: 0',
    'padding: 20px 0',
    'display: flex',
    'flex-direction: column',
    'align-items: center',
    'min-height: 100vh',
    'background: #1a1a1a',
    'box-sizing: border-box'
  ].join(' !important;') + ' !important';

  // gradio-app element — constrain its width and centre it
  var ga = document.querySelector('gradio-app');
  if (ga) {
    ga.style.cssText = [
      'display: block',
      'width: min(1200px, calc(100vw - 28px))',
      'max-width: calc(100vw - 28px)',
      'margin: 0 auto'
    ].join(' !important;') + ' !important';
  }

  // Also fix tab panels inside shadow root
  var root = (ga && ga.shadowRoot) ? ga.shadowRoot : ga || document;
  root.querySelectorAll('[role="tabpanel"]').forEach(function(el) {
    el.style.setProperty('padding', '0', 'important');
    el.style.setProperty('width', '100%', 'important');
    el.style.setProperty('min-width', '100%', 'important');
    el.style.setProperty('box-sizing', 'border-box', 'important');
    Array.from(el.children).forEach(function(c) {
      c.style.setProperty('width', '100%', 'important');
      c.style.setProperty('min-width', '100%', 'important');
      c.style.setProperty('box-sizing', 'border-box', 'important');
    });
  });
}

[50, 200, 600, 1500].forEach(function(ms) { setTimeout(lsCenter, ms); });
document.addEventListener('click', function(e) {
  const card = e.target && e.target.closest ? e.target.closest('[data-history-id]') : null;
  if (card) {
    const reportPath = card.getAttribute('data-history-id');
    if (reportPath) {
      const appRoot = document.querySelector('gradio-app')?.shadowRoot || document;
      const target = appRoot.getElementById ? appRoot.getElementById('history-target') : null;
      const input = target ? target.querySelector('input, textarea') : null;
      if (input) {
        input.value = reportPath;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
      const btn = appRoot.getElementById ? appRoot.getElementById('history-load-btn') : null;
      if (btn) {
        btn.click();
      }
    }
  }
  if (e.target && e.target.getAttribute('role') === 'tab') {
    setTimeout(lsCenter, 100);
  }
}, true);
</script>
"""

# ── Build App ─────────────────────────────────────────────────────────────────



def create_app():
    with gr.Blocks(
        title       = "LabSentinel-RAG",
        fill_height = True,
        fill_width  = True,
    ) as app:
        gr.HTML(TOPBAR_HTML)

        with gr.Tabs(elem_classes="tab-nav", elem_id="main-tabs"):

            # ── INVESTIGATE ────────────────────────────────────────────────
            with gr.TabItem("🔍  Investigate", elem_id="tab-investigate"):
                with gr.Row(equal_height=False, elem_id="investigate-row"):
                    with gr.Column(scale=4, min_width=320, elem_classes="ls-left"):
                        gr.HTML('<span class="ls-section-label">Step 1 — Lab Manual</span>')
                        manual_in = gr.File(label="Upload Lab Manual / SOP", file_count="multiple",
                                            file_types=[".pdf",".docx",".txt",".md"])
                        gr.HTML('<p style="color:var(--ls-text3);font-size:0.76em;margin:4px 0 14px;line-height:1.5">'
                                'PDF · DOCX · TXT. The AI reads it and applies its rules. No manual = universal rules.</p>')
                        gr.HTML('<span class="ls-section-label">Step 2 — Incident Video</span>')
                        video_in = gr.Video(label="Upload Video", height=190)
                        url_in   = gr.Textbox(label="Or paste a URL",
                                              placeholder="https://youtube.com/watch?v=...", lines=1)
                        gr.HTML('<p style="color:var(--ls-text3);font-size:0.76em;margin:4px 0 14px;line-height:1.5">'
                                '⚡ Same URL cached — re-runs are instant.</p>')
                        with gr.Accordion("⚙  Advanced Settings", open=False):
                            gr.HTML('<p style="color:var(--ls-text3);font-size:0.76em;margin:0 0 10px">Leave 0 for .env defaults</p>')
                            with gr.Row():
                                sr_in = gr.Number(label="Frame rate (s)", value=0, minimum=0, maximum=60, precision=0)
                                mf_in = gr.Number(label="Max frames", value=0, minimum=0, maximum=500, precision=0)
                            chunk_in = gr.Number(label="Chunk size (s)", value=0, minimum=0, maximum=7200, precision=0)
                        run_btn    = gr.Button("🚀  Run Investigation", variant="primary", size="lg", elem_id="run-btn")
                        status_out = gr.HTML(value='<div style="height:36px"></div>')
                        gr.HTML('<div style="margin-top:16px"><span class="ls-section-label">Recent Investigations</span></div>')
                        history_out = gr.HTML(value=_load_history(5), elem_classes="ls-history-wrap")

                    with gr.Column(scale=7, min_width=500, elem_classes="ls-right"):
                        summary_out = gr.Markdown()
                        report_html = gr.HTML(value=PLACEHOLDER)

                history_target = gr.Textbox(value="", elem_id="history-target")
                history_load = gr.Button("Load history", elem_id="history-load-btn")

                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        pdf_out = gr.File(label="📥 Download PDF Report", elem_classes="pdf-btn")
                    with gr.Column(scale=4):
                        with gr.Accordion("{ }  Raw JSON Report", open=False):
                            json_out = gr.Code(language="json")

                run_btn.click(
                    fn=investigate,
                    inputs=[video_in, url_in, manual_in, sr_in, mf_in, chunk_in],
                    outputs=[report_html, json_out, pdf_out, status_out, summary_out, history_out],
                )
                history_load.click(
                    fn=_load_history_report,
                    inputs=[history_target],
                    outputs=[report_html, json_out, pdf_out, status_out, summary_out, history_out],
                )

            # ── KNOWLEDGE BASE ─────────────────────────────────────────────
            with gr.TabItem("📚  Knowledge Base", elem_id="tab-kb"):
                with gr.Column(elem_id="kb-page"):
                    gr.HTML('<div class="ls-notice"><strong>Optional</strong> — Index your SOPs and incident records '
                            'for deeper rule retrieval. Works without this, but improves citation accuracy.</div>')
                    with gr.Row(equal_height=False):
                        with gr.Column():
                            gr.HTML('<span class="ls-section-label">Upload Documents</span>')
                            kb_sop = gr.File(label="Safety SOPs", file_count="multiple",
                                             file_types=[".pdf",".docx",".txt",".md"])
                            kb_man = gr.File(label="Equipment Manuals", file_count="multiple",
                                             file_types=[".pdf",".docx",".txt"])
                            kb_inc = gr.File(label="Incident Records", file_count="multiple",
                                             file_types=[".json",".txt",".pdf"])
                        with gr.Column():
                            gr.HTML('<span class="ls-section-label">Or Point to Folders on Disk</span>')
                            kb_sd = gr.Textbox(label="SOP folder path", placeholder="/path/to/sops")
                            kb_md = gr.Textbox(label="Manuals folder path", placeholder="/path/to/manuals")
                            kb_id = gr.Textbox(label="Incidents folder path", placeholder="/path/to/incidents")
                            kb_reset = gr.Checkbox(label="Reset index before building", value=False)
                    kb_btn = gr.Button("⚙  Build / Update Knowledge Base", variant="secondary",
                                       size="lg", elem_classes="ls-btn-secondary")
                    kb_out = gr.Markdown()
                    kb_btn.click(fn=build_kb,
                                 inputs=[kb_sop, kb_man, kb_inc, kb_sd, kb_md, kb_id, kb_reset],
                                 outputs=[kb_out])

            # ── STATUS ─────────────────────────────────────────────────────
            with gr.TabItem("📊  Status", elem_id="tab-status"):
                with gr.Column(elem_id="status-page"):
                    st_out = gr.HTML()
                    gr.Button("🔄  Refresh", size="sm", elem_classes="ls-btn-small").click(
                        fn=get_status, outputs=st_out)
                    app.load(fn=get_status, outputs=st_out)

            # ── HOW IT WORKS ───────────────────────────────────────────────
            with gr.TabItem("ℹ  How It Works", elem_id="tab-how"):
                with gr.Column(elem_id="how-page"):
                    gr.HTML("""
<div style="font-family:Inter,sans-serif;max-width:860px">
  <h2 style="color:var(--ls-text);font-size:1.4em;font-weight:800;margin:0 0 8px;letter-spacing:-0.4px">How LabSentinel-RAG Works</h2>
  <p style="color:var(--ls-text2);margin:0 0 24px;line-height:1.7;font-size:0.9em">
    <strong style="color:var(--ls-text)">Fully dynamic — no hardcoded rules.</strong>
    Every finding comes from your uploaded manual and what the AI sees in your video.
  </p>
  <h3 style="color:var(--ls-text);font-size:0.88em;font-weight:700;margin:0 0 10px;text-transform:uppercase;letter-spacing:0.8px">Five-Stage Pipeline</h3>
  <table style="width:100%;border-collapse:collapse;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:10px;overflow:hidden;margin-bottom:24px">
    <tr><td style="padding:10px 16px;border-bottom:1px solid var(--ls-border);width:160px;font-weight:600;color:var(--ls-orange);font-size:0.85em">1. Input</td><td style="padding:10px 16px;border-bottom:1px solid var(--ls-border);color:var(--ls-text2);font-size:0.85em">Upload lab manual (PDF/DOCX/TXT) and incident video (any length)</td></tr>
    <tr><td style="padding:10px 16px;border-bottom:1px solid var(--ls-border);font-weight:600;color:var(--ls-orange);font-size:0.85em">2. Video Analysis</td><td style="padding:10px 16px;border-bottom:1px solid var(--ls-border);color:var(--ls-text2);font-size:0.85em">Gemini Vision reads every frame with a structured PPE and hazard checklist</td></tr>
    <tr><td style="padding:10px 16px;border-bottom:1px solid var(--ls-border);font-weight:600;color:var(--ls-orange);font-size:0.85em">3. RAG Retrieval</td><td style="padding:10px 16px;border-bottom:1px solid var(--ls-border);color:var(--ls-text2);font-size:0.85em">Retrieves relevant passages from your manual for each detected event</td></tr>
    <tr><td style="padding:10px 16px;border-bottom:1px solid var(--ls-border);font-weight:600;color:var(--ls-orange);font-size:0.85em">4. LLM Reasoning</td><td style="padding:10px 16px;border-bottom:1px solid var(--ls-border);color:var(--ls-text2);font-size:0.85em">Confirms violations, cites the exact rule, gives corrective actions</td></tr>
    <tr><td style="padding:10px 16px;font-weight:600;color:var(--ls-orange);font-size:0.85em">5. Report</td><td style="padding:10px 16px;color:var(--ls-text2);font-size:0.85em">Timestamped HTML and PDF report — every violation linked to exact video time</td></tr>
  </table>
  <h3 style="color:var(--ls-text);font-size:0.88em;font-weight:700;margin:0 0 12px;text-transform:uppercase;letter-spacing:0.8px">Key Features</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:24px">
    <div style="padding:12px 14px;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:8px"><div style="font-size:0.82em;font-weight:600;color:var(--ls-text);margin-bottom:3px">Exact Timestamps</div><div style="font-size:0.78em;color:var(--ls-text2)">MM:SS or H:MM:SS matching video duration</div></div>
    <div style="padding:12px 14px;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:8px"><div style="font-size:0.82em;font-weight:600;color:var(--ls-text);margin-bottom:3px">Rule Citation</div><div style="font-size:0.78em;color:var(--ls-text2)">Quotes exact section from your manual</div></div>
    <div style="padding:12px 14px;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:8px"><div style="font-size:0.82em;font-weight:600;color:var(--ls-text);margin-bottom:3px">Any Lab Type</div><div style="font-size:0.78em;color:var(--ls-text2)">Chemistry, biology, cleanroom, hospital</div></div>
    <div style="padding:12px 14px;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:8px"><div style="font-size:0.82em;font-weight:600;color:var(--ls-text);margin-bottom:3px">Long Videos</div><div style="font-size:0.78em;color:var(--ls-text2)">Hours of CCTV in chunks automatically</div></div>
    <div style="padding:12px 14px;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:8px"><div style="font-size:0.82em;font-weight:600;color:var(--ls-text);margin-bottom:3px">Multi-Key Rotation</div><div style="font-size:0.78em;color:var(--ls-text2)">Unlimited Gemini keys, auto-rotates</div></div>
    <div style="padding:12px 14px;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:8px"><div style="font-size:0.82em;font-weight:600;color:var(--ls-text);margin-bottom:3px">Download Cache</div><div style="font-size:0.78em;color:var(--ls-text2)">Same URL re-runs are instant</div></div>
  </div>
  <h3 style="color:var(--ls-text);font-size:0.88em;font-weight:700;margin:0 0 10px;text-transform:uppercase;letter-spacing:0.8px">Free Tier (Gemini)</h3>
  <table style="border-collapse:collapse;background:var(--ls-surface);border:1px solid var(--ls-border);border-radius:10px;overflow:hidden">
    <tr style="background:var(--ls-surface2)"><td style="padding:8px 20px;font-weight:600;font-size:0.82em;color:var(--ls-text2);border-bottom:1px solid var(--ls-border)">Keys</td><td style="padding:8px 20px;font-weight:600;font-size:0.82em;color:var(--ls-text2);border-bottom:1px solid var(--ls-border)">Requests/day</td><td style="padding:8px 20px;font-weight:600;font-size:0.82em;color:var(--ls-text2);border-bottom:1px solid var(--ls-border)">Investigations/day</td></tr>
    <tr><td style="padding:8px 20px;font-size:0.82em;color:var(--ls-text2);border-bottom:1px solid var(--ls-border)">1</td><td style="padding:8px 20px;font-size:0.82em;color:var(--ls-text2);border-bottom:1px solid var(--ls-border)">~20</td><td style="padding:8px 20px;font-size:0.82em;color:var(--ls-text2);border-bottom:1px solid var(--ls-border)">~1</td></tr>
    <tr><td style="padding:8px 20px;font-size:0.82em;color:var(--ls-text2);border-bottom:1px solid var(--ls-border)">4</td><td style="padding:8px 20px;font-size:0.82em;color:var(--ls-text2);border-bottom:1px solid var(--ls-border)">~80</td><td style="padding:8px 20px;font-size:0.82em;color:var(--ls-text2);border-bottom:1px solid var(--ls-border)">~5</td></tr>
    <tr><td style="padding:8px 20px;font-size:0.82em;color:var(--ls-text2)">10</td><td style="padding:8px 20px;font-size:0.82em;color:var(--ls-text2)">~200</td><td style="padding:8px 20px;font-size:0.82em;color:var(--ls-text2)">~12</td></tr>
  </table>
  <p style="color:var(--ls-text3);font-size:0.78em;margin:10px 0 0">Add to .env: GOOGLE_API_KEY_2=... GOOGLE_API_KEY_3=... (unlimited keys supported)</p>
</div>""")

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def launch(port: int | str | None = None, share: bool | None = None):
    """Start the web UI (used by `python app.py` and `python main.py demo`)."""
    port  = _resolve_gradio_port(port if port is not None else os.getenv("GRADIO_PORT", "7860"))
    if share is None:
        share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    app   = create_app()
    app.launch(
        server_name   = "0.0.0.0",
        server_port   = port,
        share         = share,
        max_file_size = None,
        show_error    = True,
        css           = CSS,
        js            = THEME_JS,
        head          = HEAD_INJECT,
        theme         = gr.themes.Base(
            primary_hue   = gr.themes.colors.orange,
            secondary_hue = gr.themes.colors.stone,
            neutral_hue   = gr.themes.colors.stone,
            font          = [gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
        ),
    )


if __name__ == "__main__":
    launch()
