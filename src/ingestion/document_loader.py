"""
Document Loader — ingests SOP, manual, and incident documents into chunks.

Supports:
  • PDF  (pdfplumber primary, pypdf fallback)
  • DOCX / DOC
  • TXT / Markdown
  • JSON  — expects {"title": ..., "content": ...} or {"description": ...}
            Any JSON that doesn't have those keys is serialised as a string.

Each file is split into overlapping word-count chunks and returned as
a list of DocumentChunk objects ready for embedding.
"""

from __future__ import annotations
import json
import uuid
from pathlib import Path

from loguru import logger

# Import path constants directly from the settings module (not the Pydantic singleton)
from config.settings import (
    settings,
    SOP_DIR, MANUAL_DIR, INCIDENT_DIR, OUTPUT_DIR
)
from src.utils.models import DocumentChunk


# ── Text Extraction ───────────────────────────────────────────────────────────

def _extract_text(path: Path) -> str:
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _pdf_text(path)
    elif ext in (".docx", ".doc"):
        return _docx_text(path)
    elif ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    elif ext == ".json":
        return _json_text(path)
    else:
        logger.warning(f"Unsupported file type '{ext}', skipping: {path.name}")
        return ""


def _pdf_text(path: Path) -> str:
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    except Exception:
        # fallback to pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            logger.error(f"PDF extraction failed for {path.name}: {e}")
            return ""


def _docx_text(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"DOCX extraction failed for {path.name}: {e}")
        return ""


def _json_text(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # Try common content fields first
            for key in ("content", "description", "text", "body"):
                if key in data and isinstance(data[key], str):
                    # Prepend title if present
                    title = data.get("title", "")
                    return f"{title}\n{data[key]}" if title else data[key]
            # Fall back to full serialisation
            return json.dumps(data, indent=2)
        elif isinstance(data, list):
            # List of incident records
            parts = []
            for item in data:
                if isinstance(item, dict):
                    for key in ("content", "description", "text", "body"):
                        if key in item:
                            parts.append(str(item[key]))
                            break
                    else:
                        parts.append(json.dumps(item))
            return "\n\n".join(parts)
        return str(data)
    except Exception as e:
        logger.error(f"JSON extraction failed for {path.name}: {e}")
        return ""


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    chunk_size and overlap are in words, not characters.
    """
    words  = text.split()
    chunks = []
    step   = max(1, chunk_size - overlap)
    i      = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


# ── DocumentLoader ────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".json"}


class DocumentLoader:
    """
    Loads documents from the knowledge-base directories and produces
    DocumentChunk objects ready for embedding.

    Usage:
        loader = DocumentLoader()
        chunks = loader.load_all()          # from default directories
        chunks = loader.load_file(path, "sop")  # single file
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size    = chunk_size    or settings.rag.chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag.chunk_overlap

    def load_all(self) -> list[DocumentChunk]:
        """Load all documents from the three default directories."""
        from config import SOP_DIR, MANUAL_DIR, INCIDENT_DIR
        chunks: list[DocumentChunk] = []
        sources = [
            (SOP_DIR,      "sop"),
            (MANUAL_DIR,   "manual"),
            (INCIDENT_DIR, "incident"),
        ]
        for directory, source_type in sources:
            batch = self.load_directory(directory, source_type)
            chunks.extend(batch)
            logger.info(f"  {source_type}: {len(batch)} chunks from {directory}")

        logger.success(f"Total loaded: {len(chunks)} chunks from knowledge base")
        return chunks

    def load_directory(
        self,
        directory: Path,
        source_type: str,
        recursive: bool = False,
    ) -> list[DocumentChunk]:
        """Load all supported files from a directory."""
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        pattern = "**/*" if recursive else "*"
        files   = [
            f for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            logger.debug(f"No documents found in {directory}")
            return []

        chunks: list[DocumentChunk] = []
        for file_path in sorted(files):
            try:
                file_chunks = self.load_file(file_path, source_type)
                chunks.extend(file_chunks)
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
        return chunks

    def load_file(self, path: Path, source_type: str) -> list[DocumentChunk]:
        """
        Load and chunk a single file.
        source_type must be one of: "sop", "manual", "incident"
        """
        path = Path(path)
        text = _extract_text(path)
        if not text.strip():
            logger.warning(f"No text extracted from {path.name}")
            return []

        raw_chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
        title      = path.stem.replace("_", " ").replace("-", " ").title()

        return [
            DocumentChunk(
                chunk_id    = str(uuid.uuid4()),
                source_file = str(path),
                source_type = source_type,
                title       = title,
                content     = chunk,
                section     = f"chunk_{idx+1}_of_{len(raw_chunks)}",
                metadata    = {
                    "filename":     path.name,
                    "chunk_index":  idx,
                    "total_chunks": len(raw_chunks),
                },
            )
            for idx, chunk in enumerate(raw_chunks)
        ]
