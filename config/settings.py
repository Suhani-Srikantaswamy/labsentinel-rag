"""
LabSentinel-RAG — Global Configuration

All settings are configurable via environment variables or .env file.
Models, paths, and thresholds are overridable without code changes.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Project Paths ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Default data directories — override via env vars
SOP_DIR      = Path(os.getenv("LABSENTINEL_SOP_DIR",      str(DATA_DIR / "sops")))
MANUAL_DIR   = Path(os.getenv("LABSENTINEL_MANUAL_DIR",   str(DATA_DIR / "manuals")))
INCIDENT_DIR = Path(os.getenv("LABSENTINEL_INCIDENT_DIR", str(DATA_DIR / "incidents")))
VIDEO_DIR    = Path(os.getenv("LABSENTINEL_VIDEO_DIR",    str(DATA_DIR / "videos")))
EMBEDDING_DIR= Path(os.getenv("LABSENTINEL_EMBED_DIR",    str(DATA_DIR / "embeddings")))
OUTPUT_DIR   = Path(os.getenv("LABSENTINEL_OUTPUT_DIR",   str(ROOT_DIR / "outputs")))

# Create all directories on import
for _d in [SOP_DIR, MANUAL_DIR, INCIDENT_DIR, VIDEO_DIR, EMBEDDING_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── LLM / VLM Settings ───────────────────────────────────────────────────────
class LLMConfig(BaseModel):
    # Provider: auto | openai | gemini
    # "auto" picks Gemini if GOOGLE_API_KEY is set, else OpenAI
    provider: str = Field(
        default_factory=lambda: os.getenv("LABSENTINEL_PROVIDER", "auto")
    )
    # OpenAI models
    model: str = Field(
        default_factory=lambda: os.getenv("LABSENTINEL_VLM_MODEL", "gpt-4o")
    )
    reasoning_model: str = Field(
        default_factory=lambda: os.getenv("LABSENTINEL_REASONING_MODEL", "gpt-4o")
    )
    # Gemini model — gemini-3.5-flash-lite works with vision and has higher quota
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("LABSENTINEL_GEMINI_MODEL", "gemini-3.5-flash-lite")
    )
    temperature: float = float(os.getenv("LABSENTINEL_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("LABSENTINEL_MAX_TOKENS", "4096"))
    # API keys
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    def active_provider(self) -> str:
        """Return which provider will actually be used."""
        if self.provider == "openai":
            return "openai"
        if self.provider == "gemini":
            return "gemini"
        # auto: prefer gemini if key set, else openai
        if self.google_api_key:
            return "gemini"
        if self.api_key:
            return "openai"
        return "none"

    def is_configured(self) -> bool:
        return self.active_provider() in ("openai", "gemini")


# ── Video Analysis Settings ───────────────────────────────────────────────────
class VideoConfig(BaseModel):
    # Sample 1 frame every N seconds (lower = more thorough, higher API cost)
    # Default 3s gives good coverage without excessive cost
    frame_sample_rate: int = Field(
        default_factory=lambda: int(os.getenv("LABSENTINEL_FRAME_RATE", "3"))
    )
    # Hard cap on frames per video — raised to 60 for better accuracy
    # For a 90s video at 3s intervals = 30 frames naturally; cap of 60 gives room for motion detection extras
    max_frames: int = Field(
        default_factory=lambda: int(os.getenv("LABSENTINEL_MAX_FRAMES", "60"))
    )
    # For long videos: process N seconds at a time (300 = 5 min chunks)
    chunk_duration_sec: int = Field(
        default_factory=lambda: int(os.getenv("LABSENTINEL_CHUNK_SEC", "300"))
    )
    # Resize frames before sending to API (saves tokens)
    resize_width: int = int(os.getenv("LABSENTINEL_FRAME_WIDTH", "1280"))
    resize_height: int = int(os.getenv("LABSENTINEL_FRAME_HEIGHT", "720"))
    # Motion detection threshold — lowered for higher sensitivity
    # Lower = catches more subtle motion (someone removing a glove, etc.)
    motion_threshold: float = float(os.getenv("LABSENTINEL_MOTION_THRESHOLD", "10.0"))
    supported_formats: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


# ── RAG / Retrieval Settings ──────────────────────────────────────────────────
class RAGConfig(BaseModel):
    # Embedding model (runs locally — no API key needed)
    # Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (better quality)
    embedding_model: str = os.getenv("LABSENTINEL_EMBED_MODEL", "all-MiniLM-L6-v2")
    # Chunk size in words (512 ≈ ~1 paragraph)
    chunk_size: int = int(os.getenv("LABSENTINEL_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("LABSENTINEL_CHUNK_OVERLAP", "64"))
    # Retrieve more passages per event for better evidence coverage
    top_k_sops: int = int(os.getenv("LABSENTINEL_TOP_K_SOPS", "5"))
    top_k_incidents: int = int(os.getenv("LABSENTINEL_TOP_K_INCIDENTS", "3"))
    top_k_manuals: int = int(os.getenv("LABSENTINEL_TOP_K_MANUALS", "4"))
    # Similarity threshold — lowered so more manual passages are retrieved
    # (better to retrieve too many and let the LLM filter than to miss the relevant one)
    similarity_threshold: float = float(os.getenv("LABSENTINEL_SIM_THRESHOLD", "0.20"))
    chroma_persist_dir: str = str(EMBEDDING_DIR / "chroma")


# ── Report Settings ───────────────────────────────────────────────────────────
class ReportConfig(BaseModel):
    # Which formats to produce: any subset of ["json", "html", "pdf"]
    output_format: list[str] = Field(
        default_factory=lambda: os.getenv(
            "LABSENTINEL_OUTPUT_FORMATS", "json,html,pdf"
        ).split(",")
    )


# ── Master Config ─────────────────────────────────────────────────────────────
class LabSentinelConfig(BaseModel):
    llm: LLMConfig    = Field(default_factory=LLMConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    rag: RAGConfig    = Field(default_factory=RAGConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    debug: bool = Field(default_factory=lambda: os.getenv("LABSENTINEL_DEBUG", "false").lower() == "true")


# Singleton used everywhere
settings = LabSentinelConfig()
