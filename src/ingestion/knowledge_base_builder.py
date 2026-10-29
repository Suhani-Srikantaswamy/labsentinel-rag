"""
Knowledge Base Builder — embeds documents and stores them in ChromaDB.

Run once before using the investigation pipeline:
    python main.py build-kb

After adding new documents, run again — existing chunks are upserted
(no duplicates, no need to wipe the store).
"""

from __future__ import annotations
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger

from config.settings import settings
from src.utils.models import DocumentChunk
from .document_loader import DocumentLoader


# Collection names — shared constant used by both builder and retriever
COLLECTIONS = {
    "sop":      "labsentinel_sops",
    "manual":   "labsentinel_manuals",
    "incident": "labsentinel_incidents",
}


class KnowledgeBaseBuilder:
    """
    Embeds DocumentChunks with a local SentenceTransformer model and
    upserts them into a persistent ChromaDB vector store.

    No API key required — embeddings run fully locally.
    """

    def __init__(self):
        persist_dir = settings.rag.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = SentenceTransformerEmbeddingFunction(
            model_name=settings.rag.embedding_model
        )
        logger.info(
            f"KnowledgeBaseBuilder ready | "
            f"model={settings.rag.embedding_model} | "
            f"store={persist_dir}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        chunks: list[DocumentChunk] | None = None,
        sop_dir: str | None = None,
        manual_dir: str | None = None,
        incident_dir: str | None = None,
    ) -> dict[str, int]:
        """
        Build (or update) the knowledge base.

        Args:
            chunks:       Pre-loaded chunks (skips file loading if provided)
            sop_dir:      Override SOP directory path
            manual_dir:   Override manual directory path
            incident_dir: Override incident directory path

        Returns:
            Dict of source_type → chunks indexed
        """
        if chunks is None:
            loader = DocumentLoader()

            # Optionally override directories
            if sop_dir or manual_dir or incident_dir:
                from config import settings as cfg_module
                import config.settings as cfg_settings
                if sop_dir:
                    cfg_settings.SOP_DIR = Path(sop_dir)
                if manual_dir:
                    cfg_settings.MANUAL_DIR = Path(manual_dir)
                if incident_dir:
                    cfg_settings.INCIDENT_DIR = Path(incident_dir)

            chunks = loader.load_all()

        if not chunks:
            logger.warning("No chunks to index — knowledge base not updated")
            return {k: 0 for k in COLLECTIONS}

        # Group by source_type
        by_type: dict[str, list[DocumentChunk]] = {k: [] for k in COLLECTIONS}
        for chunk in chunks:
            if chunk.source_type in by_type:
                by_type[chunk.source_type].append(chunk)
            else:
                logger.warning(f"Unknown source_type '{chunk.source_type}' — skipping")

        counts: dict[str, int] = {}
        for source_type, type_chunks in by_type.items():
            if type_chunks:
                self._upsert(source_type, type_chunks)
            counts[source_type] = len(type_chunks)

        logger.success(
            f"Knowledge base updated — "
            + ", ".join(f"{k}: {v}" for k, v in counts.items())
        )
        return counts

    def get_stats(self) -> dict[str, int]:
        """Return the number of indexed chunks per collection."""
        stats = {}
        for source_type, name in COLLECTIONS.items():
            try:
                col = self.client.get_collection(name=name, embedding_function=self.ef)
                stats[source_type] = col.count()
            except Exception:
                stats[source_type] = 0
        return stats

    def reset(self) -> None:
        """Delete all collections (start fresh)."""
        for name in COLLECTIONS.values():
            try:
                self.client.delete_collection(name)
                logger.info(f"Deleted collection: {name}")
            except Exception:
                pass

    # ── Private ───────────────────────────────────────────────────────────────

    def _upsert(self, source_type: str, chunks: list[DocumentChunk]) -> None:
        name = COLLECTIONS[source_type]
        col  = self.client.get_or_create_collection(
            name             = name,
            embedding_function = self.ef,
            metadata         = {"hnsw:space": "cosine"},
        )

        ids       = [c.chunk_id for c in chunks]
        documents = [c.content  for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "source_type": c.source_type,
                "title":       c.title,
                "section":     c.section or "",
                "filename":    c.metadata.get("filename", ""),
            }
            for c in chunks
        ]

        # Batch upsert in groups of 100
        batch = 100
        for i in range(0, len(ids), batch):
            col.upsert(
                ids       = ids[i : i + batch],
                documents = documents[i : i + batch],
                metadatas = metadatas[i : i + batch],
            )

        logger.info(f"  Upserted {len(chunks)} chunks → '{name}'")


if __name__ == "__main__":
    builder = KnowledgeBaseBuilder()
    counts  = builder.build()
    print("\nKnowledge Base:")
    for k, v in counts.items():
        print(f"  {k:10s}: {v} chunks")
    print("\nCurrent totals:")
    for k, v in builder.get_stats().items():
        print(f"  {k:10s}: {v} chunks")
