"""
Evidence Retriever — Stage 3 of LabSentinel-RAG pipeline.

For each detected event (that may be a violation), queries the three
ChromaDB collections to retrieve:
  • Relevant SOP / safety-rule clauses
  • Relevant equipment manual sections
  • Similar historical incidents

Uses the same SentenceTransformer embedding model as the builder —
no API key required for retrieval.
"""

from __future__ import annotations
import uuid

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger

from config.settings import settings
from src.ingestion.knowledge_base_builder import COLLECTIONS
from src.utils.models import DetectedEvent, DocumentChunk, RetrievedEvidence
from src.utils.json_utils import build_event_query


class EvidenceRetriever:
    """
    Semantic retriever backed by ChromaDB (cosine similarity).

    Must build the knowledge base first:
        python main.py build-kb
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.rag.chroma_persist_dir
        )
        self.ef = SentenceTransformerEmbeddingFunction(
            model_name=settings.rag.embedding_model
        )
        self._cache: dict[str, chromadb.Collection | None] = {}
        self._session_client = None
        self._session_col = None
        logger.info(f"EvidenceRetriever ready | store={settings.rag.chroma_persist_dir}")

    def retrieve_for_event(
        self, event: DetectedEvent
    ) -> dict[str, list[RetrievedEvidence]]:
        """
        Retrieve evidence for a single detected event.

        Returns:
            {
                "sops":      [RetrievedEvidence, ...],
                "manuals":   [RetrievedEvidence, ...],
                "incidents": [RetrievedEvidence, ...],
            }
        """
        query = _build_query(event)
        logger.debug(f"Retrieving for event @ {event.timestamp_sec}s | query: {query[:80]}…")
        sops = self._query("sop", query, settings.rag.top_k_sops)
        if self._session_col is not None:
            # Uploaded manual is the primary authority → its passages come first
            manual_hits = self._query_collection(self._session_col, "sop", query, settings.rag.top_k_sops)
            sops = (manual_hits + sops)[: settings.rag.top_k_sops + 2]
        return {
            "sops":      sops,
            "manuals":   self._query("manual",   query, settings.rag.top_k_manuals),
            "incidents": self._query("incident", query, settings.rag.top_k_incidents),
        }

    def index_session_manual(self, manual_text: str, chunk_words: int = 180, overlap: int = 40) -> int:
        """
        Index the manual uploaded for THIS investigation into a temporary
        in-memory collection, so every violation can retrieve the most relevant
        passages from the whole manual (not just the first few thousand chars
        that fit in the prompt). Returns the number of chunks indexed.
        """
        self.clear_session_manual()
        if not manual_text or not manual_text.strip():
            return 0
        from src.ingestion.document_loader import _chunk_text

        # Keep per-file titles from the "=== filename ===" headers added by the UI/CLI
        sections, title, buf = [], "Uploaded Manual", []
        for line in manual_text.splitlines():
            if line.startswith("=== ") and line.endswith(" ===") and len(line) > 8:
                if buf:
                    sections.append((title, "\n".join(buf)))
                title, buf = line[4:-4].strip(), []
            else:
                buf.append(line)
        if buf:
            sections.append((title, "\n".join(buf)))

        docs, metas = [], []
        for title, text in sections:
            for i, chunk in enumerate(_chunk_text(text, chunk_words, overlap)):
                docs.append(chunk)
                metas.append({"source_file": title, "source_type": "sop",
                              "title": f"{title} (uploaded manual)", "section": f"passage {i+1}"})
        if not docs:
            return 0

        self._session_client = chromadb.EphemeralClient()
        self._session_col = self._session_client.create_collection(
            name=f"session_manual_{uuid.uuid4().hex[:12]}",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        for i in range(0, len(docs), 100):
            self._session_col.add(
                ids=[f"m{j}" for j in range(i, min(i + 100, len(docs)))],
                documents=docs[i:i + 100], metadatas=metas[i:i + 100])
        logger.info(f"Indexed uploaded manual for retrieval: {len(docs)} passages")
        return len(docs)

    def clear_session_manual(self) -> None:
        if self._session_client is not None and self._session_col is not None:
            try:
                self._session_client.delete_collection(self._session_col.name)
            except Exception:
                pass
        self._session_client = None
        self._session_col = None

    def retrieve(
        self,
        query: str,
        source_type: str = "sop",
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        """
        Direct free-text retrieval from a single collection.
        Useful for testing and debugging.
        """
        return self._query(source_type, query, top_k)

    def collection_sizes(self) -> dict[str, int]:
        """Return document count per collection (useful for checking KB is built)."""
        sizes = {}
        for source_type, name in COLLECTIONS.items():
            col = self._get_collection(source_type)
            sizes[source_type] = col.count() if col else 0
        return sizes

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_collection(self, source_type: str) -> chromadb.Collection | None:
        if source_type not in self._cache:
            name = COLLECTIONS[source_type]
            try:
                self._cache[source_type] = self.client.get_collection(
                    name=name, embedding_function=self.ef
                )
            except Exception:
                logger.warning(
                    f"Collection '{name}' not found. "
                    "Run `python main.py build-kb` first."
                )
                self._cache[source_type] = None
        return self._cache[source_type]

    def _query(
        self, source_type: str, query: str, top_k: int
    ) -> list[RetrievedEvidence]:
        col = self._get_collection(source_type)
        return self._query_collection(col, source_type, query, top_k)

    def _query_collection(
        self, col, source_type: str, query: str, top_k: int
    ) -> list[RetrievedEvidence]:
        if col is None or col.count() == 0:
            return []

        n = min(top_k, col.count())
        results = col.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        evidence: list[RetrievedEvidence] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity score
            similarity = max(0.0, 1.0 - float(dist))
            if similarity < settings.rag.similarity_threshold:
                continue

            chunk = DocumentChunk(
                chunk_id    = str(uuid.uuid4()),
                source_file = meta.get("source_file", ""),
                source_type = meta.get("source_type", source_type),
                title       = meta.get("title", ""),
                content     = doc,
                section     = meta.get("section", ""),
                metadata    = dict(meta),
            )
            evidence.append(
                RetrievedEvidence(
                    chunk      = chunk,
                    score      = round(similarity, 4),
                    query_used = query,
                )
            )

        return evidence


def _build_query(event: DetectedEvent) -> str:
    """Construct a rich semantic query from a DetectedEvent."""
    return build_event_query(
        event.description,
        event.objects_involved,
        event.actions_observed,
        event.potential_violation,
        event.severity,
    )
