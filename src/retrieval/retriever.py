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
        return {
            "sops":      self._query("sop",      query, settings.rag.top_k_sops),
            "manuals":   self._query("manual",   query, settings.rag.top_k_manuals),
            "incidents": self._query("incident", query, settings.rag.top_k_incidents),
        }

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
