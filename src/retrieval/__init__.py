from .retriever import _build_query

# EvidenceRetriever is imported lazily — chromadb not needed at import time
def _get_retriever():
    from .retriever import EvidenceRetriever
    return EvidenceRetriever
