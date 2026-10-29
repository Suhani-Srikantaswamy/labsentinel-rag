from .document_loader import DocumentLoader

# KnowledgeBaseBuilder and COLLECTIONS are imported lazily to avoid
# requiring chromadb at import time in test environments.
def _get_builder():
    from .knowledge_base_builder import KnowledgeBaseBuilder
    return KnowledgeBaseBuilder

def _get_collections():
    from .knowledge_base_builder import COLLECTIONS
    return COLLECTIONS
