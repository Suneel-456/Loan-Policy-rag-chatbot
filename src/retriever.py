"""Vector search over the saved index."""
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from src import config


def load_store(embeddings) -> InMemoryVectorStore:
    if not config.VECTORSTORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector index not found at {config.VECTORSTORE_PATH}. Run: python -m src.ingest"
        )
    return InMemoryVectorStore.load(str(config.VECTORSTORE_PATH), embeddings)


def retrieve(
    store: InMemoryVectorStore,
    question: str,
    k: int = None,
    min_score: float = None,
    query_vector: list[float] = None,
) -> list[tuple[Document, float]]:
    """Return the top-k (chunk, cosine score) pairs at or above min_score."""
    k = config.TOP_K if k is None else k
    min_score = config.MIN_RELEVANCE if min_score is None else min_score
    if query_vector is not None:  # reuse an embedding already computed (saves one model call)
        results = store.similarity_search_with_score_by_vector(query_vector, k=k)
    else:
        results = store.similarity_search_with_score(question, k=k)
    return [(doc, float(score)) for doc, score in results if score >= min_score]