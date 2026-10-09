"""Embedding model factory (open-source, runs locally)."""
from src import config


def get_embeddings():
    if config.USE_FAKE_MODELS:
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=256)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity == dot product
    )