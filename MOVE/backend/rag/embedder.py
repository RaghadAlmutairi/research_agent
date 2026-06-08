import logging
from typing import Any

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-large"


def _get_client():
    from openai import OpenAI
    return OpenAI()


def embed_text(text: str) -> list[float]:
    """Embed a single string and return the vector."""
    client = _get_client()
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding


def embed_chunks(chunks: list[dict[str, Any]], batch_size: int = 100) -> list[dict[str, Any]]:
    """Add an 'embedding' field to each chunk dict (batched for efficiency)."""
    client = _get_client()
    texts = [c["content"] for c in chunks]

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])

    for chunk, vec in zip(chunks, all_embeddings):
        chunk["embedding"] = vec

    return chunks
