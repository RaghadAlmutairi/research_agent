import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store for scraped & embedded chunks."""

    def __init__(self, collection_name: str = "knowledge_base", persist_dir: str = "./chroma_db"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"VectorStore ready: collection='{collection_name}' path='{persist_dir}'")

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Insert or update chunks (with embeddings) into ChromaDB."""
        if not chunks:
            return 0

        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if not embedding:
                logger.warning("Chunk missing embedding, skipping.")
                continue
            ids.append(str(uuid.uuid4()))
            embeddings.append(embedding)
            documents.append(chunk["content"])
            meta = {k: str(v) for k, v in chunk.get("metadata", {}).items()}
            meta["chunk_index"] = str(chunk.get("chunk_index", 0))
            metadatas.append(meta)

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        logger.info(f"Upserted {len(ids)} chunks into ChromaDB.")
        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-n most similar chunks for a query vector."""
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"content": doc, "metadata": meta, "score": 1 - dist})
        return output

    def count(self) -> int:
        return self.collection.count()
