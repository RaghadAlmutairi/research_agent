"""
store_research_tool    — store research results into ChromaDB
retrieve_research_tool — retrieve past research from ChromaDB
similarity_search_tool — semantic search across all stored research
"""
import json
import logging
import os
import uuid
from typing import Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
_vs = None


def _get_vs(collection: str = "research"):
    global _vs
    if _vs is None:
        from backend.rag.vector_store import VectorStore
        _vs = VectorStore(collection_name=collection)
    return _vs


@tool
def store_research_tool(research_type: str, company: str, data: dict) -> dict[str, Any]:
    """
    Store research results into ChromaDB for persistence and future retrieval.
    research_type: 'company_context' | 'competitor' | 'reviews' | 'trends' | 'report'
    """
    from backend.rag.embedder import embed_text

    content = json.dumps(data, indent=2)
    embedding = embed_text(content[:2000])

    vs = _get_vs("research")
    chunk = {
        "content": content,
        "embedding": embedding,
        "metadata": {
            "research_type": research_type,
            "company": company,
            "chunk_index": "0",
        },
        "chunk_index": 0,
    }
    stored = vs.upsert_chunks([chunk])
    logger.info(f"Stored {research_type} research for {company}")
    return {"stored": stored, "research_type": research_type, "company": company}


@tool
def retrieve_research_tool(company: str, research_type: str) -> dict[str, Any]:
    """Retrieve previously stored research for a company from ChromaDB."""
    vs = _get_vs("research")
    try:
        results = vs.collection.get(
            where={"$and": [{"company": company}, {"research_type": research_type}]},
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [])
        if not docs:
            return {"found": False, "company": company, "research_type": research_type}
        return {
            "found": True,
            "company": company,
            "research_type": research_type,
            "data": json.loads(docs[0]) if docs else {},
        }
    except Exception as e:
        return {"found": False, "error": str(e)}


@tool
def similarity_search_tool(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Semantic search across all stored research data."""
    from backend.rag.embedder import embed_text

    vs = _get_vs("research")
    query_vec = embed_text(query)
    results = vs.query(query_vec, n_results=n_results)
    return [
        {
            "content": r["content"][:500],
            "score": r["score"],
            "research_type": r["metadata"].get("research_type"),
            "company": r["metadata"].get("company"),
        }
        for r in results
    ]
