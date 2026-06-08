"""
Async task queue: CrawlManager → Cleaner → Chunker → Embedder → VectorStore
"""
import asyncio
import logging
from typing import Any

from .crawl_manager import CrawlManager
from .cleaner import clean_content, compute_hash, is_meaningful, is_noise_url
from .entity_extractor import enrich_chunks
from backend.rag.chunker import chunk_pages
from backend.rag.embedder import embed_chunks
from backend.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def process_url(
    url: str,
    manager: CrawlManager,
    vector_store: VectorStore,
) -> dict[str, Any]:
    """Full pipeline for one URL: crawl → clean → chunk → embed → store."""
    try:
        pages = await manager.crawl(url)

        filtered_pages = []
        for page in pages:
            page_url = page.get("url", "")

            # Skip sitemaps, images, assets
            if is_noise_url(page_url):
                logger.debug(f"Skipping noise URL: {page_url}")
                continue

            content = clean_content(page.get("content", ""))
            if not is_meaningful(content):
                logger.debug(f"Skipping low-content page: {page_url}")
                continue

            page["content"] = content
            page["page_hash"] = compute_hash(content)
            filtered_pages.append(page)

        if not filtered_pages:
            return {"url": url, "status": "skipped", "reason": "no meaningful content after filtering"}

        chunks = chunk_pages(filtered_pages)
        chunks = enrich_chunks(chunks)
        chunks = embed_chunks(chunks)
        stored = vector_store.upsert_chunks(chunks)

        logger.info(f"Stored {stored} chunks from {len(filtered_pages)} pages for {url}")
        return {
            "url": url,
            "status": "success",
            "pages": len(filtered_pages),
            "chunks_stored": stored,
        }

    except Exception as exc:
        logger.error(f"Failed to process {url}: {exc}")
        return {"url": url, "status": "error", "error": str(exc)}


async def process_urls(
    urls: list[str],
    manager: CrawlManager,
    vector_store: VectorStore,
    concurrency: int = 5,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(url: str):
        async with sem:
            return await process_url(url, manager, vector_store)

    return await asyncio.gather(*[_bounded(u) for u in urls])
