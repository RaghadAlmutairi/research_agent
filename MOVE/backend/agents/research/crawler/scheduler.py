"""
Scheduler: accepts a list of company URLs and drives the full
Company URL → CrawlManager → Cleaner → Chunker → Embedder → ChromaDB pipeline.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .crawl_manager import CrawlManager
from .tasks import process_urls
from backend.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    url: str
    status: str
    pages: int = 0
    chunks_stored: int = 0
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None


class Scheduler:
    """
    High-level entry point.

    Usage:
        scheduler = Scheduler(firecrawl_api_key="fc-...")
        results = asyncio.run(scheduler.run(["https://company.com"]))
    """

    def __init__(
        self,
        firecrawl_api_key: str | None = None,
        collection_name: str = "knowledge_base",
        chroma_persist_dir: str = "./chroma_db",
        requests_per_second: float = 2.0,
        concurrency: int = 5,
    ):
        self.manager = CrawlManager(
            firecrawl_api_key=firecrawl_api_key,
            requests_per_second=requests_per_second,
        )
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=chroma_persist_dir,
        )
        self.concurrency = concurrency

    async def run(self, urls: list[str]) -> list[CrawlResult]:
        """Run the full pipeline for all company URLs."""
        logger.info(f"Scheduler starting: {len(urls)} URL(s), concurrency={self.concurrency}")

        raw_results = await process_urls(
            urls=urls,
            manager=self.manager,
            vector_store=self.vector_store,
            concurrency=self.concurrency,
        )

        results = []
        for r in raw_results:
            result = CrawlResult(
                url=r["url"],
                status=r["status"],
                pages=r.get("pages", 0),
                chunks_stored=r.get("chunks_stored", 0),
                error=r.get("error"),
                finished_at=datetime.utcnow(),
            )
            results.append(result)
            logger.info(f"  {result.url} → {result.status} ({result.chunks_stored} chunks)")

        total = sum(r.chunks_stored for r in results)
        logger.info(f"Done. Knowledge base now has ~{self.vector_store.count()} total chunks (added {total}).")
        return results
