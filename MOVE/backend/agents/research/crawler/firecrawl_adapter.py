import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _obj_to_dict(obj: Any) -> dict:
    """Convert any object to a plain dict, handling Pydantic models, dicts, and plain objects."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    # Dataclass / plain object
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {}


class FirecrawlAdapter:
    """Primary crawler using the Firecrawl API."""

    def __init__(self, api_key: str):
        try:
            from firecrawl import FirecrawlApp
            self.client = FirecrawlApp(api_key=api_key)
            self.available = True
        except ImportError:
            logger.warning("firecrawl-py not installed. FirecrawlAdapter disabled.")
            self.client = None
            self.available = False

    def _page_to_dict(self, raw: Any, fallback_url: str = "") -> dict[str, Any]:
        """Normalize a single page result to a plain dict."""
        # Handle tuple responses (some SDK versions)
        if isinstance(raw, (tuple, list)):
            raw = raw[1] if len(raw) >= 2 else (raw[0] if raw else {})

        d = _obj_to_dict(raw)

        # metadata can itself be a Pydantic model (DocumentMetadata)
        metadata = d.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            metadata = _obj_to_dict(metadata)
            d["metadata"] = metadata

        return d

    def _extract_pages(self, result: Any, fallback_url: str) -> list[dict[str, Any]]:
        d = self._page_to_dict(result, fallback_url)

        pages_raw = d.get("data") or d.get("pages") or d.get("results")

        if isinstance(pages_raw, list):
            out = []
            for p in pages_raw:
                p = self._page_to_dict(p, fallback_url)
                meta = p.get("metadata") or {}
                out.append({
                    "url": meta.get("url") or meta.get("sourceURL") or fallback_url,
                    "content": p.get("markdown") or p.get("content") or "",
                    "metadata": meta,
                    "source": "firecrawl",
                })
            return out

        # Single page
        meta = d.get("metadata") or {}
        return [{
            "url": meta.get("url") or meta.get("sourceURL") or fallback_url,
            "content": d.get("markdown") or d.get("content") or "",
            "metadata": meta,
            "source": "firecrawl",
        }]

    async def scrape(self, url: str) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("Firecrawl is not available.")
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, lambda: self.client.scrape_url(url, formats=["markdown"])
        )
        pages = self._extract_pages(raw, url)
        return pages[0] if pages else {"url": url, "content": "", "metadata": {}, "source": "firecrawl"}

    async def crawl(self, url: str, limit: int = 100) -> list[dict[str, Any]]:
        if not self.available:
            raise RuntimeError("Firecrawl is not available.")
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self.client.crawl_url(
                url, limit=limit, scrape_options={"formats": ["markdown"]}
            ),
        )
        return self._extract_pages(raw, url)
