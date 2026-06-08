"""
Crawl4AI adapter.

On Windows, Playwright requires ProactorEventLoop to spawn subprocesses,
but our main loop may be SelectorEventLoop (set to suppress pipe-close noise).
Solution: run Crawl4AI in a dedicated background thread that owns its own
ProactorEventLoop, so Playwright is always happy regardless of what the
main loop is.
"""
import asyncio
import concurrent.futures
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

try:
    import crawl4ai  # noqa: F401
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    _CRAWL4AI_AVAILABLE = False

# Shared thread pool — one thread is enough; Crawl4AI is I/O-bound
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="crawl4ai")


def _run_in_proactor(coro):
    """
    Run an async coroutine in a fresh ProactorEventLoop on a background thread.
    This isolates Playwright's subprocess requirements from the main event loop.
    """
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _crawl4ai_scrape_sync(url: str) -> dict[str, Any]:
    """The actual Crawl4AI call — must run inside a ProactorEventLoop thread."""
    from crawl4ai import AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return {
            "url": url,
            "content": result.markdown or "",
            "metadata": {"title": getattr(result, "title", "")},
            "source": "crawl4ai",
        }


class Crawl4AIAdapter:
    """Fallback crawler. Runs in a dedicated thread with ProactorEventLoop on Windows."""

    def _check(self):
        if not _CRAWL4AI_AVAILABLE:
            raise ImportError(
                "crawl4ai is not installed.\n"
                "  Fix: pip install crawl4ai\n"
                "       playwright install"
            )

    async def scrape(self, url: str) -> dict[str, Any]:
        self._check()
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _executor,
                lambda: _run_in_proactor(_crawl4ai_scrape_sync(url)),
            )
            return result
        except Exception as exc:
            msg = str(exc).lower()
            # Convert setup/config errors to ImportError so retry skips them
            if any(kw in msg for kw in (
                "executable doesn't exist", "playwright install",
                "browsertype.launch", "notimplementederror",
            )):
                raise ImportError(
                    f"Crawl4AI browser not configured.\n"
                    f"  Run: playwright install\n"
                    f"  Details: {exc}"
                ) from exc
            raise

    async def crawl(self, url: str, limit: int = 100) -> list[dict[str, Any]]:
        page = await self.scrape(url)
        return [page]
