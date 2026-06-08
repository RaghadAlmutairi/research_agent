"""
CrawlManager: Firecrawl (primary) → Crawl4AI (fallback)
"""
import logging
from typing import Any

from .firecrawl_adapter import FirecrawlAdapter
from .crawl4ai_adapter import Crawl4AIAdapter
from backend.core.rate_limiter import DomainRateLimiter
from backend.core.retry import crawl_with_retry

logger = logging.getLogger(__name__)

_AUTH_KEYWORDS = ("unauthorized", "invalid token", "invalid api", "forbidden")
_SETUP_KEYWORDS = (
    "executable doesn't exist", "playwright install",
    "browsertype.launch", "browser has not been launched",
    "crawl4ai browser not configured",
)


def _is_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _AUTH_KEYWORDS)


def _is_setup_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, ImportError) or any(kw in msg for kw in _SETUP_KEYWORDS)


class CrawlManager:

    def __init__(
        self,
        firecrawl_api_key: str | None = None,
        requests_per_second: float = 2.0,
    ):
        self.firecrawl = FirecrawlAdapter(api_key=firecrawl_api_key or "")
        self.crawl4ai = Crawl4AIAdapter()
        self.rate_limiter = DomainRateLimiter(requests_per_second=requests_per_second)
        self._crawl4ai_broken = False

        if not firecrawl_api_key:
            logger.info("No FIRECRAWL_API_KEY — using Crawl4AI directly.")
            self.firecrawl.available = False

    async def _try_firecrawl_crawl(self, url: str, limit: int) -> list[dict[str, Any]] | None:
        if not self.firecrawl.available:
            return None
        try:
            logger.info(f"[Firecrawl] crawling {url} (limit={limit})")
            return await crawl_with_retry(
                lambda u: self.firecrawl.crawl(u, limit=limit), url
            )
        except Exception as exc:
            if _is_auth_error(exc):
                logger.warning(f"[Firecrawl] auth error — disabling for this session. ({exc})")
                self.firecrawl.available = False
            else:
                logger.warning(f"[Firecrawl] failed ({type(exc).__name__}): {exc}. Falling back.")
            return None

    async def _try_crawl4ai(self, url: str) -> list[dict[str, Any]] | None:
        if self._crawl4ai_broken:
            return None
        try:
            logger.info(f"[Crawl4AI] crawling {url}")
            return await crawl_with_retry(self.crawl4ai.crawl, url)
        except Exception as exc:
            if _is_setup_error(exc):
                logger.error(
                    f"[Crawl4AI] setup error — run: playwright install\n"
                    f"  Details: {exc}"
                )
                self._crawl4ai_broken = True
            else:
                logger.error(f"[Crawl4AI] failed: {exc}")
            return None

    async def crawl(self, url: str, limit: int = 100) -> list[dict[str, Any]]:
        await self.rate_limiter.wait(url)

        result = await self._try_firecrawl_crawl(url, limit)
        if result is not None:
            return result

        result = await self._try_crawl4ai(url)
        if result is not None:
            return result

        raise RuntimeError(
            f"All crawlers failed for {url}.\n"
            "  → Check FIRECRAWL_API_KEY in .env, or run: playwright install"
        )

    async def scrape(self, url: str) -> dict[str, Any]:
        await self.rate_limiter.wait(url)
        pages = await self.crawl(url, limit=1)
        return pages[0] if pages else {"url": url, "content": "", "metadata": {}, "source": "none"}
