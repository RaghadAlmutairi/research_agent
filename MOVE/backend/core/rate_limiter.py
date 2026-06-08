import asyncio
import time
from collections import defaultdict


class DomainRateLimiter:
    """Per-domain token-bucket rate limiter."""

    def __init__(self, requests_per_second: float = 2.0):
        self.delay = 1.0 / requests_per_second
        self._last_request: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _domain(self, url: str) -> str:
        from urllib.parse import urlparse
        return urlparse(url).netloc or url

    async def wait(self, url: str = "") -> None:
        """Wait until it is safe to make another request to this domain."""
        domain = self._domain(url) if url else "__default__"
        async with self._locks[domain]:
            now = time.monotonic()
            elapsed = now - self._last_request[domain]
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)
            self._last_request[domain] = time.monotonic()
