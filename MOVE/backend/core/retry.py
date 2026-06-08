import asyncio
import logging
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

_NO_RETRY_TYPES = (ImportError, NotImplementedError)

_NO_RETRY_KEYWORDS = (
    "unauthorized", "invalid token", "invalid api", "forbidden",
    "playwright install", "executable doesn't exist",
    "browser has not been launched", "not installed",
    "no crawler available",
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _NO_RETRY_TYPES):
        return False
    msg = str(exc).lower()
    return not any(kw in msg for kw in _NO_RETRY_KEYWORDS)


async def crawl_with_retry(
    crawler: Callable[[str], Awaitable[Any]],
    url: str,
    max_attempts: int = 3,
) -> Any:
    """Exponential-backoff retry. Auth/setup errors fail immediately."""
    delays = [2, 4, 8]
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await crawler(url)
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            if attempt < max_attempts:
                delay = delays[min(attempt - 1, len(delays) - 1)]
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed for {url}: "
                    f"{type(exc).__name__}: {str(exc)[:120]}. Retrying in {delay}s…"
                )
                await asyncio.sleep(delay)

    raise last_exc
