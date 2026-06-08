"""
url_validator.py
Validates and resolves competitor URLs before scraping.
Checks: reachable, is a real company site, not a directory/listing page.
"""
import re
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_BLACKLIST_DOMAINS = {
    "linkedin.com", "twitter.com", "facebook.com", "instagram.com",
    "youtube.com", "wikipedia.org", "crunchbase.com", "g2.com",
    "capterra.com", "trustpilot.com", "glassdoor.com", "reddit.com",
    "yelp.com", "amazon.com", "google.com", "bing.com",
}


def is_blacklisted(url: str) -> bool:
    """Return True if URL is a social/directory site, not a company site."""
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower().lstrip("www.")
    return any(domain == b or domain.endswith("." + b) for b in _BLACKLIST_DOMAINS)


def normalize_url(url: str) -> str:
    """Ensure URL has a scheme and trailing slash."""
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    return url


async def is_reachable(url: str, timeout: int = 8) -> bool:
    """Async check if a URL returns a 2xx/3xx response."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.head(url)
            return resp.status_code < 400
    except Exception:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url)
                return resp.status_code < 400
        except Exception:
            return False


async def validate_urls(urls: list[str]) -> list[str]:
    """Filter a list of competitor URLs to only valid, reachable, non-blacklisted ones."""
    valid = []
    tasks = []

    normalized = [normalize_url(u) for u in urls if u]
    filtered = [u for u in normalized if not is_blacklisted(u)]

    results = await asyncio.gather(*[is_reachable(u) for u in filtered], return_exceptions=True)

    for url, ok in zip(filtered, results):
        if ok is True:
            valid.append(url)
            logger.debug(f"  ✓ {url}")
        else:
            logger.debug(f"  ✗ {url} (unreachable or error)")

    return valid
