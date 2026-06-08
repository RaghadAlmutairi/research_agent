import logging, os, re
from typing import Any
from langchain_core.tools import tool
from openai import OpenAI
from backend.utils import run_async

logger = logging.getLogger(__name__)
_client = None

def _llm():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def _ask(system, user, model="gpt-4o-mini"):
    resp = _llm().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


@tool
def website_scraper_tool(url: str) -> dict[str, Any]:
    """Scrape a company website and return raw page content for all key pages."""
    from backend.agents.research.crawler.crawl_manager import CrawlManager
    from backend.agents.research.crawler.cleaner import clean_content, is_meaningful, is_noise_url
    from backend.config.settings import settings

    async def _crawl():
        mgr = CrawlManager(firecrawl_api_key=settings.firecrawl_api_key or None)
        return await mgr.crawl(url, limit=50)

    pages = run_async(_crawl())
    cleaned = []
    for p in pages:
        if is_noise_url(p.get("url", "")):
            continue
        content = clean_content(p.get("content", ""))
        if is_meaningful(content):
            cleaned.append({"url": p["url"], "content": content[:3000]})
    return {"url": url, "pages_found": len(cleaned), "pages": cleaned[:20]}


@tool
def website_discovery_tool(url: str) -> dict[str, Any]:
    """Discover key pages of a company website (pricing, features, about, blog, integrations)."""
    from backend.agents.research.crawler.crawl_manager import CrawlManager
    from backend.config.settings import settings

    KEY_PATHS = ["pricing", "features", "product", "about", "blog", "integrations", "solutions", "customers"]

    async def _scrape():
        mgr = CrawlManager(firecrawl_api_key=settings.firecrawl_api_key or None)
        return await mgr.scrape(url)

    home = run_async(_scrape())
    content = home.get("content", "")
    found_urls = re.findall(r'https?://[^\s)>"\']+', content)
    key_pages = {}
    for u in found_urls:
        for path in KEY_PATHS:
            if path in u.lower() and path not in key_pages:
                key_pages[path] = u
    return {"base_url": url, "key_pages": key_pages}


@tool
def company_context_extractor_tool(company_name: str, scraped_content: str) -> dict[str, Any]:
    """Extract structured company context from scraped content."""
    import json
    prompt = f"""Extract structured information for "{company_name}".
Return ONLY valid JSON:
{{
  "company_name": "...", "industry": "...", "description": "2-3 sentence summary",
  "products": [], "key_features": [], "target_market": [],
  "pricing_model": "freemium/subscription/one-time/enterprise/unknown",
  "pricing_details": "...", "unique_positioning": "...", "technologies_used": []
}}
Website content:\n{scraped_content[:6000]}"""
    raw = _ask("Extract structured business intelligence. Always return valid JSON.", prompt)
    try:
        return json.loads(re.sub(r"```json|```", "", raw).strip())
    except Exception:
        return {"company_name": company_name, "raw_summary": raw}
