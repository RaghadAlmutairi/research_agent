"""
web_searcher.py
Finds competitor URLs from the web using multiple search strategies:
  1. LLM knowledge base (instant, no API needed)
  2. Site mention extraction from the target company's own website
  3. Google-style search via SerpAPI (if key is set)
"""
import asyncio
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


def _llm_discover(company_name: str, industry: str, description: str) -> list[dict[str, Any]]:
    """Use OpenAI to generate a list of known competitors with URLs."""
    import json
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""You are a competitive intelligence expert.

Company: {company_name}
Industry: {industry}
Description: {description}

List the top 8 direct competitors. For each, provide the real, working homepage URL.
Return ONLY valid JSON — no markdown, no explanation:
{{
  "competitors": [
    {{"name": "CompanyA", "website": "https://companya.com", "reason": "direct competitor because..."}},
    {{"name": "CompanyB", "website": "https://companyb.com", "reason": "..."}}
  ]
}}

Rules:
- Only real companies with actual websites
- Use https:// URLs
- No social media, no directories (LinkedIn, Crunchbase, G2, etc.)
- No placeholders like example.com"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Competitive intelligence expert. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
    try:
        data = json.loads(raw)
        return data.get("competitors", [])
    except Exception:
        return []


def _extract_from_website(scraped_content: str, base_domain: str) -> list[str]:
    """Extract external company URLs mentioned in the target company's own website."""
    all_urls = re.findall(r'https?://(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})[^\s)>"\']*', scraped_content)

    # Filter out the company's own domain and common noise
    noise = {
        "google", "facebook", "twitter", "linkedin", "youtube", "instagram",
        "wp-content", "googleapis", "cloudflare", "amazonaws", "fonts",
        "gravatar", "placeholder", "elementor", "w3", "schema",
    }
    external = []
    for domain in set(all_urls):
        root = domain.split(".")[0].lower()
        if base_domain.lower() not in domain.lower() and root not in noise:
            external.append(f"https://{domain}")

    return external[:15]


async def search_serpapi(query: str) -> list[str]:
    """Search via SerpAPI if SERPAPI_KEY is configured."""
    api_key = os.getenv("SERPAPI_KEY", "")
    if not api_key:
        return []
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={"q": query, "api_key": api_key, "num": 10},
            )
            results = resp.json().get("organic_results", [])
            return [r.get("link", "") for r in results if r.get("link")]
    except Exception as e:
        logger.warning(f"SerpAPI search failed: {e}")
        return []


async def find_competitor_urls(
    company_name: str,
    industry: str,
    description: str,
    scraped_content: str = "",
    base_domain: str = "",
) -> list[dict[str, Any]]:
    """
    Multi-strategy competitor discovery.
    Returns a deduplicated list of competitor dicts with name, website, reason, source.
    """
    found: dict[str, dict] = {}  # website → data

    # Strategy 1: LLM knowledge
    logger.info("[WebSearcher] Strategy 1: LLM knowledge base")
    llm_results = _llm_discover(company_name, industry, description)
    for c in llm_results:
        website = c.get("website", "").rstrip("/")
        if website:
            found[website] = {**c, "source": "llm_knowledge"}

    # Strategy 2: Extract from company website
    if scraped_content and base_domain:
        logger.info("[WebSearcher] Strategy 2: Website mention extraction")
        extracted = _extract_from_website(scraped_content, base_domain)
        for url in extracted:
            if url not in found:
                found[url] = {"name": url, "website": url, "reason": "mentioned on target site", "source": "website_extraction"}

    # Strategy 3: SerpAPI (optional)
    if os.getenv("SERPAPI_KEY"):
        logger.info("[WebSearcher] Strategy 3: SerpAPI search")
        query = f'"{industry}" competitors alternatives to "{company_name}"'
        serp_urls = await search_serpapi(query)
        for url in serp_urls:
            if url not in found:
                found[url] = {"name": url, "website": url, "reason": "search result", "source": "serpapi"}

    return list(found.values())
