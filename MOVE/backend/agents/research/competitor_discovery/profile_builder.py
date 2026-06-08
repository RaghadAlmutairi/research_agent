import logging, json, os, re
from typing import Any
from backend.utils import run_async

logger = logging.getLogger(__name__)


def _llm_extract(company_name: str, content: str) -> dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""Extract a structured competitor profile for "{company_name}".
Website content:\n{content[:5000]}
Return ONLY valid JSON:
{{"name":"...","tagline":"...","description":"...","industry":"...","target_market":[],"key_features":[],"pricing_model":"unknown","pricing_tiers":[{{"name":"Free","price":"$0/mo","features":[]}}],"positioning":"...","strengths":[],"weaknesses":[],"technologies":[],"integrations":[],"recent_updates":[]}}"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"Competitive intelligence analyst. Return valid JSON only."},{"role":"user","content":prompt}],
        temperature=0.1,
    )
    raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
    try: return json.loads(raw)
    except: return {"name": company_name, "raw_summary": raw}


async def build_profile(competitor_url: str, competitor_name: str = "") -> dict[str, Any]:
    """Scrape a competitor site and return a structured profile dict."""
    from backend.agents.research.crawler.crawl_manager import CrawlManager
    from backend.agents.research.crawler.cleaner import clean_content, is_meaningful, is_noise_url
    from backend.config.settings import settings
    from urllib.parse import urlparse

    name = competitor_name or urlparse(competitor_url).netloc.replace("www.", "")
    logger.info(f"[ProfileBuilder] {name}")

    try:
        mgr = CrawlManager(firecrawl_api_key=settings.firecrawl_api_key or None)
        pages = await mgr.crawl(competitor_url, limit=20)
    except Exception as e:
        return {"name": name, "website": competitor_url, "error": str(e)}

    KEY_PATHS = ["pricing","features","product","about","home","solutions"]
    parts = []
    for page in pages:
        if is_noise_url(page.get("url","")): continue
        c = clean_content(page.get("content",""))
        if is_meaningful(c):
            priority = any(k in page.get("url","").lower() for k in KEY_PATHS)
            parts.append((priority, f"[{page['url']}]\n{c[:1200]}"))

    parts.sort(key=lambda x: not x[0])
    combined = "\n\n".join(c for _, c in parts[:10])

    if not combined.strip():
        return {"name": name, "website": competitor_url, "error": "no content extracted"}

    profile = _llm_extract(name, combined)
    profile["website"] = competitor_url
    profile["pages_scraped"] = len(parts)
    return profile
