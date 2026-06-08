import json, logging, os, re
from typing import Any
from langchain_core.tools import tool
from openai import OpenAI
from backend.utils import run_async

logger = logging.getLogger(__name__)

def _llm(): return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def _ask(system, user):
    resp = _llm().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
def _parse(raw): return json.loads(re.sub(r"```json|```","",raw).strip())


@tool
def competitor_discovery_tool(company_name: str, industry: str, description: str) -> dict[str, Any]:
    """Discover top competitors for a company using AI reasoning."""
    prompt = f"""Company: {company_name}\nIndustry: {industry}\nDescription: {description}
List top 5 direct competitors. Return ONLY valid JSON:
{{"competitors":[{{"name":"CompanyA","website":"https://companya.com","reason":"..."}}]}}
Only real companies with actual websites."""
    raw = _ask("Competitive intelligence expert. Return valid JSON only.", prompt)
    try: return _parse(raw)
    except: return {"competitors": [], "raw": raw}


@tool
def competitor_scraper_tool(competitor_name: str, competitor_url: str) -> dict[str, Any]:
    """Scrape a competitor's key pages and extract structured intelligence."""
    from backend.agents.research.crawler.crawl_manager import CrawlManager
    from backend.agents.research.crawler.cleaner import clean_content, is_meaningful, is_noise_url
    from backend.config.settings import settings

    async def _crawl():
        mgr = CrawlManager(firecrawl_api_key=settings.firecrawl_api_key or None)
        return await mgr.crawl(competitor_url, limit=30)

    try: pages = run_async(_crawl())
    except Exception as e: return {"name": competitor_name, "error": str(e)}

    content_parts = []
    for page in pages:
        if is_noise_url(page.get("url","")): continue
        c = clean_content(page.get("content",""))
        if is_meaningful(c): content_parts.append(f"[{page['url']}]\n{c[:1500]}")

    combined = "\n\n".join(content_parts[:10])
    prompt = f"""Extract competitive intelligence for "{competitor_name}".
Return ONLY valid JSON:
{{"name":"{competitor_name}","description":"...","key_features":[],"pricing":"...","pricing_tiers":[{{"name":"tier","price":"$X/mo","features":[]}}],"target_market":[],"positioning":"...","strengths":[],"weaknesses":[],"recent_updates":[]}}
Content:\n{combined[:5000]}"""
    raw = _ask("Extract competitive intelligence. Return valid JSON only.", prompt)
    try:
        result = _parse(raw); result["url"] = competitor_url; return result
    except: return {"name": competitor_name, "url": competitor_url, "raw": raw}


@tool
def competitor_comparison_tool(our_company: dict, competitors: list) -> dict[str, Any]:
    """Compare our company against scraped competitors."""
    prompt = f"""OUR COMPANY:\n{json.dumps(our_company,indent=2)[:2000]}\nCOMPETITORS:\n{json.dumps(competitors,indent=2)[:4000]}
Return ONLY valid JSON:
{{"our_advantages":[],"competitor_advantages":[],"feature_gaps":[],"pricing_position":"cheaper/similar/expensive","market_gaps":[],"differentiation_opportunities":[],"threats":[]}}"""
    raw = _ask("Strategic competitive analyst. Return valid JSON only.", prompt)
    try: return _parse(raw)
    except: return {"raw_comparison": raw}


@tool
def pricing_analysis_tool(competitors: list) -> dict[str, Any]:
    """Analyze pricing strategies across competitors."""
    comp_str = json.dumps([{"name":c.get("name"),"pricing":c.get("pricing"),"tiers":c.get("pricing_tiers")} for c in competitors],indent=2)
    prompt = f"""Analyze pricing across these competitors.\n{comp_str[:3000]}
Return ONLY valid JSON:
{{"market_price_range":{{"low":"$X/mo","mid":"$Y/mo","high":"$Z/mo"}},"common_pricing_models":[],"pricing_trends":[],"opportunities":[],"recommended_positioning":"..."}}"""
    raw = _ask("Pricing strategy analyst. Return valid JSON only.", prompt)
    try: return _parse(raw)
    except: return {"raw": raw}
