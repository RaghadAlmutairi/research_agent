"""
Competitor Tracker
Goal: Track competitors, pricing, positioning, and feature changes.

Flow: Discover Competitors → Scrape Each → Compare → Analyze Pricing
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def run(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: discover and profile competitors."""
    context = state.get("company_context", {})
    company_name = context.get("company_name", state.get("company_name", "Unknown"))
    industry = context.get("industry", "Technology")
    description = context.get("description", "")

    logger.info(f"[CompetitorTracker] Discovering competitors for {company_name}")

    from backend.agents.research.tools.competitor_scraper import (
        competitor_discovery_tool,
        competitor_scraper_tool,
        competitor_comparison_tool,
        pricing_analysis_tool,
    )

    # 1. Discover competitors
    discovery = competitor_discovery_tool.invoke({
        "company_name": company_name,
        "industry": industry,
        "description": description,
    })
    competitors_list = discovery.get("competitors", [])
    logger.info(f"[CompetitorTracker] Found {len(competitors_list)} competitors")

    # 2. Scrape each competitor (limit to top 3 to save time/cost)
    scraped_competitors = []
    for comp in competitors_list[:3]:
        name = comp.get("name", "")
        website = comp.get("website", "")
        if not website:
            continue
        logger.info(f"[CompetitorTracker] Scraping {name}")
        try:
            data = competitor_scraper_tool.invoke({
                "competitor_name": name,
                "competitor_url": website,
            })
            scraped_competitors.append(data)
        except Exception as e:
            logger.warning(f"Failed to scrape {name}: {e}")
            scraped_competitors.append({"name": name, "url": website, "error": str(e)})

    # 3. Compare
    comparison = {}
    if scraped_competitors:
        comparison = competitor_comparison_tool.invoke({
            "our_company": context,
            "competitors": scraped_competitors,
        })

    # 4. Pricing analysis
    pricing = {}
    if scraped_competitors:
        pricing = pricing_analysis_tool.invoke({"competitors": scraped_competitors})

    logger.info(f"[CompetitorTracker] Done. {len(scraped_competitors)} competitors profiled.")

    return {
        **state,
        "competitors_discovered": competitors_list,
        "competitors_scraped": scraped_competitors,
        "competitor_comparison": comparison,
        "pricing_analysis": pricing,
    }
