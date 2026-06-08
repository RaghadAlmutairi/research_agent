"""
Market Insight Scanner
Goal: Understand market, customers, identify opportunities, estimate demand.

Flow: Company Website → Extract Products → Find Industry → Customer Segments
      → Pain Points → Generate Market Insight Report
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def run(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: scan company website and build market insight."""
    url = state["company_url"]
    company_name = state.get("company_name", url)
    logger.info(f"[MarketInsightScanner] Starting for {url}")

    from backend.agents.research.tools.website_scraper import (
        website_scraper_tool,
        company_context_extractor_tool,
    )

    # 1. Scrape website
    scraped = website_scraper_tool.invoke({"url": url})
    pages = scraped.get("pages", [])
    combined_content = "\n\n".join(
        f"[{p['url']}]\n{p['content']}" for p in pages[:10]
    )

    # 2. Extract company context
    context = company_context_extractor_tool.invoke({
        "company_name": company_name,
        "scraped_content": combined_content,
    })

    market_insights = {
        "industry": context.get("industry", "Unknown"),
        "target_market": context.get("target_market", []),
        "products": context.get("products", []),
        "key_features": context.get("key_features", []),
        "pricing_model": context.get("pricing_model", "unknown"),
        "unique_positioning": context.get("unique_positioning", ""),
        "market_pain_points": [],       # filled by review miner
        "opportunities": [],            # filled by gap detector
    }

    logger.info(f"[MarketInsightScanner] Industry: {context.get('industry')}")

    return {
        **state,
        "company_context": context,
        "market_insights": market_insights,
        "scraped_pages": pages,
    }
