"""
Trend Discovery Engine
Goal: Identify trends, emerging technologies, and new opportunities.

Sources: TechCrunch, Product Hunt, Crunchbase, Company Blogs, Industry News
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def run(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: collect and analyze market trends."""
    context = state.get("company_context", {})
    industry = context.get("industry", "Technology")
    description = context.get("description", "")

    logger.info(f"[TrendDiscoveryEngine] Scanning trends for {industry}")

    from backend.agents.research.tools.trend_collector import (
        trend_collector_tool,
        trend_clustering_tool,
        signal_detection_tool,
    )
    from backend.agents.research.tools.review_miner import (
        review_collector_tool,
        sentiment_analysis_tool,
        pain_point_extractor_tool,
        feature_request_extractor_tool,
    )
    from backend.agents.research.tools.gap_detector import (
        gap_detector_tool,
        opportunity_scorer_tool,
    )

    company_name = context.get("company_name", "Unknown")

    # 1. Collect trends
    trends_raw = trend_collector_tool.invoke({
        "industry": industry,
        "company_description": description,
    })

    # 2. Cluster trends
    clustered = trend_clustering_tool.invoke({"trends_data": trends_raw})

    # 3. Early signals
    signals = signal_detection_tool.invoke({
        "industry": industry,
        "trends_data": clustered,
    })

    # 4. Reviews & customer voice
    reviews = review_collector_tool.invoke({
        "company_name": company_name,
        "industry": industry,
    })
    sentiment = sentiment_analysis_tool.invoke({"reviews_data": reviews})
    pain_points = pain_point_extractor_tool.invoke({
        "company_name": company_name,
        "reviews_data": reviews,
    })
    feature_requests = feature_request_extractor_tool.invoke({
        "company_name": company_name,
        "reviews_data": reviews,
    })

    # 5. Gap detection
    gaps = gap_detector_tool.invoke({
        "company_context": context,
        "competitors": state.get("competitors_scraped", []),
        "reviews": pain_points,
        "trends": clustered,
    })

    # 6. Score opportunities
    opportunities = opportunity_scorer_tool.invoke({
        "gaps": gaps,
        "trends": clustered,
        "company_context": context,
    })

    logger.info("[TrendDiscoveryEngine] Done.")

    return {
        **state,
        "trends_raw": trends_raw,
        "trends_clustered": clustered,
        "early_signals": signals,
        "reviews": reviews,
        "sentiment": sentiment,
        "pain_points": pain_points,
        "feature_requests": feature_requests,
        "market_gaps": gaps,
        "scored_opportunities": opportunities,
    }
