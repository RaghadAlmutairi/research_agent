"""
Research Agent — LangGraph orchestration

START
  ↓
market_insight_scanner     — scrape company, extract context
  ↓
competitor_discovery       — auto-find + profile competitors
  ↓
competitor_tracker         — compare, pricing analysis
  ↓
trend_discovery_engine     — trends, reviews, pain points, gaps, opportunities
  ↓
research_synthesis         — ResearchReport + insights
  ↓
END
"""
import json
import logging
import os
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from .scanners.market_insight_scanner import run as market_insight_scanner
from .scanners.competitor_tracker import run as competitor_tracker
from .scanners.trend_discovery_engine import run as trend_discovery_engine
from .synthesis.report_builder import report_builder_tool
from .synthesis.insight_extractor import insight_extractor_tool
from .tools.chroma_tools import store_research_tool
from backend.utils import run_async




logger = logging.getLogger(__name__)


def _setup_langsmith():
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "MOVE-Research-Agent")
        logger.info("LangSmith tracing enabled.")


class ResearchState(TypedDict, total=False):
    company_url: str
    company_name: str
    company_context: dict
    market_insights: dict
    scraped_pages: list
    competitors_discovered: list
    competitors_profiled: list
    discovery_summary: str
    competitors_scraped: list
    competitor_comparison: dict
    pricing_analysis: dict
    trends_raw: dict
    trends_clustered: dict
    early_signals: dict
    reviews: dict
    sentiment: dict
    pain_points: dict
    feature_requests: dict
    market_gaps: dict
    scored_opportunities: dict
    report: dict
    insights: dict
    error: str


def competitor_discovery_node(state: ResearchState) -> ResearchState:
    """Auto-discover and profile competitors using CompetitorDiscoveryAgent."""
    import asyncio
    from backend.agents.research.competitor_discovery.discovery_agent import CompetitorDiscoveryAgent

    url = state["company_url"]
    context = state.get("company_context", {})
    name = context.get("company_name") or state.get("company_name") or url

    logger.info(f"[CompetitorDiscovery] Auto-discovering competitors for {name}")

    try:
        agent = CompetitorDiscoveryAgent(max_competitors=5, profile_concurrency=3)
        result = run_async(agent.run(url, name))

        profiles = [p.__dict__ for p in result.competitors]
        logger.info(f"[CompetitorDiscovery] {result.total_found} found → {result.total_profiled} profiled")

        return {
            **state,
            "competitors_discovered": profiles,
            "competitors_profiled": profiles,
            "competitors_scraped": profiles,
            "discovery_summary": result.summary(),
        }
    except Exception as e:
        logger.error(f"[CompetitorDiscovery] Failed: {e}")
        return {
            **state,
            "competitors_discovered": [],
            "competitors_profiled": [],
            "competitors_scraped": [],
            "discovery_summary": f"Discovery failed: {e}",
        }


def research_synthesis(state: ResearchState) -> ResearchState:
    logger.info("[ResearchSynthesis] Building report...")
    try:
        report = report_builder_tool.invoke({
            "company_context":  state.get("company_context", {}),
            "competitor_data":  state.get("competitors_profiled", []),
            "comparison":       state.get("competitor_comparison", {}),
            "reviews":          state.get("reviews", {}),
            "pain_points":      state.get("pain_points", {}),
            "feature_requests": state.get("feature_requests", {}),
            "trends":           state.get("trends_clustered", {}),
            "gaps":             state.get("market_gaps", {}),
            "opportunities":    state.get("scored_opportunities", {}),
        })
        insights = insight_extractor_tool.invoke({"report": report})
        company = state.get("company_name") or state.get("company_url", "unknown")
        store_research_tool.invoke({"research_type": "report",   "company": company, "data": report})
        store_research_tool.invoke({"research_type": "insights", "company": company, "data": insights})
        logger.info("[ResearchSynthesis] Done.")
        return {**state, "report": report, "insights": insights}
    except Exception as e:
        logger.error(f"[ResearchSynthesis] Error: {e}")
        return {**state, "error": str(e)}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("market_insight_scanner",  market_insight_scanner)
    graph.add_node("competitor_discovery",     competitor_discovery_node)
    graph.add_node("competitor_tracker",       competitor_tracker)
    graph.add_node("trend_discovery_engine",   trend_discovery_engine)
    graph.add_node("research_synthesis",       research_synthesis)
    graph.set_entry_point("market_insight_scanner")
    graph.add_edge("market_insight_scanner",  "competitor_discovery")
    graph.add_edge("competitor_discovery",    "competitor_tracker")
    graph.add_edge("competitor_tracker",      "trend_discovery_engine")
    graph.add_edge("trend_discovery_engine",  "research_synthesis")
    graph.add_edge("research_synthesis",      END)
    return graph.compile()


def run_research(company_url: str, company_name: str = "") -> dict[str, Any]:
    """Run the full research pipeline. Competitors are discovered automatically."""
    _setup_langsmith()
    app = build_graph()
    logger.info(f"Research Agent starting for: {company_url}")
    final_state = app.invoke({
        "company_url":  company_url,
        "company_name": company_name or company_url,
    })
    return {
        "report":            final_state.get("report", {}),
        "insights":          final_state.get("insights", {}),
        "competitors":       final_state.get("competitors_profiled", []),
        "trends":            final_state.get("trends_clustered", {}),
        "opportunities":     final_state.get("scored_opportunities", {}),
        "discovery_summary": final_state.get("discovery_summary", ""),
        "error":             final_state.get("error"),
    }
