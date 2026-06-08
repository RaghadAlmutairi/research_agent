"""
trend_collector_tool  — collect industry trends
trend_clustering_tool — cluster and rank trends
signal_detection_tool — detect early signals and emerging tech
"""
import json
import logging
import os
import re
from typing import Any
from langchain_core.tools import tool
from openai import OpenAI

logger = logging.getLogger(__name__)


def _llm():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _ask(system: str, user: str) -> str:
    resp = _llm().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> Any:
    return json.loads(re.sub(r"```json|```", "", raw).strip())


@tool
def trend_collector_tool(industry: str, company_description: str) -> dict[str, Any]:
    """
    Collect current market trends for an industry.
    Synthesizes signals from TechCrunch, Product Hunt, Crunchbase, and industry news.
    """
    prompt = f"""You are a market trends analyst with knowledge of TechCrunch, Product Hunt, Crunchbase, and industry news up to 2025.

Industry: {industry}
Company context: {company_description}

Identify the top trends relevant to this company. Return ONLY valid JSON:
{{
  "trends": [
    {{
      "name": "AI-powered automation",
      "description": "...",
      "growth": "high/medium/low",
      "confidence": 0.91,
      "time_horizon": "now/6-12mo/1-3yr",
      "sources": ["TechCrunch", "Product Hunt"],
      "impact_on_company": "direct/indirect/disruptive"
    }}
  ],
  "macro_shifts": ["shift1", "shift2"],
  "technologies_to_watch": ["tech1", "tech2"]
}}"""

    raw = _ask("Market trends analyst. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"industry": industry, "raw": raw}


@tool
def trend_clustering_tool(trends_data: dict) -> dict[str, Any]:
    """Cluster and rank trends by strategic importance and urgency."""
    prompt = f"""Cluster and prioritize these trends by strategic importance.

Trends:
{json.dumps(trends_data, indent=2)[:3000]}

Return ONLY valid JSON:
{{
  "clusters": [
    {{
      "theme": "AI & Automation",
      "trends": ["trend1", "trend2"],
      "priority": "critical/high/medium/low",
      "urgency": "act now/plan for/monitor",
      "strategic_implication": "..."
    }}
  ],
  "top_priority_trends": ["trend1", "trend2", "trend3"],
  "ignore_for_now": ["trend that's noise"]
}}"""

    raw = _ask("Strategic trend analyst. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"raw": raw}


@tool
def signal_detection_tool(industry: str, trends_data: dict) -> dict[str, Any]:
    """Detect weak/early signals of emerging technologies and market shifts."""
    prompt = f"""You are an early-signal detection specialist in the {industry} industry.

Based on these trends, identify emerging signals that most analysts are NOT yet talking about.

Current trends:
{json.dumps(trends_data, indent=2)[:2000]}

Return ONLY valid JSON:
{{
  "early_signals": [
    {{
      "signal": "...",
      "evidence": "...",
      "growth_trajectory": "exponential/linear/uncertain",
      "time_to_mainstream": "6mo/1yr/2yr/3yr+",
      "opportunity_window": "narrow/moderate/wide"
    }}
  ],
  "emerging_technologies": [
    {{
      "technology": "...",
      "maturity": "research/early-adopter/growing",
      "relevance": "high/medium/low"
    }}
  ],
  "contrarian_views": ["unexpected angle1"]
}}"""

    raw = _ask("Early-signal detection expert. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"raw": raw}
