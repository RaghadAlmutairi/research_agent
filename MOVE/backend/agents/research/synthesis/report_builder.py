"""
ResearchReport schema + report_builder_tool
"""
import json
import os
import re
from typing import Any
from pydantic import BaseModel
from langchain_core.tools import tool
from openai import OpenAI


class ResearchReport(BaseModel):
    company_summary: dict
    market_insights: dict
    competitors: list
    customer_pain_points: list
    customer_desires: list
    trends: list
    opportunities: list
    threats: list
    strategic_recommendations: list


def _ask(system: str, user: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=4000,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> Any:
    return json.loads(re.sub(r"```json|```", "", raw).strip())


@tool
def report_builder_tool(
    company_context: dict,
    competitor_data: list,
    comparison: dict,
    reviews: dict,
    pain_points: dict,
    feature_requests: dict,
    trends: dict,
    gaps: dict,
    opportunities: dict,
) -> dict[str, Any]:
    """Synthesize all research into a structured ResearchReport."""

    prompt = f"""You are a senior market intelligence analyst. Synthesize all research data into a comprehensive report.

COMPANY:
{json.dumps(company_context, indent=2)[:1000]}

COMPETITORS ({len(competitor_data)} found):
{json.dumps([c.get("name") for c in competitor_data], indent=2)}

COMPETITIVE GAPS:
{json.dumps(comparison.get("market_gaps", []), indent=2)}

CUSTOMER PAIN POINTS:
{json.dumps(pain_points.get("top_3_critical", pain_points.get("pain_points", []))[:5], indent=2)}

FEATURE REQUESTS:
{json.dumps(feature_requests.get("quick_wins", []) + feature_requests.get("strategic_features", []), indent=2)[:500]}

TOP TRENDS:
{json.dumps(trends.get("top_priority_trends", trends.get("trends", []))[:5], indent=2)[:800]}

MARKET GAPS:
{json.dumps(gaps.get("market_gaps", [])[:3], indent=2)[:500]}

OPPORTUNITIES:
{json.dumps(opportunities.get("top_opportunity", ""), indent=2)}

Return ONLY valid JSON matching this exact schema:
{{
  "company_summary": {{
    "name": "...",
    "industry": "...",
    "description": "...",
    "positioning": "...",
    "target_market": []
  }},
  "market_insights": {{
    "market_size_estimate": "...",
    "growth_direction": "growing/stable/declining",
    "key_dynamics": [],
    "buyer_behavior": []
  }},
  "competitors": [
    {{"name": "...", "positioning": "...", "key_differentiator": "..."}}
  ],
  "customer_pain_points": ["pain1", "pain2"],
  "customer_desires": ["desire1", "desire2"],
  "trends": [
    {{"trend": "...", "urgency": "now/soon/watch", "impact": "high/medium/low"}}
  ],
  "opportunities": [
    {{"opportunity": "...", "score": 8, "action": "..."}}
  ],
  "threats": ["threat1", "threat2"],
  "strategic_recommendations": [
    {{"priority": 1, "recommendation": "...", "rationale": "...", "timeline": "..."}}
  ]
}}"""

    raw = _ask("Senior market intelligence analyst. Return valid JSON only.", prompt)
    try:
        data = _parse_json(raw)
        report = ResearchReport(**data)
        return report.model_dump()
    except Exception as e:
        return {"error": str(e), "raw": raw[:2000]}
