"""
gap_detector_tool      — detect market gaps and underserved segments
opportunity_scorer_tool — score and rank strategic opportunities
"""
import json
import os
import re
from typing import Any
from langchain_core.tools import tool
from openai import OpenAI


def _ask(system: str, user: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> Any:
    return json.loads(re.sub(r"```json|```", "", raw).strip())


@tool
def gap_detector_tool(
    company_context: dict,
    competitors: list[dict],
    reviews: dict,
    trends: dict,
) -> dict[str, Any]:
    """Detect market gaps, underserved segments, and unmet customer needs."""
    prompt = f"""You are a strategic market analyst. Detect market gaps and opportunities.

OUR COMPANY:
{json.dumps(company_context, indent=2)[:1500]}

COMPETITORS:
{json.dumps([c.get("name") for c in competitors], indent=2)}

CUSTOMER PAIN POINTS:
{json.dumps(reviews.get("pain_points", reviews.get("negative_themes", [])), indent=2)[:1000]}

TRENDS:
{json.dumps(trends.get("trends", [])[:5], indent=2)[:1000]}

Return ONLY valid JSON:
{{
  "market_gaps": [
    {{
      "gap": "No affordable solution for SMBs under $50/mo",
      "evidence": "pain points + competitor pricing",
      "size_estimate": "large/medium/small",
      "difficulty_to_fill": "low/medium/high"
    }}
  ],
  "underserved_segments": [
    {{
      "segment": "...",
      "why_underserved": "...",
      "entry_strategy": "..."
    }}
  ],
  "unmet_needs": ["need1", "need2"],
  "whitespace_opportunities": ["opportunity1"]
}}"""

    raw = _ask("Strategic market gap analyst. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"raw": raw}


@tool
def opportunity_scorer_tool(
    gaps: dict,
    trends: dict,
    company_context: dict,
) -> dict[str, Any]:
    """Score and rank strategic opportunities by impact, feasibility, and urgency."""
    prompt = f"""Score and rank strategic opportunities for this company.

COMPANY:
{json.dumps(company_context, indent=2)[:1000]}

GAPS:
{json.dumps(gaps, indent=2)[:1500]}

TRENDS:
{json.dumps(trends.get("top_priority_trends", trends.get("trends", []))[:5], indent=2)[:1000]}

Return ONLY valid JSON:
{{
  "scored_opportunities": [
    {{
      "opportunity": "...",
      "impact_score": 8,
      "feasibility_score": 7,
      "urgency_score": 9,
      "composite_score": 8.0,
      "rationale": "...",
      "recommended_action": "...",
      "time_to_act": "now/3mo/6mo/1yr"
    }}
  ],
  "top_opportunity": "...",
  "quick_wins": ["opportunity with low effort, high impact"],
  "strategic_bets": ["high risk, high reward opportunity"]
}}"""

    raw = _ask("Strategic opportunity scorer. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"raw": raw}
