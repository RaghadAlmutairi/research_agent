"""
insight_extractor_tool — extract key insights from the full research report
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
def insight_extractor_tool(report: dict) -> dict[str, Any]:
    """Extract the most critical actionable insights from a research report."""
    prompt = f"""From this research report, extract the most critical actionable insights.

Report:
{json.dumps(report, indent=2)[:4000]}

Return ONLY valid JSON:
{{
  "executive_summary": "3-4 sentence summary of the most important findings",
  "top_insights": [
    {{
      "insight": "...",
      "why_it_matters": "...",
      "action": "..."
    }}
  ],
  "immediate_actions": ["action to take this week"],
  "competitive_moat": "what makes this company defensible",
  "biggest_risk": "single biggest threat",
  "best_opportunity": "single best opportunity right now"
}}"""

    raw = _ask("Strategic insight extractor. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"raw": raw}
