"""
review_collector_tool      — collect simulated/LLM-generated review insights
sentiment_analysis_tool    — analyze sentiment from review text
pain_point_extractor_tool  — extract customer pain points
feature_request_extractor_tool — extract requested features
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
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


@tool
def review_collector_tool(company_name: str, industry: str) -> dict[str, Any]:
    """
    Collect customer review intelligence for a company.
    Uses AI to synthesize common review patterns from G2, Capterra, Trustpilot, and Reddit.
    """
    prompt = f"""You are a customer research analyst with access to review platforms.

Based on your knowledge of "{company_name}" in the "{industry}" industry, synthesize what customers typically say on platforms like G2, Capterra, Trustpilot, and Reddit.

Return ONLY valid JSON:
{{
  "overall_sentiment": "positive/mixed/negative",
  "average_rating": 4.2,
  "total_reviews_analyzed": 150,
  "platforms": ["G2", "Capterra"],
  "positive_themes": ["theme1", "theme2"],
  "negative_themes": ["theme1", "theme2"],
  "sample_quotes": [
    {{"sentiment": "positive", "text": "quote here", "source": "G2"}},
    {{"sentiment": "negative", "text": "quote here", "source": "Capterra"}}
  ]
}}"""

    raw = _ask("Customer review analyst. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"company": company_name, "raw": raw}


@tool
def sentiment_analysis_tool(reviews_data: dict) -> dict[str, Any]:
    """Analyze sentiment breakdown from collected review data."""
    prompt = f"""Analyze the sentiment from this review data and provide a detailed breakdown.

Review data:
{json.dumps(reviews_data, indent=2)[:3000]}

Return ONLY valid JSON:
{{
  "sentiment_score": 0.72,
  "positive_pct": 65,
  "neutral_pct": 20,
  "negative_pct": 15,
  "emotional_drivers": {{
    "positive": ["ease of use", "customer support"],
    "negative": ["pricing", "missing features"]
  }},
  "nps_estimate": 42,
  "loyalty_signals": ["customers recommend it", "long-term users"]
}}"""

    raw = _ask("Sentiment analysis expert. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"raw": raw}


@tool
def pain_point_extractor_tool(company_name: str, reviews_data: dict) -> dict[str, Any]:
    """Extract and rank customer pain points from review data."""
    prompt = f"""Extract customer pain points for "{company_name}" from this review intelligence.

Data:
{json.dumps(reviews_data, indent=2)[:3000]}

Return ONLY valid JSON:
{{
  "pain_points": [
    {{
      "issue": "pricing too high",
      "severity": "high",
      "frequency": "very common",
      "customer_segment": "SMBs",
      "verbatim": "example complaint phrasing"
    }}
  ],
  "top_3_critical": ["issue1", "issue2", "issue3"]
}}"""

    raw = _ask("Customer pain point analyst. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"raw": raw}


@tool
def feature_request_extractor_tool(company_name: str, reviews_data: dict) -> dict[str, Any]:
    """Extract most-requested features and product improvements from customer feedback."""
    prompt = f"""Extract feature requests and product improvement wishes for "{company_name}".

Data:
{json.dumps(reviews_data, indent=2)[:3000]}

Return ONLY valid JSON:
{{
  "feature_requests": [
    {{
      "feature": "better API",
      "demand": "high",
      "frequency": "mentioned in 30%+ reviews",
      "use_case": "developers want to integrate"
    }}
  ],
  "quick_wins": ["small improvement with high impact"],
  "strategic_features": ["large feature with competitive advantage"]
}}"""

    raw = _ask("Product analyst extracting feature requests. Return valid JSON only.", prompt)
    try:
        return _parse_json(raw)
    except Exception:
        return {"raw": raw}
