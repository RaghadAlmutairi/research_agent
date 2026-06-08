"""
Lightweight entity extraction from cleaned markdown content.
Uses regex heuristics by default; can be upgraded to an LLM/NER call.
"""
import re
from typing import Any


# Simple patterns – extend or swap for spaCy / OpenAI structured outputs
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"\+?[\d\s\-().]{7,20}")
_URL_RE = re.compile(r"https?://[^\s)>\"']+")
_PRICE_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")


def extract_entities(text: str) -> dict[str, Any]:
    """Extract structured entities from plain/markdown text."""
    return {
        "emails": list(set(_EMAIL_RE.findall(text))),
        "phones": list(set(_PHONE_RE.findall(text)))[:10],  # cap noise
        "urls": list(set(_URL_RE.findall(text))),
        "prices": list(set(_PRICE_RE.findall(text))),
    }


def enrich_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add extracted entities to each chunk's metadata."""
    for chunk in chunks:
        entities = extract_entities(chunk.get("content", ""))
        chunk.setdefault("metadata", {})["entities"] = entities
    return chunks
