"""
CompetitorDiscoveryAgent — auto-discover and profile competitors.
"""
import asyncio, json, logging, os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from .web_searcher import find_competitor_urls
from .url_validator import validate_urls, normalize_url
from .profile_builder import build_profile
from backend.utils import run_async

logger = logging.getLogger(__name__)


@dataclass
class CompetitorProfile:
    name: str
    website: str
    description: str = ""
    industry: str = ""
    target_market: list = field(default_factory=list)
    key_features: list = field(default_factory=list)
    pricing_model: str = "unknown"
    pricing_tiers: list = field(default_factory=list)
    positioning: str = ""
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    technologies: list = field(default_factory=list)
    integrations: list = field(default_factory=list)
    discovery_source: str = ""
    pages_scraped: int = 0
    error: str = ""


@dataclass
class CompetitorDiscoveryResult:
    company_url: str
    company_name: str
    company_context: dict
    competitors: list
    discovery_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    total_found: int = 0
    total_profiled: int = 0

    def to_dict(self):
        return {
            "company_url": self.company_url,
            "company_name": self.company_name,
            "company_context": self.company_context,
            "competitors": [c.__dict__ for c in self.competitors],
            "discovery_timestamp": self.discovery_timestamp,
            "total_found": self.total_found,
            "total_profiled": self.total_profiled,
        }

    def summary(self):
        lines = [f"Company: {self.company_name} ({self.company_url})",
                 f"Found: {self.total_found} → Profiled: {self.total_profiled}\n"]
        for c in self.competitors:
            lines.append(f"  • {c.name} — {c.website}")
            if c.positioning: lines.append(f"    Positioning: {c.positioning}")
            if c.pricing_model != "unknown": lines.append(f"    Pricing: {c.pricing_model}")
            lines.append("")
        return "\n".join(lines)


class CompetitorDiscoveryAgent:

    def __init__(self, max_competitors: int = 5, profile_concurrency: int = 3):
        self.max_competitors = max_competitors
        self.profile_concurrency = profile_concurrency

    async def _scrape_target(self, url: str) -> dict[str, Any]:
        from backend.agents.research.crawler.crawl_manager import CrawlManager
        from backend.agents.research.crawler.cleaner import clean_content, is_meaningful, is_noise_url
        from backend.agents.research.tools.website_scraper import company_context_extractor_tool
        from backend.config.settings import settings

        try:
            mgr = CrawlManager(firecrawl_api_key=settings.firecrawl_api_key or None)
            pages = await mgr.crawl(url, limit=20)
        except Exception as e:
            logger.warning(f"[Discovery] Scrape failed: {e}")
            return {"content": "", "context": {}}

        parts = []
        for page in pages:
            if is_noise_url(page.get("url","")): continue
            c = clean_content(page.get("content",""))
            if is_meaningful(c): parts.append(c[:1500])

        combined = "\n\n".join(parts[:8])
        domain = urlparse(url).netloc.replace("www.", "")
        context = company_context_extractor_tool.invoke({"company_name": domain, "scraped_content": combined})
        return {"content": combined, "context": context}

    async def _build_profiles_concurrent(self, candidates: list) -> list:
        sem = asyncio.Semaphore(self.profile_concurrency)

        async def _one(candidate):
            async with sem:
                url = candidate.get("website", "")
                name = candidate.get("name", url)
                raw = await build_profile(url, name)
                return CompetitorProfile(
                    name=raw.get("name", name), website=url,
                    description=raw.get("description",""), industry=raw.get("industry",""),
                    target_market=raw.get("target_market",[]), key_features=raw.get("key_features",[]),
                    pricing_model=raw.get("pricing_model","unknown"), pricing_tiers=raw.get("pricing_tiers",[]),
                    positioning=raw.get("positioning",""), strengths=raw.get("strengths",[]),
                    weaknesses=raw.get("weaknesses",[]), technologies=raw.get("technologies",[]),
                    integrations=raw.get("integrations",[]),
                    discovery_source=candidate.get("source","unknown"),
                    pages_scraped=raw.get("pages_scraped",0), error=raw.get("error",""),
                )

        return list(await asyncio.gather(*[_one(c) for c in candidates]))

    def _store(self, result: CompetitorDiscoveryResult):
        try:
            from backend.rag.vector_store import VectorStore
            from backend.rag.embedder import embed_text
            vs = VectorStore(collection_name="research")
            for p in result.competitors:
                content = json.dumps(p.__dict__, indent=2)
                vs.upsert_chunks([{
                    "content": content, "embedding": embed_text(content[:2000]),
                    "metadata": {"research_type":"competitor_profile","company":result.company_name,
                                 "competitor":p.name,"website":p.website,"chunk_index":"0"},
                    "chunk_index": 0,
                }])
        except Exception as e:
            logger.warning(f"[Discovery] Store failed: {e}")

    async def run(self, company_url: str, company_name: str = "") -> CompetitorDiscoveryResult:
        company_url = normalize_url(company_url)
        base_domain = urlparse(company_url).netloc
        logger.info(f"[Discovery] Starting for: {company_url}")

        target = await self._scrape_target(company_url)
        context = target.get("context", {})
        content = target.get("content", "")
        name = company_name or context.get("company_name") or base_domain
        industry = context.get("industry", "Technology")
        description = context.get("description", "")

        raw_candidates = await find_competitor_urls(
            company_name=name, industry=industry, description=description,
            scraped_content=content, base_domain=base_domain,
        )
        logger.info(f"[Discovery] {len(raw_candidates)} raw candidates")

        urls = [c.get("website","") for c in raw_candidates if c.get("website")]
        valid_urls = await validate_urls(urls)
        valid_set = {u.rstrip("/") for u in valid_urls}
        valid_cands = [c for c in raw_candidates if c.get("website","").rstrip("/") in valid_set][:self.max_competitors]

        profiles = await self._build_profiles_concurrent(valid_cands)
        good = [p for p in profiles if not p.error]

        result = CompetitorDiscoveryResult(
            company_url=company_url, company_name=name, company_context=context,
            competitors=good, total_found=len(raw_candidates), total_profiled=len(good),
        )
        self._store(result)
        return result


async def discover_competitors(company_url: str, company_name: str = "", max_competitors: int = 5) -> CompetitorDiscoveryResult:
    return await CompetitorDiscoveryAgent(max_competitors=max_competitors).run(company_url, company_name)
