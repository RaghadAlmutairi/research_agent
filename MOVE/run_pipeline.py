"""
Run the data acquisition pipeline (crawl → embed → ChromaDB).

Usage:
    python run_pipeline.py https://company.com
"""
import sys, os, asyncio, logging

if sys.platform == "win32":
    import warnings; warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from backend.config.settings import settings
from backend.agents.research.crawler.scheduler import Scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


async def main(urls):
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    scheduler = Scheduler(
        firecrawl_api_key=settings.firecrawl_api_key or None,
        collection_name=settings.chroma_collection,
        chroma_persist_dir=settings.chroma_persist_dir,
        requests_per_second=settings.crawl_requests_per_second,
        concurrency=settings.crawl_concurrency,
    )
    results = await scheduler.run(urls)
    print("\n=== Results ===")
    for r in results:
        icon = "✓" if r.status == "success" else "✗"
        print(f"  {icon} {r.url} | pages={r.pages} chunks={r.chunks_stored} status={r.status}")
        if r.error:
            print(f"      error: {r.error}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <url1> [url2 ...]")
        sys.exit(1)
    asyncio.run(main(sys.argv[1:]))
