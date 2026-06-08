"""
Run competitor discovery from the command line.

Usage:
    python run_competitor_discovery.py https://beamdata.ai/ "BeamData"
"""
import sys, os, json, asyncio, logging

if sys.platform == "win32":
    import warnings; warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from backend.agents.research.competitor_discovery import discover_competitors


async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_competitor_discovery.py <url> [company_name]")
        sys.exit(1)
    url  = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"\n🔍 Discovering competitors for: {url}\n{'─'*60}")
    result = await discover_competitors(url, name, max_competitors=5)
    print(result.summary())
    with open("competitor_discovery_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print("✅ Saved to competitor_discovery_result.json")

if __name__ == "__main__":
    asyncio.run(main())
