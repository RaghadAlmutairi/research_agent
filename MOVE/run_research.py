"""
Run the Research Agent from the command line.

Usage:
    python run_research.py https://beamdata.ai/ "BeamData"
"""
import sys, os, json, logging

if sys.platform == "win32":
    import warnings; warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from backend.agents.research.research_agent import run_research


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_research.py <url> [company_name]")
        sys.exit(1)

    url  = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"\n🔍 Research Agent: {url}\n{'─'*60}")
    result = run_research(url, name)

    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
        sys.exit(1)

    report   = result.get("report", {})
    insights = result.get("insights", {})
    summary  = report.get("company_summary", {})

    print(f"\n{'═'*60}\n📊 RESEARCH REPORT\n{'═'*60}")
    print(f"\n🏢  {summary.get('name', url)} — {summary.get('industry', '')}")
    print(f"    {summary.get('description', '')}")

    print(f"\n🏆  Competitors ({len(result.get('competitors', []))}):")
    for c in result.get("competitors", [])[:5]:
        print(f"    • {c.get('name')} — {c.get('website', '')}")

    print(f"\n😣  Pain Points:")
    for p in report.get("customer_pain_points", [])[:4]:
        print(f"    • {p}")

    print(f"\n📈  Top Trends:")
    for t in report.get("trends", [])[:3]:
        print(f"    • {t.get('trend', t) if isinstance(t, dict) else t}")

    print(f"\n🚀  Opportunities:")
    for o in report.get("opportunities", [])[:3]:
        print(f"    • {o.get('opportunity', o) if isinstance(o, dict) else o}")

    print(f"\n💡  Executive Summary:")
    print(f"    {insights.get('executive_summary', 'N/A')}")

    print(f"\n⚡  Immediate Actions:")
    for a in insights.get("immediate_actions", [])[:3]:
        print(f"    → {a}")

    with open("research_report.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Full report saved to research_report.json\n")

if __name__ == "__main__":
    main()
