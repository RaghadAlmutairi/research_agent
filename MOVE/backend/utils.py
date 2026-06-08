"""
Shared async utility — safe to call from anywhere:
  - Plain Python scripts
  - LangGraph nodes
  - Jupyter notebooks (after nest_asyncio.apply())
"""
import asyncio


def run_async(coro):
    """
    Run a coroutine from synchronous code regardless of context.

    Strategy:
      - If nest_asyncio has been applied (Jupyter), a running loop exists
        and has been patched to allow re-entry → use loop.run_until_complete()
      - If no loop is running (plain script) → use asyncio.run()
    """
    try:
        loop = asyncio.get_running_loop()
        # Running loop exists (Jupyter + nest_asyncio) — run directly on it
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop — plain script
        return asyncio.run(coro)
