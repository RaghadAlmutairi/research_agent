"""
Query the knowledge base.
Usage: python query_kb.py "your question here"
       python query_kb.py   (uses default demo query)
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.rag.vector_store import VectorStore
from backend.rag.embedder import embed_text

vs = VectorStore()
total = vs.count()
print(f"Knowledge base: {total} chunks\n")

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what does this company do"
print(f'Query: "{query}"\n{"─" * 60}')

query_vec = embed_text(query)
results = vs.query(query_vec, n_results=5)

for i, r in enumerate(results, 1):
    url = r["metadata"].get("url", "unknown")
    score = r["score"]
    content = r["content"].strip()[:400]
    print(f"\n[{i}] score={score:.3f} | {url}")
    print(content)
    print()
