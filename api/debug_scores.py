"""
debug_scores.py — shows the actual cosine similarity scores for a query
against everything in ChromaDB.

Usage:
    python debug_scores.py "your article summary here"
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import _embed, _get_collection

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
    "200 units of free power women safety force anti-narcotics Tamil Nadu CM Vijay priorities"

print(f"Query: {query[:100]}\n")

collection = _get_collection()
if collection.count() == 0:
    print("Collection is empty.")
    sys.exit()

embedding = _embed(query)
results = collection.query(
    query_embeddings=[embedding],
    n_results=collection.count(),
    include=["metadatas", "distances", "documents"],
)

print(f"{'Score':>6}  {'Bias':>8}  {'Source':>15}  Text")
print("-" * 80)
for meta, dist, doc in zip(
    results["metadatas"][0],
    results["distances"][0],
    results["documents"][0],
):
    score = 1.0 - (dist / 2.0)
    bias   = meta.get("bias", "?")
    source = meta.get("source", meta.get("url", ""))[:15]
    print(f"{score:>6.3f}  {bias:>8}  {source:>15}  {doc[:60]}")
