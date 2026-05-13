"""
vector_store.py — ChromaDB-backed store using IndicBERT embeddings.

ingest_article()         → embed + store an article
find_related_by_entities() → retrieve best Left/Center/Right match
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import chromadb

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model import IndicNewsEmbedder

# ── Singletons ─────────────────────────────────────────────────────────────
_embedder: Optional[IndicNewsEmbedder] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None

_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
_COLLECTION_NAME = "news_articles"
_TIME_WINDOW_HOURS = 48
_MIN_SIMILARITY = 0.72  # below this score, treat as no match


def _get_embedder() -> IndicNewsEmbedder:
    global _embedder
    if _embedder is None:
        print("[vector_store] Loading embedder…")
        _embedder = IndicNewsEmbedder()
    return _embedder


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=_DB_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[vector_store] Collection '{_COLLECTION_NAME}' ready "
              f"({_collection.count()} docs).")
    return _collection


def _embed(text: str) -> list:
    """Returns a flat float list suitable for ChromaDB."""
    embedder = _get_embedder()
    # Shape: (1, seq_len, 768) → mean-pool over seq → (768,)
    tensor = embedder.get_embeddings(text)
    pooled = tensor.mean(dim=1).squeeze(0)  # (768,)
    return pooled.tolist()


# ── Public API ─────────────────────────────────────────────────────────────

def ingest_article(
    summary: str,
    bias: str,
    named_entities: list,
    core_event_slug: str,
    title: str = "",
    url: str = "",
    source: str = "",
    published_at: str = "",
) -> None:
    """Embed the article summary and store it in ChromaDB."""
    collection = _get_collection()
    doc_id = url if url else str(hash(summary))
    embedding = _embed(summary)
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[{
            "bias": bias,
            "title": title,
            "url": url,
            "source": source,
            "published_at": published_at,
            "named_entities": ",".join(named_entities),
            "core_event_slug": core_event_slug,
        }],
    )
    print(f"[vector_store] Ingested: {title or url or doc_id[:40]} [{bias}]")


def find_related_by_entities(
    summary: str,
    named_entities: list,
    published_at: str = "",
) -> dict:
    collection = _get_collection()

    if collection.count() == 0:
        return {"left": None, "center": None, "right": None}

    query_embedding = _embed(summary)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(20, collection.count()),
        include=["metadatas", "distances", "documents"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]

    # Normalise query entities for text matching
    query_entities = [e.lower().strip() for e in named_entities if len(e.strip()) > 4]

    buckets: dict = {"Left": [], "Center": [], "Right": []}

    for meta, dist, doc in zip(metadatas, distances, documents):
        bias = meta.get("bias", "")
        if bias not in buckets:
            continue

        doc_lower = doc.lower()

        # Count how many query entities appear in the document text
        text_hits = sum(1 for e in query_entities if e in doc_lower)

        # Also check stored named_entities metadata
        stored = set(
            e.strip().lower()
            for e in meta.get("named_entities", "").split(",") if e.strip()
        )
        meta_hits = len(set(query_entities) & stored)

        total_hits = text_hits + meta_hits
        buckets[bias].append((total_hits, dist, meta, doc))

    output: dict = {"left": None, "center": None, "right": None}

    for key, bias_key in [("left", "Left"), ("center", "Center"), ("right", "Right")]:
        # Sort by entity hits first (desc), then cosine distance (asc)
        candidates = sorted(buckets[bias_key], key=lambda x: (-x[0], x[1]))
        for hits, dist, meta, doc in candidates:
            if hits == 0:
                # No entity overlap at all — skip, let auto-fetch handle it
                print(f"[vector_store] {bias_key} — no entity overlap, skipping")
                break
            output[key] = {
                "title":  meta.get("title", ""),
                "url":    meta.get("url", ""),
                "source": meta.get("source", "") or meta.get("url", ""),
                "summary": doc[:200],
                "bias":   bias_key,
            }
            print(f"[vector_store] {bias_key} matched with {hits} entity hits")
            break

    return output

