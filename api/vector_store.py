"""
vector_store.py — ChromaDB-backed store using IndicBERT embeddings.

ingest_article()         → embed + store an article
find_related_by_entities() → retrieve best Left/Center/Right match
"""

import sys
import os
import re
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

_DEFAULT_DB_PATH = "/tmp/chroma" if os.environ.get("SPACE_ID") else os.path.join(os.path.dirname(__file__), "chroma_db")
_DB_PATH = os.environ.get("CHROMA_DB_PATH", _DEFAULT_DB_PATH)
_COLLECTION_NAME = "news_articles"
_TIME_WINDOW_HOURS = 48
_MIN_SIMILARITY = 0.72  # below this score, treat as no match
_TOPIC_ANCHORS = {
    "neet", "neet-ug", "nta", "ugc", "upsc", "exam", "examination",
    "paper leak", "paper leaks", "medical entrance",
    "mekedatu", "mekedatu reservoir", "reservoir proposal",
}
_STOP_WORDS = {
    "after", "also", "article", "court", "digest", "from", "have", "large",
    "leak", "news", "paper", "party", "probe", "revisiting", "said", "says",
    "supreme", "that", "their", "this", "with", "would",
    "new", "delhi", "key", "takeaways", "visit", "visits",
}


def _get_embedder() -> IndicNewsEmbedder:
    global _embedder
    if _embedder is None:
        print("[vector_store] Loading embedder…")
        _embedder = IndicNewsEmbedder()
    return _embedder


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(_DB_PATH, exist_ok=True)
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


def _normalise_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower().strip("'\".,:;"))


def _extract_topic_terms(text: str, named_entities: list = None) -> list:
    """Extract terms that must overlap before an article is considered related."""
    terms = []
    seen = set()

    def add(term: str):
        key = _normalise_term(term)
        if not key or key in seen:
            return
        if len(key) < 3:
            return
        seen.add(key)
        terms.append(key)

    for entity in named_entities or []:
        add(entity)

    for acronym in re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)?\b", text or ""):
        if acronym not in {"NEW", "DELHI"}:
            add(acronym)

    lowered = (text or "").lower()
    for phrase in _TOPIC_ANCHORS | {"supreme court", "national testing agency"}:
        if phrase in lowered:
            add(phrase)

    for word in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text or ""):
        key = _normalise_term(word)
        if key not in _STOP_WORDS:
            add(key)
        if len(terms) >= 10:
            break

    return terms[:10]


def _count_topic_hits(query_terms: list, candidate_text: str, stored_terms: set) -> tuple[int, bool]:
    candidate_text = candidate_text or ""
    candidate_lower = candidate_text.lower()
    hits = 0
    anchor_hit = False

    def word_present(word: str) -> bool:
        return bool(re.search(rf"\b{re.escape(word)}\b", candidate_text, re.IGNORECASE))

    def phrase_present(phrase: str) -> bool:
        return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", candidate_text, re.IGNORECASE))

    def term_matches(term: str) -> bool:
        stored_hit = term in stored_terms
        if stored_hit:
            return True
        if " " in term:
            if phrase_present(term):
                return True
            words = [
                word for word in re.findall(r"[a-z0-9-]+", term)
                if len(word) >= 3 and word not in _STOP_WORDS
            ]
            return len(words) >= 2 and all(word_present(word) for word in words)
        if re.fullmatch(r"[a-z0-9-]{2,}", term):
            return term not in _STOP_WORDS and word_present(term)
        return term in candidate_lower

    for term in query_terms:
        term = _normalise_term(term)
        if term_matches(term):
            hits += 1
            if term in _TOPIC_ANCHORS or any(anchor in term for anchor in _TOPIC_ANCHORS):
                anchor_hit = True

    return hits, anchor_hit


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
    exclude_url: str = "",
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

    query_terms = _extract_topic_terms(summary, named_entities)
    needs_anchor = any(
        term in _TOPIC_ANCHORS or any(anchor in term for anchor in _TOPIC_ANCHORS)
        for term in query_terms
    )

    buckets: dict = {"Left": [], "Center": [], "Right": []}

    for meta, dist, doc in zip(metadatas, distances, documents):
        bias = meta.get("bias", "")
        if bias not in buckets:
            continue
        if exclude_url and meta.get("url", "").strip() == exclude_url.strip():
            print(f"[vector_store] Skipping opened article URL: {exclude_url[:70]}")
            continue

        stored = set(
            e.strip().lower()
            for e in meta.get("named_entities", "").split(",") if e.strip()
        )
        total_hits, anchor_hit = _count_topic_hits(query_terms, doc, stored)
        similarity = 1 - float(dist)
        buckets[bias].append((total_hits, anchor_hit, similarity, dist, meta, doc))

    output: dict = {"left": None, "center": None, "right": None}

    for key, bias_key in [("left", "Left"), ("center", "Center"), ("right", "Right")]:
        # Sort by entity hits first (desc), then cosine distance (asc)
        candidates = sorted(buckets[bias_key], key=lambda x: (-x[0], -x[2], x[3]))
        for hits, anchor_hit, similarity, dist, meta, doc in candidates:
            if hits == 0:
                # No entity overlap at all — skip, let auto-fetch handle it
                print(f"[vector_store] {bias_key} — no entity overlap, skipping")
                break
            if not needs_anchor and hits < 2:
                print(f"[vector_store] {bias_key} — weak topic overlap ({hits}), skipping")
                continue
            if needs_anchor and not anchor_hit:
                print(f"[vector_store] {bias_key} — missing topic anchor, skipping")
                continue
            if similarity < _MIN_SIMILARITY and hits < 2:
                print(f"[vector_store] {bias_key} — weak similarity {similarity:.2f}, skipping")
                continue
            output[key] = {
                "title":  meta.get("title", ""),
                "url":    meta.get("url", ""),
                "source": meta.get("source", "") or meta.get("url", ""),
                "summary": doc[:200],
                "bias":   bias_key,
            }
            print(f"[vector_store] {bias_key} matched with {hits} topic hits, similarity {similarity:.2f}")
            break

    return output
