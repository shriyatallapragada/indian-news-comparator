"""
engine.py — ML pipelines for the News Comparator API.

Provides:
  initialize_database()     — seeds ChromaDB from mock_database.csv
  get_perspective()         — semantic search filtered by bias label
  extract_summary()         — extractive summarisation via cosine similarity
  FastAPI app with POST /analyze_perspective
"""

import os
import sys
import uuid
import json
import re
import asyncio
import torch
import torch.nn.functional as F
import pandas as pd
import chromadb
import nltk
import spacy

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from groq import AsyncGroq
from news_fetch import build_search_terms, is_relevant
from nlp.analyzer import extract_search_query

import requests as http_requests
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

_thread_pool = ThreadPoolExecutor(max_workers=3)
_NEWSAPI_KEY  = os.environ.get("NEWSAPI_KEY", "")

# ── spaCy model (loaded once) ──────────────────────────────────────────────
_nlp = spacy.load("en_core_web_sm")

# ── Repo root on path so `model` package resolves ─────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model import IndicNewsEmbedder
from model.bias_classifier import FullBiasPipeline

# ── NLTK sentence tokenizer (download only if missing) ─────────────────────
def _ensure_nltk_resource(resource: str, download_name: str) -> None:
    try:
        nltk.data.find(resource)
    except (LookupError, OSError):
        try:
            nltk.download(download_name, quiet=True)
        except Exception as exc:
            print(f"[engine] NLTK resource {download_name} unavailable: {exc}")


_ensure_nltk_resource("tokenizers/punkt", "punkt")

# ── Paths ──────────────────────────────────────────────────────────────────
_DEFAULT_DB_PATH = "/tmp/chroma" if os.environ.get("SPACE_ID") else os.path.join(os.path.dirname(__file__), "chroma_db")
_DB_PATH  = os.environ.get("CHROMA_DB_PATH", _DEFAULT_DB_PATH)
_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mock_database.csv")
_COLLECTION = "news_articles"  # shared with vector_store.py

# ── Bias anchor texts (mirrors analyzer.py) ───────────────────────────────
_BIAS_ANCHOR_TEXTS = {
    "Left": (
        "The government must do more to protect workers, minorities, and the poor. "
        "Activists demand accountability as inequality rises and democratic forces push back against authoritarian overreach."
    ),
    "Center": (
        "Officials announced the new policy today. Experts say the decision will have mixed effects. "
        "Both sides of the debate have raised concerns about the long-term impact on the economy and governance."
    ),
    "Right": (
        "The ruling establishment's latest policy threatens individual freedom and economic growth. "
        "Critics argue that government overreach and excessive regulation are harming businesses and national security."
    ),
}

# ── Singletons ─────────────────────────────────────────────────────────────
_embedder: Optional[IndicNewsEmbedder] = None
_pipeline: Optional[FullBiasPipeline] = None
_collection = None
_anchor_vectors: dict = {}  # kept as fallback if pipeline weights missing
_MIN_PERSPECTIVE_SIMILARITY = 0.72
_TOPIC_ANCHORS = {
    "neet", "neet-ug", "nta", "ugc", "upsc", "exam", "examination",
    "paper leak", "paper leaks", "medical entrance",
}
_STOP_WORDS = {
    "after", "also", "article", "court", "digest", "from", "have", "large",
    "news", "paper", "party", "probe", "revisiting", "said", "says",
    "supreme", "that", "their", "this", "with", "would",
}


def _get_embedder() -> IndicNewsEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = IndicNewsEmbedder()
    return _embedder


def _get_pipeline() -> FullBiasPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FullBiasPipeline()
        _HEAD_PATH = os.path.join(os.path.dirname(__file__), "..", "train", "bias_classifier.pth")
        _pipeline.load(_HEAD_PATH)
        _pipeline.eval()
    return _pipeline


def _get_collection():
    global _collection
    if _collection is None:
        os.makedirs(_DB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=_DB_PATH)
        _collection = client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[engine] Collection '{_COLLECTION}' ready ({_collection.count()} docs).")
    return _collection


# ── Helper: mean-pool (1, seq, 768) → flat list ────────────────────────────
def _to_vector(text: str) -> list:
    embedder = _get_embedder()
    hidden = embedder.get_embeddings(text)          # (1, 512, 768)
    pooled = hidden.mean(dim=1).squeeze(0)          # (768,)
    return pooled.cpu().tolist()


def _to_tensor(text: str) -> torch.Tensor:
    """Returns a (1, 768) CPU tensor for cosine similarity calculations."""
    embedder = _get_embedder()
    hidden = embedder.get_embeddings(text)          # (1, 512, 768)
    return hidden.mean(dim=1).cpu()                 # (1, 768)


def _normalise_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower().strip("'\".,:;"))


def _extract_topic_terms(text: str, extra_terms: list = None) -> list:
    terms = []
    seen = set()

    def add(term: str):
        key = _normalise_term(term)
        if not key or key in seen or len(key) < 3:
            return
        seen.add(key)
        terms.append(key)

    for term in extra_terms or []:
        add(term)

    for acronym in re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)?\b", text or ""):
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


def _count_topic_hits(query_terms: list, candidate_text: str, stored_terms: str = "") -> tuple[int, bool]:
    candidate_lower = f"{candidate_text or ''} {stored_terms or ''}".lower()
    hits = 0
    anchor_hit = False

    for term in query_terms:
        term = _normalise_term(term)
        words = [w for w in re.findall(r"[a-z0-9-]+", term) if len(w) >= 3]
        direct_hit = term in candidate_lower
        word_hit = any(re.search(rf"\b{re.escape(word)}\b", candidate_lower) for word in words)
        if direct_hit or word_hit:
            hits += 1
            if term in _TOPIC_ANCHORS or any(anchor in term for anchor in _TOPIC_ANCHORS):
                anchor_hit = True

    return hits, anchor_hit


# ── Requirement 1: ChromaDB Semantic Search ────────────────────────────────

def initialize_database() -> int:
    """
    Reads mock_database.csv, embeds each row's text with IndicNewsEmbedder,
    and upserts into ChromaDB with source / bias_label / url metadata.
    Returns the number of documents ingested.
    """
    df = pd.read_csv(_CSV_PATH)
    collection = _get_collection()

    ids, embeddings, documents, metadatas = [], [], [], []

    for _, row in df.iterrows():
        text = str(row["text"])
        ids.append(str(row["article_id"]))
        embeddings.append(_to_vector(text))
        documents.append(text)
        metadatas.append({
            "source":     str(row["source"]),
            "bias":       str(row["bias_label"]),  # unified key
            "url":        str(row["url"]),
            "topic":      str(row.get("topic", "")),
        })
        print(f"  Embedded [{row['bias_label']}] {row['source']}")

    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=documents, metadatas=metadatas)
    print(f"[engine] Database initialised — {len(ids)} articles stored.")
    return len(ids)


def get_perspective(user_text_vector: list, target_bias: str,
                    user_text: str = "") -> Optional[dict]:
    """
    Queries ChromaDB with `user_text_vector` and returns the single closest
    article whose bias_label matches `target_bias` AND shares entity overlap
    with the user's article text.

    Returns a dict with keys: text, source, bias_label, url
    or None if no relevant match exists.
    """
    collection = _get_collection()
    if collection.count() == 0:
        print("[engine] Collection is empty — run initialize_database() first.")
        return None

    results = collection.query(
        query_embeddings=[user_text_vector],
        n_results=min(collection.count(), 20),
        where={"bias": target_bias},
        include=["metadatas", "documents", "distances"],
    )

    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return None

    query_terms = _extract_topic_terms(user_text)
    needs_anchor = any(
        term in _TOPIC_ANCHORS or any(anchor in term for anchor in _TOPIC_ANCHORS)
        for term in query_terms
    )

    candidates = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        stored_terms = " ".join([
            str(meta.get("topic", "")),
            str(meta.get("title", "")),
            str(meta.get("named_entities", "")),
        ])
        hits, anchor_hit = _count_topic_hits(query_terms, doc, stored_terms)
        similarity = 1 - float(dist)
        candidates.append((hits, anchor_hit, similarity, dist, doc, meta))

    candidates.sort(key=lambda item: (-item[0], -item[2], item[3]))

    for hits, anchor_hit, similarity, dist, doc, meta in candidates:
        if hits == 0:
            print(f"[engine] {target_bias} candidate rejected: no topic overlap")
            break
        if needs_anchor and not anchor_hit:
            print(f"[engine] {target_bias} candidate rejected: missing topic anchor")
            continue
        if similarity < _MIN_PERSPECTIVE_SIMILARITY and hits < 2:
            print(f"[engine] {target_bias} candidate rejected: weak similarity {similarity:.2f}")
            continue
        print(f"[engine] {target_bias} matched with {hits} topic hits, similarity {similarity:.2f}")
        return {
            "text":       doc,
            "source":     meta.get("source", ""),
            "bias_label": target_bias,
            "url":        meta.get("url", ""),
        }

    return None


# ── Requirement 2: Extractive Summarisation ───────────────────────────────

def extract_summary(article_text: str, embedder_instance: IndicNewsEmbedder,
                    top_n: int = 2) -> str:
    """
    Returns the top `top_n` sentences most representative of the article,
    selected by cosine similarity to the full-article embedding.

    Steps:
      1. Sentence-tokenise with nltk
      2. Embed the full article → main_vector (1, 768)
      3. Embed each sentence    → sentence_vectors
      4. Rank by cosine similarity to main_vector
      5. Return top_n sentences joined in their original order
    """
    sentences = nltk.tokenize.sent_tokenize(article_text)
    if not sentences:
        return article_text

    # Single-sentence edge case
    if len(sentences) == 1:
        return sentences[0]

    # Embed full article
    with torch.no_grad():
        main_hidden = embedder_instance.get_embeddings(article_text)
        main_vec    = main_hidden.mean(dim=1).cpu()             # (1, 768)

        # Embed each sentence and score it
        scores = []
        for sent in sentences:
            sent_hidden = embedder_instance.get_embeddings(sent)
            sent_vec    = sent_hidden.mean(dim=1).cpu()         # (1, 768)
            sim = F.cosine_similarity(main_vec, sent_vec).item()
            scores.append(sim)

    # Pick indices of top_n sentences, preserve original order
    top_indices = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    )
    return " ".join(sentences[i] for i in top_indices)


# ── Requirement 2 (new): Missing Entity Detection ─────────────────────────

_ENTITY_TYPES = {"ORG", "PERSON", "GPE", "EVENT"}


def get_missing_entities(user_text: str, retrieved_text: str) -> List[str]:
    """
    Returns entities present in `retrieved_text` but absent from `user_text`,
    filtered to ORG / PERSON / GPE / EVENT labels.
    Capped at 5 results, nicely capitalised.
    """
    user_doc      = _nlp(user_text)
    retrieved_doc = _nlp(retrieved_text)

    def extract(doc) -> set:
        return {
            ent.text.strip().lower()
            for ent in doc.ents
            if ent.label_ in _ENTITY_TYPES
            and len(ent.text.strip()) > 3  # skip fragments like "Tam", "EC"
        }

    user_entities      = extract(user_doc)
    retrieved_entities = extract(retrieved_doc)

    missing = retrieved_entities - user_entities

    # Restore nice capitalisation from the original doc
    cased = {
        ent.text.strip().lower(): ent.text.strip()
        for ent in retrieved_doc.ents
        if ent.label_ in _ENTITY_TYPES
    }

    result = [cased.get(e, e.title()) for e in missing]
    # Deduplicate (casing variants) and cap at 5
    seen: set = set()
    unique = []
    for item in result:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
        if len(unique) == 5:
            break

    return unique


# ── Bias score computation ────────────────────────────────────────────────

def _get_anchor_vectors() -> dict:
    """Lazily compute and cache anchor embeddings (mean-pooled tensors)."""
    global _anchor_vectors
    if _anchor_vectors:
        return _anchor_vectors
    embedder = _get_embedder()
    with torch.no_grad():
        for label, text in _BIAS_ANCHOR_TEXTS.items():
            hidden = embedder.get_embeddings(text)       # (1, seq, 768)
            _anchor_vectors[label] = hidden.mean(dim=1)  # (1, 768)
    return _anchor_vectors


def _compute_bias_score(text: str) -> float:
    """
    Returns a score on the -5..+5 axis.

    Primary path: IndicBERT → LiquidBrain → BiasClassifier
      score = (P_right - P_left) * 5

    Fallback (if classifier weights not trained yet): cosine similarity
    to Left/Right anchor texts — same as the original approach.
    """
    embedder = _get_embedder()
    with torch.no_grad():
        hidden = embedder.get_embeddings(text)   # (1, seq, 768)

    pipeline = _get_pipeline()

    # Primary: use the trained LNN + classifier head
    if pipeline is not None:
        try:
            score = pipeline.predict_score(hidden)
            print(f"[engine] LNN bias score: {score:+.2f}")
            return score
        except Exception as e:
            print(f"[engine] LNN scoring failed ({e}), falling back to anchors")

    # Fallback: anchor cosine similarity
    user_vec = hidden.mean(dim=1)
    anchors  = _get_anchor_vectors()
    sim_left  = F.cosine_similarity(user_vec, anchors["Left"]).item()
    sim_right = F.cosine_similarity(user_vec, anchors["Right"]).item()
    raw = (sim_right - sim_left)
    score = round(raw * 5, 2)
    print(f"[engine] Anchor fallback bias score: {score:+.2f}")
    return score


# ── LLM: Smart Insight Generation ────────────────────────────────────────

_GROQ_MODEL = "llama-3.1-8b-instant"


async def generate_smart_insights(
    original_text: str,
    alt_text: str,
    pytorch_score: float,
) -> dict:
    if not alt_text or alt_text.strip() == "":
        return {
            "reasoning": f"The ML model scored this article's framing at {pytorch_score:.2f}.",
            "missing_context": "No alternative perspective available.",
        }

    prompt = f"""You are a strict JSON data API. Analyze these two articles.
ML Score: {pytorch_score:.2f}
Original: {original_text[:1000]}
Alternative: {alt_text[:1000]}
Return ONLY a JSON object with exactly two keys: 'reasoning' (1 sentence explaining why the Original matches the ML Score of {pytorch_score:.2f} based on word choice) and 'missing_context' (1 sentence stating what fact/angle is in the Alternative but missing from the Original)."""

    print(f"DEBUG: Sending prompt to LLM with score: {pytorch_score:.2f}")

    llm_response_text = ""
    try:
        api_key = os.environ.get("GROQ_API_KEY", "")
        client = AsyncGroq(api_key=api_key)
        response = await client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        llm_response_text = response.choices[0].message.content.strip()

        # Bulletproof JSON extraction
        match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return json.loads(llm_response_text)

    except Exception as e:
        print(f"DEBUG LLM ERROR: {e}")
        return {
            "reasoning": f"Calculated ML Score: {pytorch_score:.2f}.",
            "missing_context": "Context analysis temporarily unavailable.",
        }


# ── Auto-seed: fetch & ingest live articles when ChromaDB has no match ────

def _auto_seed(query: str, named_entities: list) -> None:
    """
    Fetches articles from NewsAPI/Guardian for `query`, classifies their bias,
    and ingests them directly into ChromaDB — no HTTP round-trip needed.
    Called in a thread so it doesn't block the async endpoint.
    """
    from nlp.analyzer import analyze_rss_summary

    guardian_key = os.environ.get("GUARDIAN_API_KEY", "")
    articles = []

    # Try NewsAPI first
    if _NEWSAPI_KEY:
        try:
            r = http_requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "language": "en", "sortBy": "relevancy",
                        "pageSize": 15, "apiKey": _NEWSAPI_KEY},
                timeout=8,
            ).json()
            articles = r.get("articles", [])
            print(f"[auto-seed] NewsAPI({query!r}) → {len(articles)} articles")
        except Exception as e:
            print(f"[auto-seed] NewsAPI error: {e}")
    else:
        print("[auto-seed] NEWSAPI_KEY not set, skipping NewsAPI")

    # Guardian fallback
    if not articles and guardian_key:
        try:
            r = http_requests.get(
                "https://content.guardianapis.com/search",
                params={"q": query, "show-fields": "trailText,headline",
                        "order-by": "relevance", "page-size": 15,
                        "api-key": guardian_key},
                timeout=8,
            ).json()
            results = r.get("response", {}).get("results", [])
            articles = [
                {"title":       item.get("fields", {}).get("headline", ""),
                 "description": item.get("fields", {}).get("trailText", ""),
                 "url":         item.get("webUrl", ""),
                 "source":      {"name": "The Guardian"}}
                for item in results
            ]
            print(f"[auto-seed] Guardian({query!r}) → {len(articles)} articles")
        except Exception as e:
            print(f"[auto-seed] Guardian error: {e}")

    if not articles:
        print(f"[auto-seed] No articles found for {query!r}")
        return

    collection  = _get_collection()
    buckets     = {"Left": False, "Center": False, "Right": False}
    used_sources = set()
    search_terms = build_search_terms(query, named_entities, limit=6)

    for article in articles:
        text   = article.get("description") or article.get("title", "")
        url    = article.get("url", "")
        source_raw = article.get("source", {})
        source = source_raw.get("name", "Unknown") if isinstance(source_raw, dict) else str(source_raw)

        if not text or not url or "consent.yahoo" in url or source in used_sources:
            continue
        if search_terms and not is_relevant(article, search_terms, named_entities):
            print(f"[auto-seed] Skipping unrelated article: {article.get('title', '')[:70]}")
            continue

        # Skip sports/entertainment noise
        combined = (article.get("title", "") + " " + text)
        if any(kw in combined.lower() for kw in [
            "ipl", "stadium", "cricket", "score", "match", "batting",
            "bollywood", "wedding", "recipe", "fitness", "marathon"
        ]):
            continue

        tagged = analyze_rss_summary(text)
        bias   = tagged.get("bias", "Center")

        if buckets.get(bias):
            continue  # already have one for this bias

        article_id = url  # use URL as stable ID to avoid duplicates
        vector     = _to_vector(text)

        collection.upsert(
            ids=[article_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[{
                "source": source,
                "bias":   bias,
                "url":    url,
                "topic":  query,
                "title":  article.get("title", ""),
                "named_entities": ",".join(tagged.get("named_entities", [])),
                "core_event_slug": tagged.get("core_event_slug", ""),
            }],
        )
        buckets[bias] = True
        used_sources.add(source)
        print(f"[auto-seed] Ingested [{bias}] {source}")

        if all(buckets.values()):
            break


# ── Requirement 3: FastAPI endpoint ───────────────────────────────────────

app = FastAPI(title="News Comparator Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PerspectiveRequest(BaseModel):
    user_text:   str
    target_lean: str = "Center"   # "Left" | "Center" | "Right"


class ArticlePayload(BaseModel):
    topic:      str
    source:     str
    bias_label: str
    url:        str
    text:       str


class PerspectiveResponse(BaseModel):
    perspective_summary: Optional[str]
    source_name:         Optional[str]
    url:                 Optional[str]
    missing_entities:    List[str] = []
    bias_score:          Optional[float] = None
    reasoning:           Optional[str] = None
    missing_context:     Optional[str] = None


@app.post("/analyze_perspective", response_model=PerspectiveResponse)
async def analyze_perspective(req: PerspectiveRequest):
    embedder = _get_embedder()

    # 1. Embed the user's article
    user_vector = _to_vector(req.user_text)

    # 2. Find the closest article with the requested bias
    article = get_perspective(user_vector, req.target_lean, req.user_text)

    # 3. If no match, auto-seed from live news and retry once
    if article is None:
        print(f"[engine] No {req.target_lean} match — auto-seeding…")
        query = extract_search_query(req.user_text[:3000])
        if not query:
            query_terms = build_search_terms(req.user_text[:1000], [], limit=5)
            query = " OR ".join(f'"{term}"' for term in query_terms[:3]) if query_terms else req.user_text[:80].strip()

        print(f"[engine] Auto-seed query: {query!r}")
        loop  = asyncio.get_event_loop()
        seed_terms = build_search_terms(req.user_text[:1000], [], limit=6)
        await loop.run_in_executor(_thread_pool, _auto_seed, query, seed_terms)
        article = get_perspective(user_vector, req.target_lean, req.user_text)

    if article is None:
        # Still nothing — return bias score only, no perspective
        pytorch_score = _compute_bias_score(req.user_text)
        return PerspectiveResponse(
            perspective_summary=None,
            source_name=None,
            url=None,
            missing_entities=[],
            bias_score=round(pytorch_score, 2),
            reasoning=None,
            missing_context=None,
        )

    # 4. Summarise the retrieved article
    summary = extract_summary(article["text"], embedder)

    # 5. Find entities in the alternative article missing from the user's article
    missing = get_missing_entities(req.user_text, article["text"])

    # 6. Compute the real PyTorch bias score via the LNN pipeline
    pytorch_score = _compute_bias_score(req.user_text)

    # 7. Generate LLM insights
    insights = await generate_smart_insights(
        original_text=req.user_text,
        alt_text=article["text"],
        pytorch_score=pytorch_score,
    )

    return PerspectiveResponse(
        perspective_summary=summary,
        source_name=article["source"],
        url=article["url"],
        missing_entities=missing,
        bias_score=round(pytorch_score, 2),
        reasoning=insights["reasoning"],
        missing_context=insights["missing_context"],
    )


@app.post("/api/ingest_live_batch")
async def ingest_live_batch(articles: List[ArticlePayload]):
    """
    Accepts a JSON batch of live articles and ingests them into ChromaDB.
    Each article is embedded via IndicNewsEmbedder and stored with full metadata.
    """
    if not articles:
        return {"status": "ok", "ingested": 0, "message": "No articles provided."}

    collection = _get_collection()

    ids, embeddings, documents, metadatas = [], [], [], []

    for article in articles:
        article_id = str(uuid.uuid4())
        vector = _to_vector(article.text)

        ids.append(article_id)
        embeddings.append(vector)
        documents.append(article.text)
        metadatas.append({
            "source":     article.source,
            "bias":       article.bias_label,
            "bias_label": article.bias_label,
            "url":        article.url,
            "topic":      article.topic,
        })
        print(f"  [ingest] [{article.bias_label}] {article.source} — {article.url[:60]}")

    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=documents, metadatas=metadatas)

    return {
        "status": "ok",
        "ingested": len(ids),
        "message": f"{len(ids)} article(s) successfully ingested into ChromaDB.",
    }


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("[engine] Starting server on :8001")
    uvicorn.run("engine:app", host="127.0.0.1", port=8001, reload=False)
@app.post("/api/ingest")
def ingest_data():
    try:
        # Assuming Kiro named the setup function 'initialize_database'
        initialize_database() 
        return {"status": "success", "message": "Database initialized and vectors loaded!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
