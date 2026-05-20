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

load_dotenv()

# ── spaCy model (loaded once) ──────────────────────────────────────────────
_nlp = spacy.load("en_core_web_sm")

# ── Repo root on path so `model` package resolves ─────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model import IndicNewsEmbedder
from model.bias_classifier import FullBiasPipeline

# ── NLTK sentence tokenizer (downloaded once) ──────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────
_DB_PATH  = os.path.join(os.path.dirname(__file__), "chroma_db")
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

    if not docs:
        return None

    user_text_lower = user_text.lower()

    # Pick the first candidate whose text shares at least one meaningful word
    # with the user's article (words > 4 chars to skip stopwords)
    user_words = set(
        w for w in user_text_lower.split()
        if len(w) > 4 and w.isalpha()
    )

    for doc, meta in zip(docs, metadatas):
        doc_lower = doc.lower()
        overlap = sum(1 for w in user_words if w in doc_lower)
        if overlap >= 2:
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

    if article is None:
        return PerspectiveResponse(
            perspective_summary=None,
            source_name=None,
            url=None,
            missing_entities=[],
            reasoning=None,
            missing_context=None,
        )

    # 3. Summarise the retrieved article
    summary = extract_summary(article["text"], embedder)

    # 4. Find entities in the alternative article missing from the user's article
    missing = get_missing_entities(req.user_text, article["text"])

    # 5. Compute the real PyTorch bias score via cosine similarity to anchor vectors
    pytorch_score = _compute_bias_score(req.user_text)

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