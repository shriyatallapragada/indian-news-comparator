# backend/main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from nlp.analyzer import analyze_article, analyze_rss_summary
from vector_store import ingest_article, find_related_by_entities
from news_fetch import get_biased_news, build_search_terms, is_relevant
import asyncio
import os
import requests as http_requests
from concurrent.futures import ThreadPoolExecutor

_thread_pool = ThreadPoolExecutor(max_workers=3)
_NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")


def _fetch_and_ingest(query: str, named_entities: list):
    """Fetch articles from NewsAPI/Guardian and ingest into ChromaDB."""
    guardian_key = os.environ.get("GUARDIAN_API_KEY", "")
    articles = []

    # Try NewsAPI with broad query first
    if _NEWSAPI_KEY:
        try:
            r = http_requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "language": "en", "sortBy": "relevancy",
                        "pageSize": 15, "apiKey": _NEWSAPI_KEY},
                timeout=8,
            ).json()
            articles = r.get("articles", [])
            print(f"[auto-fetch] NewsAPI({query!r}) → {len(articles)} articles")
        except Exception as e:
            print(f"[auto-fetch] NewsAPI error: {e}")
    else:
        print("[auto-fetch] NEWSAPI_KEY not set, skipping NewsAPI")

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
                {"title": i.get("fields", {}).get("headline", ""),
                 "description": i.get("fields", {}).get("trailText", ""),
                 "url": i.get("webUrl", ""),
                 "source": {"name": "The Guardian"}}
                for i in results
            ]
            print(f"[auto-fetch] Guardian({query!r}) → {len(articles)} articles")
        except Exception as e:
            print(f"[auto-fetch] Guardian error: {e}")

    if not articles:
        return

    search_terms = build_search_terms(query, named_entities, limit=6)

    # Classify and ingest, one per bias
    used_sources = set()
    buckets = {"Left": False, "Center": False, "Right": False}

    for article in articles:
        text = article.get("description") or article.get("title", "")
        url  = article.get("url", "")
        source = article.get("source", {}).get("name", "Unknown")
        if not text or not url or "consent.yahoo" in url or source in used_sources:
            continue
        if search_terms and not is_relevant(article, search_terms, named_entities):
            print(f"[auto-fetch] Skipping unrelated article: {article.get('title', '')[:70]}")
            continue
        tagged = analyze_rss_summary(text)
        bias = tagged.get("bias", "Center")
        if buckets.get(bias):
            continue  # already have one for this bias
        ingest_article(
            summary=text,
            bias=bias,
            named_entities=tagged.get("named_entities", []),
            core_event_slug=tagged.get("core_event_slug", ""),
            title=article.get("title", ""),
            url=url,
            source=source,
            published_at="",
        )
        buckets[bias] = True
        used_sources.add(source)
        print(f"[auto-fetch] Ingested [{bias}] {source}")
        if all(buckets.values()):
            break

app = FastAPI(title="News Comparator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ArticleRequest(BaseModel):
    text: str


class IngestRequest(BaseModel):
    summary: str
    title: str = ""
    url: str = ""
    published_at: str = ""   # ISO 8601 e.g. "2024-06-01T10:30:00Z"


class RelatedRequest(BaseModel):
    summary: str
    named_entities: list[str] = []
    published_at: str = ""


# ── Bias analysis for the current article the user is reading ──────────────
@app.post("/api/analyze")
async def analyze(request: ArticleRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    truncated = truncate_article_text(request.text)
    result = analyze_article(truncated)
    return result


# ── Ingest an RSS article into the vector DB ───────────────────────────────
@app.post("/api/ingest")
async def ingest(request: IngestRequest):
    """
    Call this for each RSS article you want to store.
    Runs the RSS prompt to get bias/entities/slug, then embeds and stores.
    """
    if not request.summary:
        raise HTTPException(status_code=400, detail="No summary provided")

    tagged = analyze_rss_summary(request.summary)
    if "error" in tagged:
        raise HTTPException(status_code=500, detail=tagged["error"])

    ingest_article(
        summary=request.summary,
        bias=tagged.get("bias", "Center"),
        named_entities=tagged.get("named_entities", []),
        core_event_slug=tagged.get("core_event_slug", ""),
        title=request.title,
        url=request.url,
        published_at=request.published_at,
    )
    return {"status": "ingested", "tagged": tagged}


# ── Find related articles from the vector DB ──────────────────────────────
@app.post("/api/related")
async def related(request: RelatedRequest):
    if not request.summary:
        raise HTTPException(status_code=400, detail="No summary provided")

    result = find_related_by_entities(
        summary=request.summary,
        named_entities=request.named_entities,
        published_at=request.published_at,
    )

    # Auto-fetch if any perspective is missing or has low entity overlap
    missing_leans = [k for k in ["left", "center", "right"] if result.get(k) is None]
    
    if missing_leans and request.summary:
        # Use stable topic terms rather than noisy bylines/title fragments.
        terms = build_search_terms(request.summary, request.named_entities, limit=5)
        query = " OR ".join(f'"{term}"' for term in terms[:3]) if terms else request.summary[:80].strip()
        print(f"[related] Missing {missing_leans} — auto-fetching for: {query!r}")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _thread_pool, _fetch_and_ingest, query, request.named_entities
        )

        result = find_related_by_entities(
            summary=request.summary,
            named_entities=request.named_entities,
            published_at=request.published_at,
        )

    return result


# ── Legacy RSS ingest endpoint (kept for compatibility) ────────────────────
@app.post("/api/ingest-rss")
async def ingest_rss(request: ArticleRequest):
    result = analyze_rss_summary(request.text)
    return result


# ── News fetch endpoint (replaces the old port-5000 Flask server) ──────────
@app.get("/news")
async def news(
    q: str = Query(..., description="Search query / article title"),
    keywords: str = Query("", description="Comma-separated named entities"),
    source_event: str = Query("", description="Core event slug"),
):
    """
    Fetches Left/Center/Right articles from NewsAPI / Guardian for the given query.
    Previously served on port 5000; now unified on port 8000.
    """
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    result = get_biased_news(q, keywords=kw_list, source_event=source_event)
    return result


# ── Helpers ────────────────────────────────────────────────────────────────
def truncate_article_text(raw_text: str, max_chars: int = 3000) -> str:
    """
    Hard-cap the article at max_chars to stay within Groq's TPM limit.
    Takes the first 2/3 and last 1/3 of the budget so we capture both
    the lede and the conclusion, which carry the most framing signal.
    """
    text = " ".join(raw_text.split())  # collapse all whitespace
    if len(text) <= max_chars:
        return text

    head = int(max_chars * 0.67)
    tail = max_chars - head
    return text[:head] + " [...] " + text[-tail:]
