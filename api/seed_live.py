"""
seed_live.py — fetches real articles from NewsAPI/Guardian and ingests them
into ChromaDB via the running engine on port 8001.

Usage:
    python seed_live.py "topic to search"
    python seed_live.py "Modi economy"
"""

import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Import analyzer for bias classification
import sys
sys.path.insert(0, os.path.dirname(__file__))
from nlp.analyzer import analyze_rss_summary

NEWSAPI_KEY     = os.environ.get("NEWSAPI_KEY", "")
GUARDIAN_KEY    = os.environ.get("GUARDIAN_API_KEY", "")
ENGINE_URL      = os.environ.get("ENGINE_URL", "http://127.0.0.1:8001/api/ingest_live_batch")


def fetch_newsapi(topic: str) -> list:
    if not NEWSAPI_KEY:
        print("NEWSAPI_KEY not set, skipping NewsAPI")
        return []
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": topic,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 15,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=8,
        ).json()
        articles = r.get("articles", [])
        print(f"NewsAPI({topic!r}) → {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


def fetch_guardian(topic: str) -> list:
    if not GUARDIAN_KEY:
        return []
    try:
        r = requests.get(
            "https://content.guardianapis.com/search",
            params={
                "q": topic,
                "show-fields": "trailText,headline",
                "order-by": "relevance",
                "page-size": 15,
                "api-key": GUARDIAN_KEY,
            },
            timeout=8,
        ).json()
        results = r.get("response", {}).get("results", [])
        print(f"Guardian({topic!r}) → {len(results)} articles")
        return [
            {
                "title":       item.get("fields", {}).get("headline") or item.get("webTitle", ""),
                "description": item.get("fields", {}).get("trailText", ""),
                "url":         item.get("webUrl", ""),
                "source":      {"name": "The Guardian"},
            }
            for item in results
        ]
    except Exception as e:
        print(f"Guardian error: {e}")
        return []


def classify_articles(articles: list, topic: str) -> list:
    """Classify each article's bias concurrently, return payloads ready to ingest."""
    buckets = {"Left": None, "Center": None, "Right": None}
    used_sources = set()

    def classify(article):
        text = article.get("description") or article.get("title") or ""
        url  = article.get("url", "")
        source = article.get("source", {}).get("name", "Unknown")
        if not text or not url or "consent.yahoo" in url:
            return None, None, None, None
        result = analyze_rss_summary(text)
        return article, result.get("bias"), source, text

    print(f"Classifying {len(articles)} articles…")
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(classify, a) for a in articles]
        for f in as_completed(futures):
            article, bias, source, text = f.result()
            if not article or bias not in buckets:
                continue
            if buckets[bias] is None and source not in used_sources:
                buckets[bias] = {
                    "topic":      topic,
                    "source":     source,
                    "bias_label": bias,
                    "url":        article["url"],
                    "text":       text,
                }
                used_sources.add(source)
                print(f"  [{bias}] {source} — {article['url'][:70]}")

    return [v for v in buckets.values() if v is not None]


def seed(topic: str):
    print(f"\nFetching articles for: '{topic}'")

    articles = fetch_newsapi(topic)
    if not articles:
        articles = fetch_guardian(topic)

    if not articles:
        print("No articles found. Try a different topic.")
        return

    payloads = classify_articles(articles, topic)

    if not payloads:
        print("Could not classify any articles into Left/Center/Right.")
        return

    print(f"\nIngesting {len(payloads)} articles into ChromaDB…")
    res = requests.post(ENGINE_URL, json=payloads, timeout=120)

    if res.ok:
        print(f"Done. {res.json()['message']}")
    else:
        print(f"Ingest failed: {res.status_code} — {res.text}")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "India politics"
    seed(topic)
