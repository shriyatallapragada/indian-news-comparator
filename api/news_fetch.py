import requests
import json
import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from nlp.analyzer import analyze_rss_summary

API_KEY = "5856049a571545c9b02e5d355651f250"
GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY", "")


def get_domain(url):
    return urlparse(url).netloc.replace("www.", "")


def load_bias_map():
    with open("source_bias.json", "r") as f:
        return json.load(f)


def sanitize_entity(entity: str) -> str:
    """
    Strip characters that break NewsAPI's query parser.
    Also collapse acronyms like 'University Grants Commission (UGC)' → 'UGC'
    by preferring the content inside parens if present, otherwise the full string.
    """
    import re
    # If entity has parenthetical acronym like "University Grants Commission (UGC)", use the acronym
    match = re.search(r'\(([A-Z]{2,})\)', entity)
    if match:
        return match.group(1).strip()
    return re.sub(r'[()\[\]{}&|!]', '', entity).strip()


def build_query(named_entities):
    """
    Build a strict NewsAPI query from LLM-extracted named entities.
    Every entity is quoted and joined with AND for exact phrase matching.
    """
    if not named_entities:
        return ""
    clean = [sanitize_entity(e) for e in named_entities[:3] if sanitize_entity(e)]
    return " AND ".join(f'"{e}"' for e in clean)


def is_relevant(article: dict, keywords: list, original_keywords: list = None) -> bool:
    """
    Requires the article to mention at least one of the top 2 most specific keywords
    (either sanitized or original form) in its title or description.
    Uses whole-word matching to avoid 'UGC' matching 'drug' etc.
    """
    import re
    if not keywords and not original_keywords:
        return True
    text = (
        (article.get("title") or "") + " " +
        (article.get("description") or "")
    ).lower()
    # Only check top 2 entities — most specific to the story
    priority = (keywords or [])[:2] + (original_keywords or [])[:2]
    for keyword in priority:
        # Check the full keyword phrase first
        if keyword.lower() in text:
            return True
        # Then check each significant word (5+ chars to avoid noise)
        for word in keyword.split():
            if len(word) >= 5 and re.search(rf'\b{re.escape(word.lower())}\b', text):
                return True
    return False


def fetch_from_guardian(keywords: list, keyword: str) -> list:
    """
    Fallback source with no date restriction — searches The Guardian's full archive.
    Free API key at https://open-platform.theguardian.com/access/
    Returns articles in the same shape as NewsAPI for compatibility.
    """
    if not GUARDIAN_API_KEY:
        print("No GUARDIAN_API_KEY set, skipping Guardian fallback")
        return []

    # Build query from top 2 clean keywords, or fall back to page title keyword
    if keywords:
        query = " AND ".join(f'"{e}"' for e in keywords[:2])
    else:
        query = keyword

    try:
        r = requests.get(
            "https://content.guardianapis.com/search",
            params={
                "q": query,
                "show-fields": "trailText,headline",
                "order-by": "relevance",
                "page-size": 10,
                "api-key": GUARDIAN_API_KEY,
            },
            timeout=5,
        ).json()

        results = r.get("response", {}).get("results", [])
        print(f"Guardian({query}) → {len(results)} articles")

        # Normalise to NewsAPI article shape
        articles = []
        for item in results:
            fields = item.get("fields", {})
            articles.append({
                "title": fields.get("headline") or item.get("webTitle", ""),
                "description": fields.get("trailText", ""),
                "url": item.get("webUrl", ""),
                "source": {"name": "The Guardian"},
            })
        return articles

    except Exception as e:
        print(f"Guardian API error: {e}")
        return []


def get_biased_news(keyword, keywords=None, source_event: str = ""):
    all_articles = []
    clean_keywords = [sanitize_entity(k) for k in (keywords or []) if sanitize_entity(k)]

    try:
        base_url = "https://newsapi.org/v2/everything"
        common = {"language": "en", "sortBy": "relevancy", "pageSize": 10, "apiKey": API_KEY}

        # Strategy 1: qInTitle with top 2 entities — most precise
        if clean_keywords and len(clean_keywords) >= 2:
            title_query = " AND ".join(f'"{e}"' for e in clean_keywords[:2])
            r = requests.get(base_url, params={**common, "qInTitle": title_query}, timeout=5).json()
            all_articles = r.get("articles", [])
            print(f"qInTitle({title_query}) → {len(all_articles)} articles")

        # Strategy 2: q with all entities
        if not all_articles and clean_keywords:
            q_query = " AND ".join(f'"{e}"' for e in clean_keywords)
            r = requests.get(base_url, params={**common, "q": q_query}, timeout=5).json()
            all_articles = r.get("articles", [])
            print(f"q({q_query}) → {len(all_articles)} articles")

        # Strategy 3: q with just the top entity — broadest NewsAPI fallback
        if not all_articles and clean_keywords:
            broad_query = f'"{clean_keywords[0]}"'
            r = requests.get(base_url, params={**common, "q": broad_query}, timeout=5).json()
            all_articles = r.get("articles", [])
            print(f"q broad({broad_query}) → {len(all_articles)} articles")

    except Exception as e:
        print(f"NewsAPI error: {e}")

    # Filter relevance now so we know if NewsAPI actually gave us usable articles
    relevant_from_newsapi = (
        [a for a in all_articles if is_relevant(a, clean_keywords, keywords or [])]
        if (clean_keywords or keywords)
        else all_articles
    )
    print(f"NewsAPI relevant articles: {len(relevant_from_newsapi)}")

    # Strategy 4: Guardian fallback — runs if NewsAPI returned nothing usable
    if not relevant_from_newsapi:
        print("NewsAPI yielded no relevant articles — trying Guardian API")
        guardian_articles = fetch_from_guardian(clean_keywords, keyword)
        all_articles = guardian_articles
    else:
        all_articles = relevant_from_newsapi

    return process_perspectives(all_articles, keywords=clean_keywords, original_keywords=keywords or [], source_event=source_event)


def process_perspectives(articles: list, keywords: list = None, original_keywords: list = None, source_event: str = "") -> dict:
    """
    1. Deduplicate by URL.
    2. Pre-filter by keyword overlap — drop articles that don't mention any entity.
    3. Classify remaining articles concurrently via Groq (max 10).
    4. Bucket into Left / Center / Right with publisher uniqueness.
    """
    # 1. Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for article in articles:
        url = article.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)

    # 2. Pre-filter: must mention at least one keyword word
    # (articles from get_biased_news may already be pre-filtered, but this is a safety net)
    if keywords or original_keywords:
        relevant = [a for a in unique_articles if is_relevant(a, keywords or [], original_keywords or [])]
        print(f"{len(relevant)}/{len(unique_articles)} articles passed keyword relevance filter")
        # If filtering wiped everything (e.g. Guardian articles with loose text), use all unique
        if not relevant:
            print("Relevance filter removed all articles — using unfiltered set")
            relevant = unique_articles
    else:
        relevant = unique_articles

    # Hard cap at 10 to limit Groq calls
    relevant = relevant[:10]
    print(f"Classifying {len(relevant)} articles")

    # 3. Classify concurrently
    buckets = {"Left": [], "Center": [], "Right": []}

    def classify(article):
        summary = article.get("description") or article.get("title") or ""
        if not summary:
            return None, None
        result = analyze_rss_summary(summary)
        if "error" in result:
            print(f"LLM error, skipping: {result['error'][:60]}")
            return None, None
        return article, result.get("bias")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(classify, a): a for a in relevant}
        for future in as_completed(futures):
            article, bias = future.result()
            if article and bias in buckets:
                print(f"[{bias}] {article.get('title', '')[:80]}")
                buckets[bias].append(article)

    print(f"Buckets — Left:{len(buckets['Left'])} Center:{len(buckets['Center'])} Right:{len(buckets['Right'])}")

    # 4. Select one per bucket with publisher uniqueness; missing buckets stay null
    used_publishers = set()
    output = {"left": None, "center": None, "right": None}

    for key, bucket_key in [("left", "Left"), ("center", "Center"), ("right", "Right")]:
        for article in buckets[bucket_key]:
            publisher = article.get("source", {}).get("name")
            if publisher and publisher not in used_publishers:
                output[key] = article
                used_publishers.add(publisher)
                break

    return output


# TEST
#result = get_biased_news("farmers protest")


#def display(article, label):
   # if article:
     #   print(f"\n{label}:")
      #  print("Title:", article["title"])
       # print("Source:", article["source"]["name"])
       # print("URL:", article["url"])
   # else:
      #  print(f"\n{label}: No article found")


#display(result["left"], "LEFT")
#display(result["center"], "CENTER")
#display(result["right"], "RIGHT")
