import re
import requests
import json
import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from nlp.analyzer import analyze_rss_summary

load_dotenv()

API_KEY = os.environ.get("NEWSAPI_KEY", "")
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
    cleaned = re.sub(r'[()\[\]{}&|!]', '', entity).strip(" '\".,:;")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned


# Generic single-word terms that make terrible NewsAPI queries
_SKIP_ENTITIES = {
    "india", "china", "us", "uk", "eu", "un", "ug", "the", "modi",
    "government", "minister", "parliament", "court", "police",
    "congress", "bjp", "party", "state", "new", "said",
    "revisiting supreme court's", "quarterly digest",
}

_STOP_WORDS = {
    "about", "after", "again", "against", "also", "amid", "among", "article",
    "before", "being", "between", "could", "court", "digest", "during",
    "from", "have", "into", "large", "news", "paper", "revisiting", "said",
    "says", "should", "sources", "supreme", "that", "their", "there",
    "these", "this", "those", "through", "under", "while", "with", "would",
}

_DOMAIN_TERMS = {
    "neet", "neet-ug", "nta", "ugc", "upsc", "exam", "examination",
    "paper leak", "paper leaks", "leak", "cancellation", "medical entrance",
}


def extract_search_terms(text: str, limit: int = 6) -> list:
    """Pull stable topic terms from article text without loading ML models."""
    if not text:
        return []

    terms = []
    seen = set()

    def add(term: str):
        term = sanitize_entity(term)
        key = term.lower()
        if not key or key in seen or key in _SKIP_ENTITIES:
            return
        if len(key) < 3 and not term.isupper():
            return
        seen.add(key)
        terms.append(term)

    for acronym in re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)?\b", text):
        add(acronym)

    lowered = text.lower()
    for phrase in sorted(_DOMAIN_TERMS, key=len, reverse=True):
        if phrase in lowered:
            add(phrase.upper() if phrase in {"neet", "neet-ug", "nta", "ugc", "upsc"} else phrase)

    for phrase in ("Supreme Court", "National Testing Agency", "paper leak"):
        if phrase.lower() in lowered:
            add(phrase)

    words = re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text)
    for word in words:
        key = word.lower().strip("'")
        if key not in _STOP_WORDS and key not in _SKIP_ENTITIES:
            add(word)
        if len(terms) >= limit:
            break

    return terms[:limit]


def build_search_terms(keyword: str = "", keywords: list = None, limit: int = 6) -> list:
    """Combine extracted entities with article/title terms for external search."""
    terms = []
    seen = set()

    def add(term: str):
        term = sanitize_entity(term)
        key = term.lower()
        if not key or key in seen or key in _SKIP_ENTITIES:
            return
        if len(key) < 3 and not term.isupper():
            return
        seen.add(key)
        terms.append(term)

    for entity in keywords or []:
        add(entity)

    for term in extract_search_terms(keyword, limit=limit):
        add(term)

    def priority(term: str) -> tuple:
        key = term.lower()
        if key in {"neet", "neet-ug", "nta", "ugc", "upsc"}:
            return (0, -len(term))
        if "paper leak" in key or "medical entrance" in key:
            return (1, -len(term))
        if key in _DOMAIN_TERMS or any(anchor in key for anchor in _DOMAIN_TERMS):
            return (2, -len(term))
        if re.fullmatch(r"[A-Z][A-Z0-9-]{2,}", term):
            return (3, -len(term))
        if "supreme court" in key or "national testing agency" in key:
            return (4, -len(term))
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", term):
            return (7, -len(term))
        return (5, -len(term))

    return sorted(terms, key=priority)[:limit]


def build_query(named_entities):
    """
    Build a NewsAPI query from named entities.
    Uses the first entity (should be a PERSON) as the primary term,
    optionally combined with the second entity via OR.
    """
    if not named_entities:
        return ""

    clean = [sanitize_entity(e) for e in named_entities if sanitize_entity(e)]
    clean = [e for e in clean if e.lower() not in _SKIP_ENTITIES]

    if not clean:
        return ""

    # Just use the top 1-2 entities with OR — simple and effective
    top = clean[:2]
    return " OR ".join(f'"{e}"' for e in top)


# Domains and title patterns to exclude — sports, entertainment, lifestyle
_EXCLUDE_PATTERNS = [
    r'\bIPL\b', r'\bstadium\b', r'\bpitch report\b', r'\bcricket\b',
    r'\bscore\b', r'\bmatch\b', r'\bwicket\b', r'\bbatting\b',
    r'\bBollywood\b', r'\bwedding\b', r'\borgasm\b', r'\bKylie\b',
    r'\bhippo\b', r'\brecipe\b', r'\bfitness\b', r'\bmarathon\b',
]
_EXCLUDE_RE = re.compile('|'.join(_EXCLUDE_PATTERNS), re.IGNORECASE)


def is_relevant(article: dict, keywords: list, original_keywords: list = None) -> bool:
    """
    Requires the article to mention at least one of the top 2 most specific keywords.
    Also excludes sports, entertainment, and lifestyle articles.
    """
    import re
    title = article.get("title") or ""
    desc  = article.get("description") or ""
    combined = title + " " + desc

    # Exclude sports/entertainment/lifestyle noise
    if _EXCLUDE_RE.search(combined):
        return False

    search_terms = build_search_terms("", (keywords or []) + (original_keywords or []), limit=8)

    if not search_terms:
        return True

    text = combined.lower()
    priority = search_terms[:5]
    for keyword in priority:
        key = keyword.lower()
        if key in text:
            return True
        for word in re.findall(r"[A-Za-z0-9-]+", key):
            if len(word) >= 4 and word not in _STOP_WORDS and re.search(rf'\b{re.escape(word)}\b', text):
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
    terms = build_search_terms(keyword, keywords or [], limit=4)
    query = " OR ".join(f'"{e}"' for e in terms[:3]) if terms else keyword

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
    clean_keywords = build_search_terms(keyword, keywords or [], limit=6)

    if API_KEY:
        try:
            base_url = "https://newsapi.org/v2/everything"
            common = {"language": "en", "sortBy": "relevancy", "pageSize": 10, "apiKey": API_KEY}

            # Strategy 1: qInTitle with the first (most specific) entity
            if clean_keywords:
                primary = clean_keywords[0]
                r = requests.get(base_url, params={**common, "qInTitle": f'"{primary}"'}, timeout=5).json()
                all_articles = r.get("articles", [])
                print(f"qInTitle({primary!r}) → {len(all_articles)} articles")

            # Strategy 2: q with OR of top 2 entities
            if not all_articles and clean_keywords:
                q_query = " OR ".join(f'"{e}"' for e in clean_keywords[:3])
                r = requests.get(base_url, params={**common, "q": q_query}, timeout=5).json()
                all_articles = r.get("articles", [])
                print(f"q OR({q_query}) → {len(all_articles)} articles")

            # Strategy 3: q with just the first entity — broadest fallback
            if not all_articles and clean_keywords:
                r = requests.get(base_url, params={**common, "q": f'"{clean_keywords[0]}"'}, timeout=5).json()
                all_articles = r.get("articles", [])
                print(f"q broad({clean_keywords[0]!r}) → {len(all_articles)} articles")

        except Exception as e:
            print(f"NewsAPI error: {e}")
    else:
        print("NEWSAPI_KEY not set, skipping NewsAPI")

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
