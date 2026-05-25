"""
analyzer.py — bridges the model (IndicBERT + LiquidBrain) with the API.

analyze_article()     → full bias analysis for the article the user is reading
analyze_rss_summary() → lightweight tagging for RSS/ingested articles
extract_search_query() → LLM-powered 2-3 word NewsAPI search query
"""

import sys
import os
import re
import torch
from typing import Optional

# Make the repo root importable so `model` package resolves correctly
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model import IndicNewsEmbedder, NewsComparatorBrain

# ── spaCy singleton ────────────────────────────────────────────────────────
_spacy_nlp = None

def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

# ── Singleton model instances (loaded once at import time) ─────────────────
_embedder: Optional[IndicNewsEmbedder] = None
_brain: Optional[NewsComparatorBrain] = None

# Reference vectors for Left / Center / Right bias anchors
_BIAS_ANCHORS = {
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

_anchor_states: dict = {}


def _load_models():
    global _embedder, _brain, _anchor_states
    if _embedder is not None:
        return
    print("[analyzer] Loading IndicBERT + LiquidBrain…")
    _embedder = IndicNewsEmbedder()
    _brain = NewsComparatorBrain()
    _brain.eval()

    # Pre-compute anchor states once
    with torch.no_grad():
        for label, text in _BIAS_ANCHORS.items():
            vec = _embedder.get_embeddings(text)
            _anchor_states[label] = _brain(vec)
    print("[analyzer] Models ready.")


def _classify_bias(text: str) -> tuple:
    """
    Embeds `text`, runs it through the LiquidBrain, then picks the bias label
    whose anchor state is closest (lowest divergence = most similar).
    Returns (label, confidence_0_to_1).
    """
    _load_models()
    with torch.no_grad():
        vec = _embedder.get_embeddings(text)
        state = _brain(vec)

    divergences = {
        label: _brain.calculate_divergence(state, anchor)
        for label, anchor in _anchor_states.items()
    }
    # Closest anchor = lowest divergence
    best_label = min(divergences, key=divergences.get)
    # Confidence: how much closer the best is vs the average of the others
    others = [v for k, v in divergences.items() if k != best_label]
    confidence = (sum(others) / len(others)) - divergences[best_label]
    confidence = max(0.0, min(1.0, confidence))
    return best_label, confidence


def _clean_text(text: str) -> str:
    """
    Strip common web boilerplate patterns before NLP processing.
    Removes site-name prefixes like "India News The Hindu |", nav fragments, etc.
    """
    # Remove leading "Section Name | Site Name" patterns
    text = re.sub(r'^[^.!?]{0,80}[|\-–]\s*', '', text)
    # Remove "India News", "World News" type section labels at the start
    text = re.sub(r'^(?:India|World|Business|Sports|Tech|Politics)\s+News\s+', '', text, flags=re.IGNORECASE)
    # Remove common byline prefixes before entity extraction.
    text = re.sub(r'\b(?:By|Author|Written by)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', '', text)
    return text.strip()


def _extract_entities(text: str) -> list:
    """
    spaCy-based named entity extraction.
    Returns PERSON, GPE, ORG, EVENT entities — skips boilerplate fragments.
    """
    import spacy
    try:
        nlp = _get_spacy()
    except Exception:
        # Fallback to regex if spaCy unavailable
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        candidates = re.findall(pattern, text)
        seen: set = set()
        entities: list = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                entities.append(c)
        return entities[:8]

    cleaned = _clean_text(text)
    doc = nlp(cleaned[:1000])

    KEEP = {"PERSON", "GPE", "ORG", "EVENT"}
    SKIP = {"india", "china", "us", "uk", "eu", "un", "government",
            "parliament", "court", "police", "congress", "bjp", "party"}
    SKIP_FRAGMENTS = {
        "revisiting", "quarterly digest", "follow us", "read more",
        "latest news", "supreme court's",
    }

    # Collect by type
    by_type: dict = {"PERSON": [], "GPE": [], "ORG": [], "EVENT": []}
    seen: set = set()

    for ent in doc.ents:
        name = ent.text.strip()
        key  = name.lower()
        dedupe_key = re.sub(r'^(?:the|a|an)\s+', '', key)
        start = max(0, ent.start_char - 25)
        prefix = cleaned[start:ent.start_char].lower()
        if (ent.label_ in KEEP
                and len(name) > 3
                and dedupe_key not in seen
                and key not in SKIP
                and not any(fragment in key for fragment in SKIP_FRAGMENTS)
                and not (ent.label_ == "PERSON" and re.search(r'\b(by|author|written by)\s*$', prefix))
                and not name.isdigit()):
            seen.add(dedupe_key)
            by_type[ent.label_].append(name)

    # Acronyms like NEET-UG and NTA are often the strongest topic anchors,
    # but spaCy may miss them or treat them inconsistently.
    acronyms = []
    for match in re.findall(r'\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)?\b', cleaned[:1200]):
        key = match.lower()
        if key not in seen and key not in {"html", "http", "https", "ug"}:
            seen.add(key)
            acronyms.append(match)

    for phrase in ("Supreme Court", "National Testing Agency", "paper leak"):
        key = phrase.lower()
        if key in cleaned.lower() and key not in seen:
            seen.add(key)
            acronyms.append(phrase)

    # Build final list: acronyms/topic institutions first, then PERSON/GPE/ORG.
    entities: list = []
    entities.extend(acronyms)
    for label in ("PERSON", "GPE", "ORG"):
        entities.extend(by_type[label])
        if len(entities) >= 8:
            break

    return entities[:8]


def _make_slug(text: str) -> str:
    """Turns the first sentence into a URL-safe slug."""
    first = text.split(".")[0][:60]
    return re.sub(r'[^a-z0-9]+', '-', first.lower()).strip('-')


def _summarise(text: str, max_chars: int = 200) -> str:
    """Returns the first `max_chars` characters as a rough summary, boilerplate stripped."""
    clean = " ".join(_clean_text(text).split())
    return clean[:max_chars] + ("…" if len(clean) > max_chars else "")


# ── Public API ─────────────────────────────────────────────────────────────

def analyze_article(text: str) -> dict:
    """
    Full analysis for the article the user is currently reading.
    Returns the shape expected by the extension's popup.js.
    """
    bias_label, confidence = _classify_bias(text)
    entities = _extract_entities(text)
    summary = _summarise(text)
    slug = _make_slug(text)
    search_query = extract_search_query(text)

    return {
        "bias_classification": bias_label,
        "confidence": round(confidence, 4),
        "named_entities": entities,
        "article_summary": summary,
        "core_event_slug": slug,
        "search_query": search_query,
        "step_1_target_analysis": (
            f"Entities detected: {', '.join(entities)}" if entities
            else "No prominent named entities detected."
        ),
        "step_2_alignment_logic": (
            f"The narrative framing aligns most closely with {bias_label}-leaning language "
            f"(divergence confidence: {confidence:.2f})."
        ),
    }


def analyze_rss_summary(summary: str) -> dict:
    """
    Lightweight tagging for RSS/ingested articles.
    Returns bias, named_entities, and core_event_slug.
    """
    bias_label, _ = _classify_bias(summary)
    entities = _extract_entities(summary)
    slug = _make_slug(summary)

    return {
        "bias": bias_label,
        "named_entities": entities,
        "core_event_slug": slug,
    }


def extract_search_query(text: str) -> str:
    """
    Uses Groq LLM to extract a precise 2-4 word NewsAPI search query
    from the article text. Falls back to top spaCy entities if LLM fails.

    Example: "CBI probe Supreme Court Cockroach Janta Party fake advocates"
    → returns: "Supreme Court CBI Cockroach Janta Party"
    """
    import os
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        entities = _extract_entities(text)
        return " ".join(entities[:2]) if entities else text[:60]

    prompt = (
        "Extract the best short search query for finding related news about this article. "
        "Return only 2-4 words or a short phrase, without punctuation or explanation. "
        "Prefer Indian political terminology when relevant.\n\n"
        f"Article:\n{text[:1800].strip()}"
    )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        query = response.choices[0].message.content.strip()
        query = re.sub(r'^["\']+|["\']+$', '', query).strip()
        query = re.sub(r"\s+", " ", query)
        if query:
            return query
    except Exception as e:
        print(f"[analyzer] Groq search query extraction failed: {e}")

    entities = _extract_entities(text)
    return " ".join(entities[:2]) if entities else text[:60]

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": (
                    "Extract the single most searchable 2-4 word phrase from this Indian news article "
                    "that would find related articles on NewsAPI. Return ONLY the search phrase, nothing else.\n\n"
                    f"Article: {text[:400]}"
                )
            }],
            temperature=0.0,
            max_tokens=20,
        )
        query = response.choices[0].message.content.strip().strip('"').strip("'")
        # Sanity check — must be short and not a sentence
        if query and len(query) < 60 and "\n" not in query:
            print(f"[analyzer] LLM search query: {query!r}")
            return query
    except Exception as e:
        print(f"[analyzer] LLM query extraction failed: {e}")

    # Fallback to spaCy entities
    entities = _extract_entities(text)
    return " ".join(entities[:2]) if entities else text[:60]
