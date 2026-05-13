"""
analyzer.py — bridges the model (IndicBERT + LiquidBrain) with the API.

analyze_article()     → full bias analysis for the article the user is reading
analyze_rss_summary() → lightweight tagging for RSS/ingested articles
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


def _extract_entities(text: str) -> list:
    """
    Lightweight regex-based named-entity extraction.
    Picks capitalised multi-word phrases (proper nouns) as entity candidates.
    """
    # Match sequences of Title-Case words (2–4 words), skip sentence starts
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    candidates = re.findall(pattern, text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    entities: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            entities.append(c)
    return entities[:8]


def _make_slug(text: str) -> str:
    """Turns the first sentence into a URL-safe slug."""
    first = text.split(".")[0][:60]
    return re.sub(r'[^a-z0-9]+', '-', first.lower()).strip('-')


def _summarise(text: str, max_chars: int = 200) -> str:
    """Returns the first `max_chars` characters as a rough summary."""
    clean = " ".join(text.split())
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

    return {
        "bias_classification": bias_label,
        "confidence": round(confidence, 4),
        "named_entities": entities,
        "article_summary": summary,
        "core_event_slug": slug,
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
