import sys

sys.path.insert(0, "api")

import engine
import news_fetch
import vector_store


def test_topic_hits_are_case_insensitive_for_single_word_terms():
    assert vector_store._count_topic_hits(
        ["mekedatu"],
        "Karnataka discusses the Mekedatu reservoir plan.",
        set(),
    ) == (1, True)


def test_supplied_article_relevance_accepts_same_event():
    accepted, hits, anchor_hit = engine._is_relevant_supplied_article(
        "NEET UG exam paper leak NTA probe",
        "NTA says NEET-UG exam paper leak allegations are under investigation.",
    )

    assert accepted is True
    assert hits >= 2
    assert anchor_hit is True


def test_supplied_article_relevance_uses_named_entities():
    accepted, hits, _ = engine._is_relevant_supplied_article(
        "A story about a hearing in Delhi.",
        "The Supreme Court hearing continued today.",
        ["Supreme Court"],
    )

    assert accepted is True
    assert hits >= 1


def test_supplied_article_relevance_rejects_unrelated_article():
    accepted, hits, anchor_hit = engine._is_relevant_supplied_article(
        "NEET UG exam paper leak NTA probe",
        "Bollywood actor announces a wedding in Mumbai.",
    )

    assert accepted is False
    assert hits == 0
    assert anchor_hit is False


def test_news_perspectives_backfill_second_relevant_article(monkeypatch):
    articles = [
        {
            "title": "NEET paper leak probe expands",
            "description": "NTA NEET exam paper leak probe expands.",
            "url": "https://example.com/a",
            "source": {"name": "Source A"},
        },
        {
            "title": "NTA faces questions over NEET",
            "description": "NEET exam paper leak questions continue.",
            "url": "https://example.com/b",
            "source": {"name": "Source B"},
        },
    ]
    monkeypatch.setattr(news_fetch, "analyze_rss_summary", lambda _: {"bias": "Center"})

    result = news_fetch.process_perspectives(
        articles,
        keywords=["NEET", "NTA", "paper leak"],
    )

    assert sum(1 for article in result.values() if article) >= 2


def test_news_perspectives_excludes_opened_url(monkeypatch):
    articles = [
        {
            "title": "Opened article on NEET leak",
            "description": "NTA NEET exam paper leak probe.",
            "url": "https://example.com/open",
            "source": {"name": "Opened Source"},
        },
        {
            "title": "Different article on NEET leak",
            "description": "NTA NEET exam paper leak probe continues.",
            "url": "https://example.com/other",
            "source": {"name": "Other Source"},
        },
    ]
    monkeypatch.setattr(news_fetch, "analyze_rss_summary", lambda _: {"bias": "Center"})

    result = news_fetch.process_perspectives(
        articles,
        keywords=["NEET", "NTA", "paper leak"],
        exclude_url="https://example.com/open/",
    )

    urls = {article["url"] for article in result.values() if article}
    assert "https://example.com/open" not in urls


def test_news_perspectives_do_not_backfill_unrelated_india_news(monkeypatch):
    articles = [
        {
            "title": "India, Italy upgrade ties to special strategic partnership after Modi-Meloni talks",
            "description": "The two sides agreed on defence cooperation and a bilateral trade target.",
            "url": "https://example.com/italy",
            "source": {"name": "BusinessLine"},
        },
        {
            "title": "'NEET, CBSE, SSC. And today CUET': Rahul Gandhi attacks exam conduct",
            "description": "NEW DELHI: Congress MP Rahul Gandhi criticised exam irregularities.",
            "url": "https://example.com/neet",
            "source": {"name": "Times of India"},
        },
    ]
    monkeypatch.setattr(news_fetch, "analyze_rss_summary", lambda _: {"bias": "Center"})

    result = news_fetch.process_perspectives(
        articles,
        keywords=["Modi-Meloni Talks", "Italy", "strategic", "deals", "visit"],
        original_keywords=["Narendra Modi", "PM Modi", "Italy"],
    )

    urls = {article["url"] for article in result.values() if article}
    assert "https://example.com/italy" in urls
    assert "https://example.com/neet" not in urls
