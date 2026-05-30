import sys

sys.path.insert(0, "api")

import engine
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


def test_supplied_article_relevance_rejects_unrelated_article():
    accepted, hits, anchor_hit = engine._is_relevant_supplied_article(
        "NEET UG exam paper leak NTA probe",
        "Bollywood actor announces a wedding in Mumbai.",
    )

    assert accepted is False
    assert hits == 0
    assert anchor_hit is False
