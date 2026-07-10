"""Unit tests for scoring helpers (F1, failure attribution)."""

from __future__ import annotations

from evaluation.memory_bench.score import attribute_failure, normalize_text, token_f1


def test_normalize_text_strips_punctuation_articles_case() -> None:
    assert normalize_text("The Adoption Agencies!") == "adoption agencies"
    assert normalize_text("8 May, 2023") == "8 may 2023"


def test_token_f1_exact_and_partial() -> None:
    assert token_f1("adoption agencies", "Adoption agencies") == 1.0
    assert token_f1("she researched adoption agencies", "adoption agencies") > 0.5
    assert token_f1("completely different", "adoption agencies") == 0.0


def test_token_f1_empty_gold_or_prediction() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("something", "") == 0.0
    assert token_f1("", "gold") == 0.0


def test_attribute_failure_extraction() -> None:
    stage = attribute_failure(["s1"], sessions_with_episodes=set(), retrieved_sessions={"s2"})
    assert stage == "extraction"


def test_attribute_failure_retrieval() -> None:
    stage = attribute_failure(["s1"], sessions_with_episodes={"s1"}, retrieved_sessions={"s2"})
    assert stage == "retrieval"


def test_attribute_failure_generation() -> None:
    stage = attribute_failure(["s1"], sessions_with_episodes={"s1"}, retrieved_sessions={"s1", "s2"})
    assert stage == "generation"


def test_attribute_failure_without_evidence_defaults_to_generation() -> None:
    assert attribute_failure([], set(), set()) == "generation"
