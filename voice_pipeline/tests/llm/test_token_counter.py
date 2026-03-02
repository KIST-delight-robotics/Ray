"""Unit tests for the token counter factory."""

from __future__ import annotations

from voice_pipeline.llm.token_counter import create_token_counter


class TestCreateTokenCounter:
    def test_known_model_counts_tokens(self) -> None:
        counter = create_token_counter("gpt-4o")
        count = counter("hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_empty_string_returns_zero(self) -> None:
        counter = create_token_counter("gpt-4o")
        assert counter("") == 0

    def test_unknown_model_falls_back(self) -> None:
        counter = create_token_counter("totally-unknown-model-xyz")
        count = counter("hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_consistent_counts(self) -> None:
        counter = create_token_counter("gpt-4o")
        text = "The quick brown fox jumps over the lazy dog."
        assert counter(text) == counter(text)

    def test_longer_text_more_tokens(self) -> None:
        counter = create_token_counter("gpt-4o")
        short = counter("hi")
        long = counter("hi " * 100)
        assert long > short
