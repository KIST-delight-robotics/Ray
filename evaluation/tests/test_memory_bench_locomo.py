"""Unit tests for the LoCoMo dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.memory_bench.datasets.locomo import load_locomo, parse_session_datetime


@pytest.fixture()
def locomo_path(tmp_path: Path) -> Path:
    """Minimal fixture covering the format quirks seen in the real file."""
    data = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "Caroline",
                "speaker_b": "Melanie",
                "session_1": [
                    {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
                    {
                        "speaker": "Melanie",
                        "dia_id": "D1:2",
                        "text": "Look at this.",
                        "img_url": "['http://example.com/x.jpg']",
                        "blip_caption": "a painting of a sunrise",
                    },
                    {"speaker": "Caroline", "dia_id": "D1:3", "text": "   "},
                ],
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "Hi again."}],
                "session_2_date_time": "10:00 am on 25 May, 2023",
                "session_10": [{"speaker": "Caroline", "dia_id": "D10:1", "text": "Long time!"}],
                "session_10_date_time": "9:05 am on 2 January, 2024",
            },
            "qa": [
                {
                    "question": "When did Melanie paint a sunrise?",
                    "answer": 2022,
                    "evidence": ["D1:2", "D2:1"],
                    "category": 2,
                },
                {
                    "question": "What did Caroline research?",
                    "answer": "Adoption agencies",
                    "evidence": "['D2:1']",
                    "category": "1",
                },
                {
                    "question": "What did Caroline realize?",
                    "adversarial_answer": "self-care is important",
                    "evidence": "['D1:1']",
                    "category": "5",
                },
            ],
        },
        {
            "sample_id": "conv-2",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hello"}],
                "session_1_date_time": "4:00 pm on 1 June, 2023",
            },
            "qa": [],
        },
    ]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(data))
    return path


def test_perspectives_are_both_speakers(locomo_path: Path) -> None:
    conv = load_locomo(locomo_path)[0]
    assert conv.perspectives == ["Caroline", "Melanie"]
    assert "Caroline and Melanie" in conv.participants_desc


def test_sessions_sorted_numerically(locomo_path: Path) -> None:
    conv = load_locomo(locomo_path)[0]
    assert [s.index for s in conv.sessions] == [1, 2, 10]


def test_session_timestamp_format(locomo_path: Path) -> None:
    conv = load_locomo(locomo_path)[0]
    assert conv.sessions[0].timestamp == "2023-05-08 13:56:00"
    assert conv.sessions[2].timestamp == "2024-01-02 09:05:00"


def test_image_caption_merged_and_blank_turn_dropped(locomo_path: Path) -> None:
    session = load_locomo(locomo_path)[0].sessions[0]
    assert len(session.turns) == 2  # 공백 텍스트 턴은 제외
    assert session.turns[1].text == "Look at this. [shares a photo: a painting of a sunrise]"


def test_qa_normalization(locomo_path: Path) -> None:
    qa = load_locomo(locomo_path)[0].qa

    assert qa[0].answer == "2022"  # int → str
    assert qa[0].category == "temporal"  # 2 → 유형명
    assert qa[0].evidence == ["D1:2", "D2:1"]
    assert qa[0].evidence_sessions == [1, 2]

    assert qa[1].category == "multi-hop"  # "1" → 유형명
    assert qa[1].evidence == ["D2:1"]  # "['D2:1']" → list

    assert qa[2].adversarial
    assert qa[2].category == "adversarial"
    assert qa[2].answer == ""
    assert qa[2].adversarial_answer == "self-care is important"


def test_sample_id_filter(locomo_path: Path) -> None:
    conversations = load_locomo(locomo_path, sample_ids=["conv-2"])
    assert [c.sample_id for c in conversations] == ["conv-2"]


def test_parse_session_datetime_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        parse_session_datetime("sometime in May")
