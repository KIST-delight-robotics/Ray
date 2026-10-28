"""Tests for voice_pipeline.session_context time labels."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from voice_pipeline.memory.storage import EVENT_ROLE, RADIO_ROLE, RADIO_START, SQLiteMemoryStorage
from voice_pipeline.session_context import (
    format_current_session_marker,
    format_local_time,
    load_recent_conversations,
)

_SEOUL = ZoneInfo("Asia/Seoul")
_NOW = datetime(2026, 3, 30, 9, 0, tzinfo=_SEOUL)  # = 2026-03-30 00:00 UTC


class TestFormatLocalTime:
    @pytest.mark.parametrize(
        ("timestamp", "expected"),
        [
            ("2026-03-30 00:00:00", "2026-03-30 (Mon) 09:00, today, just now"),
            ("2026-03-29 23:59:00", "2026-03-30 (Mon) 08:59, today, 1 minute ago"),
            ("2026-03-29 21:30:00.123456", "2026-03-30 (Mon) 06:30, today, 2 hours ago"),
            ("2026-03-28 21:30:00", "2026-03-29 (Sun) 06:30, yesterday"),
            ("2026-03-20 00:00:00", "2026-03-20 (Fri) 09:00, 10 days ago"),
        ],
    )
    def test_utc_shown_in_local_time_with_relative_label(self, timestamp: str, expected: str) -> None:
        assert format_local_time(timestamp, _NOW) == expected

    def test_relative_day_follows_the_calendar_not_elapsed_hours(self) -> None:
        now = datetime(2026, 3, 30, 0, 10, tzinfo=_SEOUL)
        # 20분 전이지만 어제 날짜다
        assert format_local_time("2026-03-29 14:50:00", now) == "2026-03-29 (Sun) 23:50, yesterday"

    def test_future_time_has_no_relative_label(self) -> None:
        # 시계가 어긋나 저장 시각이 now 보다 뒤인 경우
        assert format_local_time("2026-03-30 01:00:00", _NOW) == "2026-03-30 (Mon) 10:00"


class TestFormatCurrentSessionMarker:
    def test_marker_shows_local_start_time_and_zone(self) -> None:
        assert format_current_session_marker(_NOW) == "[Current session — started 2026-03-30 (Mon) 09:00 KST]"


def _words(text: str) -> int:
    return len(text.split())


def _storage(sessions: dict[str, list[tuple[str, str, str]]]) -> SQLiteMemoryStorage:
    st = SQLiteMemoryStorage(":memory:")
    for sid, rows in sessions.items():
        for role, text, ts in rows:
            st.add_utterance(sid, role, text, ts, _words(text))
    return st


class TestLoadRecentConversations:
    def test_fills_newest_lines_backward_across_sessions_and_marks_the_cut_one(self) -> None:
        st = _storage(
            {
                "s1": [
                    ("user", "a b", "2026-03-28 10:00:00"),
                    ("assistant", "c d", "2026-03-28 10:00:05"),
                    ("user", "e f", "2026-03-28 10:00:09"),
                ],
                "s2": [("user", "g h", "2026-03-29 10:00:00"), ("assistant", "i j", "2026-03-29 10:00:05")],
                "cur": [("user", "now", "2026-03-30 00:00:00")],
            }
        )

        # 줄마다 3단어 — 예산 9 는 s2 두 줄 + s1 마지막 한 줄
        got = load_recent_conversations(st, "cur", _words, now=_NOW, max_tokens=9)

        assert got == (
            "Recent conversations:\n\n"
            "[Session — 2026-03-28 (Sat) 19:00, 2 days ago, last part only]\n"
            "User: e f\n\n"
            "[Session — 2026-03-29 (Sun) 19:00, yesterday]\n"
            "User: g h\n"
            "Ray: i j"
        )

    def test_radio_reading_and_event_lines_are_left_out(self) -> None:
        # 사용자가 끼어들지 않은 대본의 낭독은 기억 추출과 같이 뺀다
        st = _storage(
            {
                "s1": [
                    (EVENT_ROLE, f"{RADIO_START} k1", "2026-03-29 10:00:00"),
                    (RADIO_ROLE, "오늘의 이야기는", "2026-03-29 10:00:01"),
                    ("user", "고마워", "2026-03-29 10:01:00"),
                ]
            }
        )

        got = load_recent_conversations(st, "cur", _words, now=_NOW)

        assert got.endswith("yesterday]\nUser: 고마워")

    def test_no_earlier_sessions_gives_empty_string(self) -> None:
        st = _storage({"cur": [("user", "hi", "2026-03-30 00:00:00")]})
        assert load_recent_conversations(st, "cur", _words, now=_NOW) == ""
