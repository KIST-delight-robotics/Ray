"""Tests for voice_pipeline.session_context time labels."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from voice_pipeline.session_context import format_current_session_marker, format_local_time

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
