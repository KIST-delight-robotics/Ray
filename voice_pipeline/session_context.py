"""세션 시작 컨텍스트 — 장기기억에서 사용자 프로필 블록과 최근 세션 블록을 만든다.

두 엔진이 같은 재료를 쓴다: cascade 의 :class:`~voice_pipeline.engines.cascade.context_builder.ContextBuilder`
는 턴마다 쌓는 입력의 블록 2·3 으로, gpt_live 는 세션 시작 instructions 뒤에 한 번 붙인다.
여기서는 재료만 만들고 어디에 어떻게 넣는지는 각 엔진이 정한다.

모델에게 보이는 시각은 모두 :func:`format_local_time` 을 거친다 — 저장은 UTC, 표시는 현지 시각 + 상대 시간.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.types import TokenCounter

if TYPE_CHECKING:
    from voice_pipeline.memory.types import Episode, Profile

# Per-message API framing overhead (role markers, separators — empirically
# measured for the OpenAI Responses API). 엔진 조립기(ContextBuilder)도 같은 값을 쓴다.
PER_MESSAGE_OVERHEAD_TOKENS = 3


# ---------------------------------------------------------------------------
# Time labels
# ---------------------------------------------------------------------------

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")  # 기기 로캘(한국어)과 무관하게 영문


def format_local_time(timestamp: str, now: datetime) -> str:
    """저장된 UTC 시각을 모델에게 보일 현지 시각으로 — ``2026-10-26 (Mon) 15:05, yesterday``.

    상대 시간까지 붙여 모델이 시차·날짜를 계산하지 않게 한다.

    Args:
        timestamp: UTC 시각 문자열 (``'%Y-%m-%d %H:%M:%S'``, 마이크로초가 붙어도 된다).
        now: 현재 시각. 이 시각의 시간대로 표시하고 상대 시간을 잰다.
    """
    local = datetime.fromisoformat(timestamp).replace(tzinfo=UTC).astimezone(now.tzinfo)
    relative = _relative_label(local, now)
    return f"{_day_and_time(local)}, {relative}" if relative else _day_and_time(local)


def format_current_session_marker(now: datetime) -> str:
    """현재 세션이 시작되는 자리의 표시 — ``[Current session — started 2026-10-27 (Tue) 16:28 KST]``.

    Args:
        now: 세션 시작 시각 (시간대 있음).
    """
    return f"[Current session — started {_day_and_time(now)} {now:%Z}]"


def _day_and_time(dt: datetime) -> str:
    return f"{dt:%Y-%m-%d} ({_WEEKDAYS[dt.weekday()]}) {dt:%H:%M}"


def _relative_label(local: datetime, now: datetime) -> str:
    """오늘은 분·시간 단위, 그 전은 달력 날짜 차이. 미래 시각(시계 어긋남)이면 빈 문자열."""
    if local > now:
        return ""
    days = (now.date() - local.date()).days
    if days == 1:
        return "yesterday"
    if days > 1:
        return f"{days} days ago"
    minutes = int((now - local).total_seconds() // 60)
    if minutes < 1:
        return "today, just now"
    if minutes < 60:
        return f"today, {minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    return f"today, {hours} hour{'s' if hours != 1 else ''} ago"


# ---------------------------------------------------------------------------
# Block 2: Profile
# ---------------------------------------------------------------------------


def format_profile_block(profiles: list[Profile]) -> str:
    """Format user profiles for LLM context injection (Block 2).

    Output example::

        [User Profile]
        basic_info::name: Alice
        interest::movie: SF, especially Nolan
    """
    if not profiles:
        return ""
    # Sort by (topic, sub_topic) for stable ordering
    sorted_profiles = sorted(profiles, key=lambda p: (p.topic, p.sub_topic))
    lines = ["[User Profile]"]
    for p in sorted_profiles:
        lines.append(f"{p.topic}::{p.sub_topic}: {p.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block 3: Previous session summaries
# ---------------------------------------------------------------------------


def format_session_summary_block(
    started_at: str,
    episodes: list[Episode],
    now: datetime,
) -> str:
    """Format a single previous session's episodes as a summary block.

    Args:
        started_at: Session start timestamp (UTC, '%Y-%m-%d %H:%M:%S').
        episodes: Episodes extracted from the session.
        now: Current time — the header is shown in its timezone, with a relative label.

    Output example::

        [Session — 2026-03-28 (Sat) 23:00, yesterday]
        - User talked about watching Dune 2 over the weekend.
        - User said the Interstellar OST is their favorite.
    """
    header = f"[Session — {format_local_time(started_at, now)}]"
    if not episodes:
        return f"{header}\n(no summary available)"
    lines = [header]
    for ep in episodes:
        lines.append(f"- {ep.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session-level context: profiles (Block 2) + recent sessions (Block 3)
# ---------------------------------------------------------------------------

RECENT_SESSIONS_MAX_TOKENS = 512  # 최근 세션 블록 soft cap — 최신 세션 1개는 캡 무관 보장
SESSION_PAGE_SIZE = 20  # 세션 에피소드 lazy 로딩 배치 크기 — 순수 조회 배치, 동작에 영향 없음


def load_session_context(
    memory_storage: SQLiteMemoryStorage,
    session_id: str,
    token_counter: TokenCounter,
    *,
    now: datetime,
    carryover_session_id: str | None = None,
    max_tokens: int = RECENT_SESSIONS_MAX_TOKENS,
    page_size: int = SESSION_PAGE_SIZE,
) -> tuple[list[Profile], list[str], set[str]]:
    """Load profiles and the recent-sessions block from memory storage.

    Shared by the cascade ``ContextBuilder`` and the gpt_live engine's
    session-start instructions. Sessions without episodes (extraction
    pending, failed, or judged meaningless) are skipped — in cascade the
    carryover covers the only session whose extraction can still be
    legitimately in flight. The walk continues into older sessions until
    the soft cap binds or history is exhausted.

    Args:
        memory_storage: Episode/profile storage.
        session_id: Current session (always excluded).
        token_counter: Token counter for the soft cap.
        now: Session start time (local timezone) — basis for the session headers.
        carryover_session_id: Session already shown verbatim as carryover
            (excluded from the block). ``None`` if no carryover.
        max_tokens: Soft cap for the block.
        page_size: Episode loading batch size (pure query batching).

    Returns:
        (profiles, block texts in chronological order,
        session IDs actually included in the block).
    """
    profiles = memory_storage.get_all_profiles()
    candidates = _iter_session_blocks(memory_storage, session_id, carryover_session_id, page_size, now)
    selected = select_recent_blocks(candidates, token_counter, max_tokens=max_tokens)
    block_texts = [text for _, text in selected]
    included_ids = {sid for sid, _ in selected if sid is not None}
    return profiles, block_texts, included_ids


def _iter_session_blocks(
    memory_storage: SQLiteMemoryStorage,
    session_id: str,
    carryover_session_id: str | None,
    page_size: int,
    now: datetime,
) -> Iterator[tuple[str, str]]:
    """Yield (session_id, block_text) newest-first, skipping episode-less sessions.

    Episode loading is paged and lazy — the consumer stops pulling once the
    soft cap binds, so sessions beyond that point are never fetched.
    """
    sessions = memory_storage.get_recent_sessions(exclude_session_id=session_id)
    for start in range(0, len(sessions), page_size):
        page = [(sid, ts) for sid, ts in sessions[start : start + page_size] if sid != carryover_session_id]
        episodes_by_sid = memory_storage.get_episodes_by_session_ids([sid for sid, _ in page])
        for sid, started_at in page:
            episodes = episodes_by_sid.get(sid, [])
            if not episodes:
                continue
            yield sid, format_session_summary_block(started_at, episodes, now)


def select_recent_blocks(
    candidates: Iterable[tuple[str | None, str]],
    token_counter: TokenCounter,
    *,
    max_tokens: int = RECENT_SESSIONS_MAX_TOKENS,
) -> list[tuple[str | None, str]]:
    """Fill whole sessions newest-first under the soft cap; return chronological.

    The newest candidate is always included regardless of size (soft cap);
    older ones are appended while the running total stays within
    ``max_tokens``, stopping at the first that no longer fits (keeps the
    block temporally contiguous). Consumes the candidates iterable lazily.
    """
    selected: list[tuple[str | None, str]] = []
    spent = 0
    for sid, text in candidates:
        cost = token_counter(text) + PER_MESSAGE_OVERHEAD_TOKENS
        if selected and spent + cost > max_tokens:
            break
        selected.append((sid, text))
        spent += cost
    selected.reverse()
    return selected
