"""세션 시작 컨텍스트 — 장기기억에서 사용자 노트 블록(프로필·선호)과 최근 대화 원문을 만든다.

둘 다 세션 시작 instructions 에 들어간다(:func:`format_user_notes_block`, :func:`load_recent_conversations`).
여기서는 재료만 만들고 어디에 어떻게 넣는지는 지시문 조립(``engines/gpt_live/instructions``)이 정한다.

모델에게 보이는 시각은 모두 :func:`format_local_time` 을 거친다 — 저장은 UTC, 표시는 현지 시각 + 상대 시간.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from voice_pipeline.memory.storage import RADIO_ROLE, SQLiteMemoryStorage
from voice_pipeline.memory.writer import select_utterances
from voice_pipeline.types import TokenCounter

if TYPE_CHECKING:
    from voice_pipeline.memory.types import UserNote

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
# Block 2: User notes (profile + preferences)
# ---------------------------------------------------------------------------

PREFERENCES_HEADING = "How the user wants Ray to talk"  # 라이브 지시문이 이 제목으로 가리킨다


def format_user_notes_block(notes: list[UserNote]) -> str:
    """Format user notes for LLM context injection (Block 2). Empty sections are left out.

    Output example::

        About the user
        - The user's name is 재헌.
        - The user is a graduate student majoring in robotics.

        How the user wants Ray to talk
        - Speak in 반말.
    """
    sections: list[str] = []
    for heading, kind in (("About the user", "profile"), (PREFERENCES_HEADING, "preference")):
        lines = [f"- {n.text}" for n in notes if n.kind == kind]
        if lines:
            sections.append("\n".join([heading, *lines]))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Recent conversations (gpt_live): raw transcript across session boundaries
# ---------------------------------------------------------------------------

RECENT_CONVERSATIONS_HEADING = "Recent conversations"
RECENT_CONVERSATIONS_MAX_TOKENS = 1500  # 최근 대화 원문 예산 (줄 기준) — 최신 줄부터 거꾸로 채운다


def load_recent_conversations(
    memory_storage: SQLiteMemoryStorage,
    session_id: str,
    token_counter: TokenCounter,
    *,
    now: datetime,
    max_tokens: int = RECENT_CONVERSATIONS_MAX_TOKENS,
) -> str:
    """지난 세션들의 원문을 최신 줄부터 거꾸로 예산만큼 채워 시간순으로 보여준다.

    직전 1세션이 아니라 세션 경계를 넘어 채운다 — 책상 로봇 세션에는 한마디짜리가 많다. 원문은 턴 경계마다 이미
    저장돼 있어, 직전 세션의 기억 추출이 끝나기 전에 다시 깨워도 빠지지 않는다. 라디오 낭독·사건 줄은 기억
    추출과 같은 규칙(:func:`~voice_pipeline.memory.writer.select_utterances`)으로 뺀다. 예산에 걸린 세션은
    앞부분을 잘라 뒤쪽 줄만 남기고 머리에 표시한다.

    Args:
        memory_storage: Utterance storage.
        session_id: Current session (excluded).
        token_counter: Token counter for the budget.
        now: Session start time (local timezone) — basis for the session headers.
        max_tokens: Budget over the transcript lines.

    Returns:
        ``"Recent conversations:\\n\\n[Session — …]\\nUser: …\\nRay: …"``. 지난 대화가 없으면 빈 문자열.
    """
    blocks: list[str] = []
    spent = 0
    for sid, started_at in memory_storage.get_recent_sessions(exclude_session_id=session_id):
        rows = select_utterances(memory_storage.get_utterances(sid))
        lines = [_transcript_line(role, text) for role, text, _, _ in rows]
        kept: list[str] = []
        for line in reversed(lines):
            cost = token_counter(line)
            if spent + cost > max_tokens:
                break
            kept.append(line)
            spent += cost
        cut = len(kept) < len(lines)
        if kept:
            header = f"[Session — {format_local_time(started_at, now)}{', last part only' if cut else ''}]"
            blocks.append("\n".join([header, *reversed(kept)]))
        if cut:
            break
    if not blocks:
        return ""
    return f"{RECENT_CONVERSATIONS_HEADING}:\n\n" + "\n\n".join(reversed(blocks))


def _transcript_line(role: str, text: str) -> str:
    speaker = "User" if role == "user" else "Ray (radio talk)" if role == RADIO_ROLE else "Ray"
    return f"{speaker}: {text}"
