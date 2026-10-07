"""DJ 모드 소재 — ``assets/dj_segments/*.md`` 를 읽어 "이야기 → 노래" 세그먼트를 고르고 기록한다.

파일 하나가 세그먼트 하나. frontmatter 에 ``song``(카탈로그 키)과 ``topic``, 본문은 대화 모델이 자기 말로
풀 재료(불릿 노트). 노트는 문장이 아니라 재료 목록이어야 한다 — 문장으로 쓰면 대본이 된다.
노트 원문은 해석하지 않고 그대로 대화 모델(``thinking.append``)과 백엔드(``## Current segment`` 절)에 넘긴다.
텍스트 포맷은 :mod:`~voice_pipeline.engines.gpt_live.instructions` 에 있다.

고르기: 한 적 없는 세그먼트 중 무작위. 전부 했으면 오래된 순. 같은 세션에서 이미 쓴 것은 건너뛰고,
세션 안에서 다 돌면 다시 처음부터. 기록은 SQLite ``dj_segment_plays`` 에 남아 세션을 넘어 중복을 피한다.
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from voice_pipeline.engines.gpt_live.songs import Song
from voice_pipeline.settings import DEFAULT_DB_PATH, DJ_SEGMENTS_DIR

logger = logging.getLogger("voice_pipeline.dj")

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class Segment:
    """세그먼트 하나. ``key`` 는 파일 stem."""

    key: str
    topic: str
    song: Song
    notes: str  # frontmatter 를 뺀 본문 원문 (불릿 노트)

    @property
    def song_label(self) -> str:
        """곡 소개용 ``제목 — 가수``."""
        return f"{self.song.title} — {self.song.artist}" if self.song.artist else self.song.title


CLOSING_VERB = "들려드릴게요"  # 라이브 지시문의 종료 문구 "<artist>의 <title>, 들려드릴게요." 의 끝


def _squash(text: str) -> str:
    """띄어쓰기·문장부호·따옴표를 지우고 소문자로 — 전사 표기 차이("십센치의서랍", '야생화', ‘꿈처럼’)를 없앤다."""
    return re.sub(r"[\W_]+", "", text).lower()


def ends_with_closing_line(text: str, song: Song) -> bool:
    """모델 전사가 종료 문구 ``<가수>의 <제목>, 들려드릴게요`` 로 끝나는가. 가수·제목은 카탈로그 별칭도 인정한다.

    끝만 본다 — 문구 뒤에 말을 덧붙였으면 이야기가 끝났다고 보지 않는다. 가수가 없는 곡은 ``<제목> 들려드릴게요``.
    """
    squashed = _squash(text)
    verb = _squash(CLOSING_VERB)
    if not squashed.endswith(verb):
        return False
    head = squashed[: -len(verb)]
    titles = {t for t in map(_squash, (song.title, *song.aliases)) if t}
    artists = {a for a in map(_squash, (song.artist, *song.artist_aliases)) if a}
    if not artists:
        return any(head.endswith(t) for t in titles)
    return any(head.endswith(f"{a}의{t}") for a in artists for t in titles)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        raise ValueError("frontmatter (--- … ---) missing")
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"bad frontmatter line {line!r}")
        fields[key.strip()] = value.split("#", 1)[0].strip()  # 줄 끝 주석 제거
    return fields, text[m.end() :]


def load_segments(
    segments_dir: str | Path = DJ_SEGMENTS_DIR, catalog: Mapping[str, Song] | None = None
) -> dict[str, Segment]:
    """폴더의 ``*.md`` 를 전부 읽는다. 곡이 카탈로그에 없으면(재생 불가) 경고하고 뺀다.

    Args:
        segments_dir: 소재 폴더. 없으면 빈 dict.
        catalog: 재생 가능한 곡. 비어 있으면 모든 소재가 빠진다.

    Raises:
        RuntimeError: 파일을 읽을 수 없거나 frontmatter 형식이 틀렸을 때.
    """
    root = Path(segments_dir)
    if not root.is_dir():
        logger.info("DJ segments dir not found (%s) — DJ tools disabled", root)
        return {}
    catalog = catalog or {}
    segments: dict[str, Segment] = {}
    for path in sorted(root.glob("*.md")):
        try:
            fields, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"DJ segment load failed ({path}): {exc}") from exc
        song_key, topic, notes = fields.get("song", ""), fields.get("topic", ""), body.strip()
        if not song_key or not topic or not notes:
            raise RuntimeError(f"DJ segment load failed ({path}): needs 'song', 'topic' and notes body")
        song = catalog.get(song_key)
        if song is None:
            logger.warning("DJ segment %r skipped — song %r not playable", path.stem, song_key)
            continue
        segments[path.stem] = Segment(key=path.stem, topic=topic, song=song, notes=notes)
    logger.info("DJ segments: %d loaded", len(segments))
    return segments


class SegmentLog:
    """세그먼트를 시작한 기록 (SQLite ``dj_segment_plays``). 세션을 넘어 중복을 피하는 근거."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS dj_segment_plays ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, segment_key TEXT NOT NULL, played_at TEXT NOT NULL)"
            )
            self._conn.commit()

    def record(self, key: str) -> None:
        """시작을 기록한다 (끝까지 못 가도 한 것으로 친다)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO dj_segment_plays (segment_key, played_at) VALUES (?, ?)",
                (key, datetime.now(UTC).isoformat()),
            )
            self._conn.commit()

    def played_keys(self) -> list[str]:
        """한 적 있는 키 — 마지막으로 한 시각이 오래된 순."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT segment_key, MAX(played_at) AS last FROM dj_segment_plays"
                " GROUP BY segment_key ORDER BY last ASC"
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SegmentPicker:
    """다음 세그먼트를 고르고 기록한다. 프로세스 수명. 세션마다 :meth:`new_session` 으로 세션 내 기록을 비운다."""

    def __init__(self, segments: Mapping[str, Segment], log: SegmentLog, *, rng: random.Random | None = None) -> None:
        self._segments = dict(segments)
        self._log = log
        self._rng = rng or random.Random()
        self._session_used: set[str] = set()

    @property
    def segments(self) -> Mapping[str, Segment]:
        return self._segments

    def new_session(self) -> None:
        """세션 시작 — 세션 내 사용 기록을 비운다."""
        self._session_used.clear()

    def pick(self) -> Segment | None:
        """다음 세그먼트. 한 적 없는 것 중 무작위 → 전부 했으면 오래된 순. 소재가 없으면 None."""
        if not self._segments:
            return None
        if self._session_used >= self._segments.keys():
            self._session_used.clear()  # 세션 안에서 한 바퀴 돌았다 — 다시 처음부터
        played = self._log.played_keys()
        fresh = [k for k in self._segments if k not in played and k not in self._session_used]
        if fresh:
            key = self._rng.choice(fresh)
        else:
            key = next(k for k in played if k in self._segments and k not in self._session_used)
        self._session_used.add(key)
        self._log.record(key)
        return self._segments[key]
