"""이야기 대본 — ``assets/song_stories/*.md`` 를 읽어 "이야기 → 노래" 콘텐츠의 후보·선택·기록을 맡는다.

파일 하나가 이야기 하나. frontmatter(``song``, ``summary``, ``tags``, …) 뒤의 본문은 구조를 해석하지 않고
통째로 이야기용 백엔드 지시문에 들어간다(:mod:`~voice_pipeline.engines.gpt_live.instructions` 의
``build_story_backend_instructions``).
고를 때 백엔드가 보는 것은 요약·태그·곡뿐이다.

흐름: 백엔드 ``search_stories(query)`` → :meth:`StorySelector.search` (한 이야기는 제외, 나머지에서 키워드 점수 +
무작위 채움) → 백엔드가 하나 골라 ``start_story(id)`` → :meth:`StorySelector.begin` (직전 검색 결과에 있던 id 만
받고 :class:`StoryLog` 에 기록). ``status`` 는 작성자 참고용이며 로드에 영향을 주지 않는다.
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from voice_pipeline.engines.gpt_live.songs import Song
from voice_pipeline.settings import DEFAULT_DB_PATH, SONG_STORIES_DIR, STORY_SEARCH_LIMIT

logger = logging.getLogger("voice_pipeline.stories")

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
# 요청문에 늘 붙지만 이야기를 가려내지 못하는 말 — 점수에서 뺀다
_STOP_WORDS = frozenset(
    {"이야기", "얘기", "스토리", "노래", "곡", "해줘", "해", "줘", "들려줘", "하나", "좀", "그냥", "아무거나"}
)


@dataclass(frozen=True)
class Story:
    """이야기 하나. ``key`` 는 파일 stem, ``song`` 은 끝에 트는 곡의 카탈로그 키."""

    key: str
    song: str
    summary: str
    tags: tuple[str, ...]
    body: str  # frontmatter 를 뺀 본문 전체 (곡 프로필 + 단락 + 진행 지침)
    song_title: str = ""
    song_artist: str = ""

    def describe(self) -> dict[str, object]:
        """백엔드가 고를 때 보는 항목 — 본문은 넣지 않는다."""
        song = f"{self.song_title} — {self.song_artist}" if self.song_artist else self.song_title
        return {"id": self.key, "summary": self.summary, "tags": list(self.tags), "song": song}

    def search_text(self) -> str:
        """검색 매칭 대상 (요약·태그·곡명·가수·키)."""
        return " ".join((self.key, self.summary, *self.tags, self.song_title, self.song_artist, self.song)).lower()


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


def _parse_tags(value: str) -> tuple[str, ...]:
    return tuple(t.strip().strip("'\"") for t in value.strip("[]").split(",") if t.strip().strip("'\""))


def load_stories(
    stories_dir: str | Path = SONG_STORIES_DIR, catalog: Mapping[str, Song] | None = None
) -> dict[str, Story]:
    """폴더의 ``*.md`` 를 전부 읽는다. 곡이 카탈로그에 없으면(재생 불가) 경고하고 뺀다.

    Args:
        stories_dir: 대본 폴더. 없으면 빈 dict.
        catalog: 재생 가능한 곡. None 이면 곡 존재를 확인하지 않는다(테스트용).

    Raises:
        RuntimeError: 파일을 읽을 수 없거나 frontmatter 형식이 틀렸을 때.
    """
    root = Path(stories_dir)
    if not root.is_dir():
        logger.info("Story dir not found (%s) — story tools disabled", root)
        return {}
    stories: dict[str, Story] = {}
    for path in sorted(root.glob("*.md")):
        try:
            fields, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Story load failed ({path}): {exc}") from exc
        song_key = fields.get("song", "").strip()
        if not song_key or not fields.get("summary", "").strip():
            raise RuntimeError(f"Story load failed ({path}): frontmatter needs 'song' and 'summary'")
        song = catalog.get(song_key) if catalog is not None else None
        if catalog is not None and song is None:
            logger.warning("Story %r skipped — song %r not playable", path.stem, song_key)
            continue
        stories[path.stem] = Story(
            key=path.stem,
            song=song_key,
            summary=fields["summary"].strip(),
            tags=_parse_tags(fields.get("tags", "")),
            body=body.strip(),
            song_title=song.title if song else "",
            song_artist=song.artist if song else "",
        )
    logger.info("Stories: %d loaded", len(stories))
    return stories


class StoryLog:
    """이야기를 시작한 기록 (SQLite ``story_plays``). 검색이 최근에 한 이야기를 빼는 근거."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS story_plays ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, story_key TEXT NOT NULL, played_at TEXT NOT NULL)"
            )
            self._conn.commit()

    def record(self, key: str) -> None:
        """이야기 시작을 기록한다 (끝까지 못 가도 한 것으로 친다)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO story_plays (story_key, played_at) VALUES (?, ?)", (key, datetime.now(UTC).isoformat())
            )
            self._conn.commit()

    def played_keys(self) -> list[str]:
        """한 적 있는 이야기 키 — 마지막으로 한 시각이 오래된 순."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT story_key, MAX(played_at) AS last FROM story_plays GROUP BY story_key ORDER BY last ASC"
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def search_stories(
    stories: Mapping[str, Story],
    query: str,
    *,
    played: Sequence[str] = (),
    limit: int = STORY_SEARCH_LIMIT,
    rng: random.Random | None = None,
) -> list[Story]:
    """후보를 고른다. 한 적 없는 이야기만 대상으로 하되, 전부 했으면 오래된 것부터 다시 후보가 된다.

    ``query`` 의 단어가 요약·태그·곡명에 들어 있는 이야기를 점수순으로 앞에 두고, 남는 자리는 무작위로 채운다.
    빈 query 면 전부 무작위.
    """
    rng = rng or random.Random()
    played_set = set(played)
    fresh = [s for s in stories.values() if s.key not in played_set]
    if fresh:
        pool = fresh
    else:
        order = {k: i for i, k in enumerate(played)}
        pool = sorted(stories.values(), key=lambda s: order.get(s.key, -1))[:limit]  # 오래된 순
    words = [w.lower() for w in _TOKEN_RE.findall(query) if w not in _STOP_WORDS]
    scored: list[tuple[int, Story]] = []
    for story in pool:
        text = story.search_text()
        scored.append((sum(1 for w in words if w in text), story))
    rng.shuffle(scored)  # 같은 점수끼리는 무작위 순서
    matched = [s for score, s in sorted(scored, key=lambda t: -t[0]) if score > 0][:limit]
    rest = [s for _, s in scored if s not in matched]
    rng.shuffle(rest)
    return matched + rest[: max(0, limit - len(matched))]


class StorySelector:
    """검색과 시작을 잇는다 — ``start_story`` 는 직전 검색 결과에 있던 id 만 받는다.

    검색 핸들러는 executor 스레드에서, ``begin`` 은 루프 스레드에서 부르므로 결과 집합은 락으로 보호한다.
    """

    def __init__(
        self,
        stories: Mapping[str, Story],
        log: StoryLog,
        *,
        limit: int = STORY_SEARCH_LIMIT,
        rng: random.Random | None = None,
    ) -> None:
        self._stories = dict(stories)
        self._log = log
        self._limit = limit
        self._rng = rng
        self._lock = threading.Lock()
        self._offered: frozenset[str] = frozenset()

    @property
    def stories(self) -> Mapping[str, Story]:
        return self._stories

    def search(self, query: str) -> list[Story]:
        """후보를 돌려주고 그 id 들을 다음 ``begin`` 의 허용 집합으로 기억한다."""
        found = search_stories(self._stories, query, played=self._log.played_keys(), limit=self._limit, rng=self._rng)
        with self._lock:
            self._offered = frozenset(s.key for s in found)
        return found

    def begin(self, key: str) -> Story | None:
        """검색 결과에 있던 id 면 기록하고 이야기를 돌려준다. 아니면 None (검색을 거치게 한다)."""
        with self._lock:
            allowed = key in self._offered
            if allowed:
                self._offered = frozenset()
        if not allowed:
            return None
        story = self._stories.get(key)
        if story is not None:
            self._log.record(key)
        return story
