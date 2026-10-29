"""라디오 — 레이가 준비된 대본을 대사 하나씩 읽는 1인 방송.

대본: ``assets/radio_scripts/<이름>.md`` 하나가 소재 하나. frontmatter 에 ``song``(카탈로그 키)과 ``topic``, 본문은 한
줄에 대사 하나(1~2문장)이고 마지막 대사가 곡 소개다. 작성 규칙은 :func:`check_script`. 고르기는 한 적 없는 것 중
무작위, 기록은 SQLite ``radio_script_plays`` (세션을 넘어 중복을 피한다).

진행(:class:`Tracker`): 대사를 ``thinking.append`` 로 하나씩 앞서 넣는다. 모델은 다음 대사가 와 있으면 이어 읽고,
다 말했는데 없으면 지어낸다.
- 다음 대사는 지금 대사의 마지막 문장에서 남은 글자가 :attr:`Tracker.PUSH_REMAIN` 이하일 때 넣는다. 더 일찍 넣으면
  읽지 않은 대사가 둘인 채 앞 대사를 건너뛰기도 한다.
- 어디까지 읽었는지는 출력 전사와 대본의 글자 일치로만 따라간다. 일치가 끊기면(바꿔 말함·대답) 앞서 넣지 않고,
  :attr:`Tracker.STALL_SEC` 조용해졌을 때 확실히 안 말한 것만 다시 넣거나 다음 대사로 넘어간다.
- 곡 소개 대사를 다 읽고 조용하면 곡을 튼다.

끼어들기(:class:`Radio`): 사용자 전사가 오면 넣기를 보류하고 판정기(:class:`RadioJudge`)에 on / paused 를 묻는다.
paused 는 오는 즉시(보류가 풀린 뒤에 와도), on 은 그 뒤 새 말도 새 반응도 없을 때 적용한다. 판정이 늦으면 넣을 때가
된 뒤 :attr:`Radio.DUE_WAIT_SEC` 더 기다리고 on 으로 진행한다.

멈춘 뒤(:class:`RadioResumeJudge`): 대화가 :attr:`Radio.RESUME_QUIET_SEC` 잠잠해지면 복귀 판정기에 묻고, go_back 이면
아직 안 읽힌 대사부터 다시 넣는다. stay_paused 거나 멈춘 동안 사용자가 곡을 틀었으면 혼자 돌아가지 않고, 사용자가
이어 달라고 할 때 백엔드가 ``resume_radio`` 를 부른다.

이 모듈은 세션 I/O 를 하지 않는다. 루프가 시각·전사·무음을 넣고, 돌려받은 할 일(:class:`Think`, :class:`UpdateBackend`,
:class:`PlaySong`, :class:`Record`)을 실행한다.
"""

from __future__ import annotations

import difflib
import json
import logging
import random
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from voice_pipeline.engines.gpt_live.instructions import (
    RADIO_AFTER_SONG_OPENING,
    RADIO_AUTO_RESUME_OPENING,
    RADIO_ENDED_NOTE,
    RADIO_JUDGE_FORMAT,
    RADIO_JUDGE_INSTRUCTIONS,
    RADIO_LATE_LINE_NOTE,
    RADIO_LINE_TEMPLATE,
    RADIO_ON_NOTE,
    RADIO_PAUSED_NOTE,
    RADIO_RESUME_JUDGE_INSTRUCTIONS,
    RADIO_RESUME_JUDGE_SECOND,
    build_radio_judge_prompt,
    build_radio_resume_prompt,
    radio_resume_judge_format,
)
from voice_pipeline.engines.gpt_live.songs import Song
from voice_pipeline.memory.storage import RADIO_PAUSED, RADIO_RESUME, RADIO_SONG, RADIO_START
from voice_pipeline.settings import DEFAULT_DB_PATH, RADIO_SCRIPTS_DIR
from voice_pipeline.trace import record_call
from voice_pipeline.types import ILLM

logger = logging.getLogger("voice_pipeline.radio")

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SENTENCE_END_RE = re.compile(r"(?<=[.?!])\s+")
_NON_LETTER_RE = re.compile(r"[\W_]+")


def letters(text: str) -> str:
    """공백·문장부호를 뺀 글자 — 전사와 대본은 이것끼리 맞춘다."""
    return _NON_LETTER_RE.sub("", text)


def sentences(line: str) -> list[str]:
    return [s for s in _SENTENCE_END_RE.split(line.strip()) if s]


# ---------------------------------------------------------------------------
# 대본
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadioScript:
    """대본 하나. ``key`` 는 파일 stem, ``lines`` 의 마지막이 곡 소개."""

    key: str
    topic: str
    song: Song
    lines: tuple[str, ...]


def check_script(lines: Sequence[str]) -> list[str]:
    """추적이 기대는 작성 규칙을 어긴 곳 (빈 목록이면 통과).

    - 대사는 1~2문장 — 끼어들면 모델이 읽던 대사의 나머지를 말할 수 있다.
    - 대사의 마지막 문장은 :attr:`Tracker.PUSH_REMAIN` 자보다 길게(곡 소개 대사는 빼고) — 다음 대사를 넣을 여유.
    - 앞뒤 두 대사 안에 :attr:`Tracker.TRACK_BLOCK` 자 넘는 같은 구절 없음 — 추적이 엉뚱한 자리와 일치한다.
    - 숫자·로마자 없음 — 전사 표기가 갈린다.
    """
    problems = []
    for i, line in enumerate(lines):
        n = i + 1
        count = len(sentences(line))
        if not 1 <= count <= 2:
            problems.append(f"line {n}: {count} sentences")
        if i < len(lines) - 1 and len(letters(sentences(line)[-1])) <= Tracker.PUSH_REMAIN:
            problems.append(f"line {n}: last sentence is {Tracker.PUSH_REMAIN} letters or shorter")
        if re.search(r"[0-9A-Za-z]", line):
            problems.append(f"line {n}: digits or Latin letters")
        for j in range(i + 1, min(i + 3, len(lines))):
            m = difflib.SequenceMatcher(None, letters(line), letters(lines[j]), autojunk=False)
            block = m.find_longest_match(0, len(letters(line)), 0, len(letters(lines[j])))
            if block.size >= Tracker.TRACK_BLOCK:
                phrase = letters(line)[block.a : block.a + block.size]
                problems.append(f"lines {n} and {j + 1}: repeat {phrase!r}")
    return problems


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


def load_scripts(
    scripts_dir: str | Path = RADIO_SCRIPTS_DIR, catalog: Mapping[str, Song] | None = None
) -> dict[str, RadioScript]:
    """폴더의 ``*.md`` 를 전부 읽는다. 곡이 카탈로그에 없으면(재생 불가) 경고하고 뺀다. 작성 규칙 위반은 경고만.

    Args:
        scripts_dir: 대본 폴더. 없으면 빈 dict.
        catalog: 재생 가능한 곡. 비어 있으면 모든 대본이 빠진다.

    Raises:
        RuntimeError: 파일을 읽을 수 없거나 frontmatter 형식이 틀렸을 때.
    """
    root = Path(scripts_dir)
    if not root.is_dir():
        logger.info("Radio scripts dir not found (%s) — radio tools disabled", root)
        return {}
    catalog = catalog or {}
    scripts: dict[str, RadioScript] = {}
    for path in sorted(root.glob("*.md")):
        try:
            fields, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"radio script load failed ({path}): {exc}") from exc
        song_key, topic = fields.get("song", ""), fields.get("topic", "")
        lines = tuple(line.strip() for line in body.splitlines() if line.strip())
        if not song_key or not topic or not lines:
            raise RuntimeError(f"radio script load failed ({path}): needs 'song', 'topic' and lines")
        song = catalog.get(song_key)
        if song is None:
            logger.warning("Radio script %r skipped — song %r not playable", path.stem, song_key)
            continue
        for problem in check_script(lines):
            logger.warning("Radio script %r: %s", path.stem, problem)
        scripts[path.stem] = RadioScript(key=path.stem, topic=topic, song=song, lines=lines)
    logger.info("Radio scripts: %d loaded", len(scripts))
    return scripts


class ScriptLog:
    """대본을 시작한 기록 (SQLite ``radio_script_plays``). 세션을 넘어 중복을 피하는 근거."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS radio_script_plays ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, script_key TEXT NOT NULL, played_at TEXT NOT NULL)"
            )
            self._conn.commit()

    def record(self, key: str) -> None:
        """시작을 기록한다 (끝까지 못 가도 한 것으로 친다)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO radio_script_plays (script_key, played_at) VALUES (?, ?)",
                (key, datetime.now(UTC).isoformat()),
            )
            self._conn.commit()

    def played_keys(self) -> list[str]:
        """한 적 있는 키 — 마지막으로 한 시각이 오래된 순."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT script_key, MAX(played_at) AS last FROM radio_script_plays"
                " GROUP BY script_key ORDER BY last ASC"
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class ScriptPicker:
    """다음 대본을 고르고 기록한다. 프로세스 수명. 세션마다 :meth:`new_session` 으로 세션 내 기록을 비운다."""

    def __init__(self, scripts: Mapping[str, RadioScript], log: ScriptLog, *, rng: random.Random | None = None) -> None:
        self._scripts = dict(scripts)
        self._log = log
        self._rng = rng or random.Random()
        self._session_used: set[str] = set()

    @property
    def scripts(self) -> Mapping[str, RadioScript]:
        return self._scripts

    def new_session(self) -> None:
        """세션 시작 — 세션 내 사용 기록을 비운다."""
        self._session_used.clear()

    def pick(self) -> RadioScript | None:
        """다음 대본. 한 적 없는 것 중 무작위 → 전부 했으면 오래된 순. 대본이 없으면 None."""
        if not self._scripts:
            return None
        if self._session_used >= self._scripts.keys():
            self._session_used.clear()  # 세션 안에서 한 바퀴 돌았다 — 다시 처음부터
        played = self._log.played_keys()
        fresh = [k for k in self._scripts if k not in played and k not in self._session_used]
        if fresh:
            key = self._rng.choice(fresh)
        else:
            key = next(k for k in played if k in self._scripts and k not in self._session_used)
        self._session_used.add(key)
        self._log.record(key)
        return self._scripts[key]


# ---------------------------------------------------------------------------
# 진행 추적
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Push:
    """다음 대사(``index``)를 넣는다."""

    index: int
    text: str
    by: str  # text(일치) / stall(조용해짐) / start


@dataclass(frozen=True)
class Repush:
    """끊긴 문장부터 다시 넣는다 (대사마다 한 번)."""

    text: str


@dataclass(frozen=True)
class Closing:
    """곡 소개까지 읽었다 — 곡을 튼다."""

    by: str  # closing(곡 소개 끝) / quiet(끝을 못 찾고 조용함) / cap(상한)


Action = Push | Repush | Closing


class Tracker:
    """대본 하나를 읽는 진행. 출력 전사로 어디까지 읽었는지 따라가 다음 할 일(:meth:`next_action`)을 정한다.

    시각은 호출자의 단조 시계(초). 무음은 받은 무음 조각 길이로 받는다 — 조각이 오지 않는 네트워크 정지를 쉼으로
    보지 않게. 대사 위치는 넣은 대사를 이어 붙인 글자열 기준.
    """

    TRACK_BLOCK = 8  # 진행으로 칠 전사·대본 일치 최소 길이. 짧으면 이른 넣기가 는다
    DONE_SLACK = 4  # 대사 끝 이만큼 안까지 일치하면 끝난 것 (끝말 "요" 같은 차이)
    PUSH_REMAIN = 13  # 마지막 문장에서 남은 글자가 이만큼(약 2 s 분량) 이하면 다음 대사
    STALL_SEC = 1.5  # 넣은 뒤 말하고 이만큼 조용하면 멈춘 것 — 대사 안 쉼보다 길게
    CUT_LETTERS = 8  # 멈췄을 때 문장 끝까지 이만큼 이상 남은 문장만 다시 넣는다 — 어미 몇 자는 넘어간다
    REPUSH_WAIT_SEC = 3.0  # 다시 넣은 뒤 이만큼 지나도 조용하면 다음 대사(곡 소개면 곡)
    CLOSING_QUIET_SEC = 0.5  # 곡 소개를 끝까지 읽고 이만큼 조용하면 곡 — 곡을 틀며 스트림을 닫으니 덧붙임은 잘린다
    CLOSING_FALLBACK_SEC = 4.0  # 마지막 대사까지 넣었는데 끝을 못 찾으면, 말한 뒤 이만큼 조용할 때 곡
    CLOSING_CAP_SEC = 60.0  # 마지막 대사를 넣고 이만큼 지나면 그냥 곡 (침묵·추적 실패 안전망)

    def __init__(self, lines: Sequence[str]) -> None:
        if not lines:
            raise ValueError("a script needs at least one line")
        self.lines = tuple(lines)
        self._letters = [letters(t) for t in self.lines]
        self._offsets = [0]
        for x in self._letters:
            self._offsets.append(self._offsets[-1] + len(x))
        last = [letters(sentences(t)[-1]) for t in self.lines]
        self._last_start = [self._offsets[i + 1] - len(s) for i, s in enumerate(last)]
        self._out: list[tuple[float, str]] = []  # (도착 시각, 출력 전사 조각)
        self._pushes: list[float] = []  # 넣은 대사마다 넣은 시각 (재개 때 읽히지 않은 것은 지운다)
        self._done = 0  # 끝까지 읽힌 대사 수 (앞에서부터, 건너뛴 대사 포함)
        self._track_from = 0.0  # 진행 계산에 쓰는 전사의 시작 시각 — 재개 때 옮긴다
        self._track_unit = 0  # 진행 계산에 쓰는 첫 대사
        self._holds: list[list[float]] = []  # 보류 구간 [시작, 끝]. 이 동안의 말은 "말한 글자"로 세지 않는다
        self._repushed = -1  # 다시 넣은 대사
        self._repushed_at = 0.0
        self._first_voice: float | None = None  # 마지막으로 넣은 뒤 첫 모델 음성
        self._done_at: list[float] = []  # 대사마다 끝난 것으로 센 시각
        # (키, (reach, said_after), 보류 구간 [(시작, 레이가 대본으로 돌아온 시각)])
        self._cache: tuple[tuple[object, ...], tuple[int, int], list[tuple[float, float]]] | None = None

    # -- 입력 ----------------------------------------------------------------

    def add_output(self, t: float, delta: str) -> None:
        """모델 출력 전사 조각. 끝난 대사는 조각마다 센다 — 보류·멈춤 중에도(재개 지점이 된다)."""
        self._out.append((t, delta))
        self._update_done()

    def add_voice(self, t: float) -> None:
        """모델 말소리 조각이 왔다."""
        if self._pushes and self._first_voice is None and t >= self._pushes[-1]:
            self._first_voice = t

    def hold(self, t: float) -> None:
        """사용자 말을 받기 시작했다 — 넣기를 멈추고 그동안의 모델 말은 대사를 읽은 것으로 세지 않는다."""
        if not self._holds or self._holds[-1][1] != float("inf"):
            self._holds.append([t, float("inf")])

    def release(self, t: float) -> None:
        """보류가 풀렸다. 그 뒤 모델 말(맞장구에 대한 대답)도 대본과 다시 일치할 때까지 세지 않는다."""
        if self._holds and self._holds[-1][1] == float("inf"):
            self._holds[-1][1] = t

    def mark_pushed(self, t: float) -> None:
        self._pushes.append(t)
        self._first_voice = None

    def mark_repushed(self, t: float) -> None:
        self._repushed, self._repushed_at = len(self._pushes) - 1, t

    def resume(self, t: float) -> int | None:
        """재개 — 넣었지만 읽히지 않은 대사를 지우고 그 대사부터 새로 따라간다.

        Returns:
            다시 넣을 대사 번호. 다 읽었으면 None.
        """
        del self._pushes[self._done :]
        self._track_from, self._track_unit = t, self._done
        self._first_voice = None
        self._cache = None
        return self._done if self._done < len(self.lines) else None

    # -- 상태 ----------------------------------------------------------------

    @property
    def pushed(self) -> int:
        return len(self._pushes)

    @property
    def last_pushed_at(self) -> float | None:
        """마지막으로 대사를 넣은 시각."""
        return self._pushes[-1] if self._pushes else None

    @property
    def done(self) -> int:
        self._update_done()
        return self._done

    def reading(self) -> tuple[int, float]:
        """(읽던 대사 번호, 그 대사를 읽기 시작한 무렵).

        넣은 대사 중 끝나지 않은 첫 대사다. 다만 앞서 넣어 두기만 하고 아직 들어서지 않았으면(대사 사이) 방금 끝난 앞
        대사다. 시작은 앞 대사가 끝난 때와 이 대사를 넣은 때 중 늦은 쪽 — 첫 대사·재개·곡 뒤는 넣은 때가 된다.
        """
        i = min(self.done, max(self.pushed - 1, 0))
        if 0 < i < self.pushed and i <= len(self._done_at) and self._pushes[i] < self._done_at[i - 1]:
            if self._progress()[0] <= self._offsets[i]:
                i -= 1
        start = self._pushes[i] if i < self.pushed else 0.0
        if 0 < i <= len(self._done_at):
            start = max(start, self._done_at[i - 1])
        return i, start

    def interruptions(self) -> list[tuple[float, float]]:
        """보류마다 (시작, 레이가 대본으로 돌아온 시각) — 사용자 말과 레이 반응이 오간 때. 안 돌아왔으면 끝이 inf."""
        if not self._pushes:
            return []
        self._progress()
        assert self._cache is not None
        return list(self._cache[2])

    def off_script_letters(self, since: float) -> int:
        """``since`` 뒤 모델 말 가운데 읽던 대사·넣어 둔 대사와 맞지 않는 글자 수 — 끼어든 뒤의 반응."""
        text = letters("".join(d for t, d in self._out if t >= since))
        script = "".join(self._letters[max(0, self.done - 1) : self.pushed])
        matched = sum(
            b.size
            for b in difflib.SequenceMatcher(None, text, script, autojunk=False).get_matching_blocks()
            if b.size >= self.TRACK_BLOCK
        )
        return len(text) - matched

    # -- 판단 ----------------------------------------------------------------

    def next_action(self, t: float, quiet: float) -> Action | None:
        """지금 할 일. 보류 중이거나 넣은 대사가 없으면 부르지 않는다.

        Args:
            t: 지금 시각.
            quiet: 마지막 모델 음성 뒤 받은 무음 조각 길이(초).
        """
        if not self._pushes:
            return None
        self._update_done()
        last = len(self._pushes) - 1
        reach, said_after = self._progress()
        stalled = self._first_voice is not None and quiet >= self.STALL_SEC
        if len(self._pushes) >= len(self.lines):  # 곡 소개까지 넣었다
            if self._done >= len(self.lines) and quiet >= self.CLOSING_QUIET_SEC:
                return Closing("closing")
            unread = self._unread(reach, said_after) if stalled else []
            if unread and self._repushed != last:
                return Repush(" ".join(unread))
            waited = self._repushed != last or t - self._repushed_at >= self.REPUSH_WAIT_SEC
            if self._first_voice is not None and quiet >= self.CLOSING_FALLBACK_SEC and waited:
                return Closing("quiet")
            if t - self._pushes[-1] >= self.CLOSING_CAP_SEC:
                return Closing("cap")
            return None
        nxt = len(self._pushes)
        if reach > self._last_start[last] and self._offsets[last + 1] - reach <= self.PUSH_REMAIN:
            return Push(nxt, self.lines[nxt], "text")
        if not stalled:
            return None
        unread = self._unread(reach, said_after)
        if unread and self._repushed != last:
            return Repush(" ".join(unread))
        if unread and t - self._repushed_at < self.REPUSH_WAIT_SEC:
            return None
        return Push(nxt, self.lines[nxt], "stall")

    def _unread(self, reach: int, said_after: int) -> list[str]:
        """멈췄을 때 다시 넣을 문장 — 확실히 안 말한 것만. 빈 목록이면 다음으로 넘어간다.

        마지막 일치 뒤로 대사 끝까지 :attr:`CUT_LETTERS` 자 이상 남았고, 그 뒤 말한 글자가 남은 양의 절반 미만일
        때(바꿔 말하며 다 읽은 게 아닐 때) 마지막으로 넣은 대사와 그 앞 대사에서 끝까지 그만큼 남은 문장들. 끊긴
        문장은 처음부터.
        """
        last = len(self._pushes) - 1
        remain = self._offsets[last + 1] - reach
        if remain < self.CUT_LETTERS or 2 * said_after >= remain:
            return []
        out = []
        for u in range(max(0, last - 1), last + 1):
            end = self._offsets[u]
            for s in sentences(self.lines[u]):
                end += len(letters(s))
                if end - reach >= self.CUT_LETTERS:
                    out.append(s)
        return out

    def _update_done(self) -> None:
        if not self._pushes:
            return
        reach, _ = self._progress()
        while self._done < len(self._pushes) and reach >= self._offsets[self._done + 1] - self.DONE_SLACK:
            self._done += 1
            self._done_at.append(self._out[-1][0] if self._out else 0.0)

    def _progress(self) -> tuple[int, int]:
        """(reach, said_after) — 일치한 가장 먼 대본 글자 위치와, 그 일치 뒤 말한 글자 수(보류·대답 빼고).

        일치는 끝나지 않은 첫 대사의 앞 대사부터 넣은 대사까지와, 그 앞 대사를 넣은 뒤(재개 뒤면 재개 뒤) 전사로 본다.
        아직 아무 대사와도 맞지 않은 말(시작·재개 때의 "네, 시작할게요")은 말한 글자로 치지 않는다.
        """
        key = (len(self._out), len(self._pushes), self._done, self._track_from, str(self._holds))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        base = min(max(self._track_unit, self._done - 1), len(self._pushes) - 1)
        since = max(self._track_from, self._pushes[base])
        raw, at = "", []
        for t, d in self._out:
            if t >= since:
                raw += d
                at += [t] * len(d)
        idx = [i for i, ch in enumerate(raw) if letters(ch)]
        text = "".join(raw[i] for i in idx)
        script = "".join(self._letters[base : len(self._pushes)])
        blocks = [
            b
            for b in difflib.SequenceMatcher(None, text, script, autojunk=False).get_matching_blocks()
            if b.size >= self.TRACK_BLOCK
        ]
        reach = reach_t = 0
        if blocks:
            reach, reach_t = blocks[-1].b + blocks[-1].size, blocks[-1].a + blocks[-1].size
        # 보류가 풀린 뒤의 말은 대본과 다시 일치하는 곳(그때 읽던 일치가 이어지면 풀린 때)까지 뺀다
        spans = [(at[idx[b.a]], at[idx[b.a + b.size - 1]]) for b in blocks]
        excluded = []
        for a, e in self._holds:
            if e != float("inf") and not any(t0 <= e <= t1 for t0, t1 in spans):
                e = min((t0 for t0, _ in spans if t0 >= e), default=float("inf"))
            excluded.append((a, e))
        said_after = 0
        if reach_t:
            said_after = sum(1 for i in idx[reach_t:] if not any(a <= at[i] < e for a, e in excluded))
        result = (self._offsets[base] + reach, said_after)
        self._cache = (key, result, excluded)
        return result


# ---------------------------------------------------------------------------
# 판정기
# ---------------------------------------------------------------------------


class RadioJudge:
    """끼어든 말을 보고 라디오가 on / paused 중 어디여야 하는지 판정한다 (별도 LLM 호출, 구조화 출력).

    on = 짧은 반응·감탄이거나 레이가 흘리고 계속 읽음. paused = 그 밖에 레이에게 말을 건 모든 것 — 그만두자는 말도
    멈춤으로 받아 책갈피를 남긴다(잘못 들었어도 이어 달라면 되돌릴 수 있게).
    """

    def __init__(self, llm: ILLM) -> None:
        self._llm = llm

    def classify(self, line: str, exchange: Sequence[tuple[str, str]], earlier: Sequence[tuple[str, str]] = ()) -> str:
        """판정 하나. 실패하면 예외 — 호출자가 on 으로 본다. 걸린 시간은 로그와 호출 기록에 남긴다.

        Args:
            line: 레이가 읽던 대사.
            exchange: 끼어든 뒤의 (화자 user/assistant, 전사) 차례.
            earlier: 같은 대사 안에서 앞서 있었던 끼어들기의 차례.
        """
        start = time.monotonic()
        state = reason = ""
        try:
            stream = self._llm.generate(
                [
                    {"role": "system", "content": RADIO_JUDGE_INSTRUCTIONS},
                    {"role": "user", "content": build_radio_judge_prompt(line, exchange, earlier)},
                ],
                tools=[],
                response_format=RADIO_JUDGE_FORMAT,
            )
            out = json.loads("".join(stream))
            state, reason = str(out.get("state", "")), str(out.get("reason", ""))
            if state not in ("on", "paused"):
                raise RuntimeError(f"radio judge: unexpected state {state!r}")
        except Exception:
            _record_judge(self._llm, "judge", start, "error", {"state": state, "reason": reason})
            raise
        elapsed = _record_judge(self._llm, "judge", start, "ok", {"state": state, "reason": reason})
        logger.info("Radio judge: %s %.2fs (%s)", state, elapsed, reason)
        return state


class RadioResumeJudge:
    """멈춘 대화가 잠잠해졌을 때 레이가 라디오로 스스로 돌아갈지 판정한다 (별도 LLM 호출, 구조화 출력).

    go_back = 기본. wait_for_answer = 레이가 마지막에 묻거나 제안했는데 답이 없다. stay_paused = 사용자가 라디오를
    끝내 달라고 했거나(낭독을 멈추게 한 말, 대화 중엔 라디오를 가리킨 말만), 레이에게 한 마지막 말이 기다려 달라는 말.
    """

    def __init__(self, llm: ILLM) -> None:
        self._llm = llm

    def decide(
        self, line: str, interruption: str, conversation: Sequence[tuple[str, str]], *, second: bool = False
    ) -> str:
        """판정 하나. 실패하면 예외. 걸린 시간은 로그와 호출 기록에 남긴다.

        Args:
            line: 멈출 때 레이가 읽던 대사.
            interruption: 낭독을 멈추게 한 사용자 말.
            conversation: 멈춘 뒤의 (화자 user/assistant, 전사) 차례.
            second: wait_for_answer 뒤 다시 묻는다 — go_back / stay_paused 만.
        """
        start = time.monotonic()
        decision = reason = ""
        system = RADIO_RESUME_JUDGE_INSTRUCTIONS + (RADIO_RESUME_JUDGE_SECOND if second else "")
        try:
            stream = self._llm.generate(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": build_radio_resume_prompt(line, interruption, conversation)},
                ],
                tools=[],
                response_format=radio_resume_judge_format(second=second),
            )
            out = json.loads("".join(stream))
            decision, reason = str(out.get("decision", "")), str(out.get("reason", ""))
            allowed = ("go_back", "stay_paused") if second else ("go_back", "wait_for_answer", "stay_paused")
            if decision not in allowed:
                raise RuntimeError(f"radio resume judge: unexpected decision {decision!r}")
        except Exception:
            _record_judge(self._llm, "resume_judge", start, "error", {"decision": decision, "reason": reason})
            raise
        elapsed = _record_judge(self._llm, "resume_judge", start, "ok", {"decision": decision, "reason": reason})
        logger.info("Radio resume judge: %s %.2fs (%s)", decision, elapsed, reason)
        return decision


def _record_judge(llm: ILLM, operation: str, start: float, status: str, fields: Mapping[str, str]) -> float:
    """판정 호출 기록 한 건. 걸린 시간(초)을 돌려준다."""
    elapsed = time.monotonic() - start
    model = getattr(llm, "model", "unknown")
    metadata = json.dumps(dict(fields), ensure_ascii=False)
    record_call("radio", operation, model, elapsed * 1000.0, status=status, metadata=metadata)
    return elapsed


# ---------------------------------------------------------------------------
# 라디오 상태
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Think:
    """대화 모델에 ``thinking.append``."""

    text: str


@dataclass(frozen=True)
class UpdateBackend:
    """백엔드 지시문을 지금 라디오 상태로 바꾼다."""


@dataclass(frozen=True)
class PlaySong:
    """대본의 곡을 튼다."""

    song: Song


@dataclass(frozen=True)
class Record:
    """사건 줄을 utterances 에 남긴다 — 상태가 바뀐 자리를 말 사이에 표시한다."""

    event: str


Command = Think | UpdateBackend | PlaySong | Record


@dataclass
class _JudgeCall:
    future: Future[str]
    sent_at: float
    upto_user: float  # 이 판정이 본 마지막 사용자 전사 시각
    upto_reaction: int  # 이 판정이 본 반응 글자 수
    hold_start: float  # 이 판정이 속한 끼어들기(보류)


@dataclass
class _Hold:
    """사용자 말 하나를 받는 동안의 판정 상태."""

    start: float
    judged_user: float = -1.0
    judged_reaction: int = 0
    latest_on: _JudgeCall | None = None
    last_sent: float | None = None  # 마지막 판정 호출 시각
    due_since: float | None = None  # 판정 없이 넣어야 할 순간(넣기·무음 폴백·곡)이 온 시각


@dataclass
class _ResumeCheck:
    """멈춘 동안 라디오로 돌아갈지 묻는 판정 하나."""

    future: Future[str]
    lull: float  # 이 판정이 속한 잠잠함이 시작된 시각 — 그 뒤 누가 말하면 버린다
    second: bool  # wait_for_answer 뒤 다시 묻는 것
    attempt: int


@dataclass
class _Transcripts:
    """전사 스냅샷을 도착 시각이 붙은 조각으로 바꾼다."""

    texts: dict[str, str] = field(default_factory=dict)
    deltas: list[tuple[float, str, str]] = field(default_factory=list)  # (도착 시각, 화자, 조각)

    def add(self, t: float, segment_id: str, speaker: str, text: str) -> str:
        old = self.texts.get(segment_id, "")
        delta = text[len(old) :] if text.startswith(old) else text
        self.texts[segment_id] = text
        if delta.strip():
            self.deltas.append((t, speaker, delta))
            return delta
        return ""


class Radio:
    """세션 하나의 라디오 — 시작·진행·끼어들기 판정·멈춤·재개.

    상태: off → (start_radio) on → (판정 paused) paused → (resume_radio 또는 잠잠해져 복귀 판정 go_back) on … /
    (사용자가 곡을 멈춤) off. 멈춘 동안 남은 대사가 책갈피다 — 재개하면 거기부터, 새로 시작하면 버린다. 곡은 루프가
    틀고 끝나면 :meth:`on_song_end` 로 알린다.
    """

    USER_QUIET_SEC = 0.6  # 사용자 전사가 이만큼 멈추면 판정한다
    RECHECK_LETTERS = 6  # 판정 뒤 모델이 대본 밖 말을 이만큼 더 하면 다시 판정한다
    DUE_WAIT_SEC = 3.0  # 넣을 때가 됐는데 판정이 없으면 이만큼 기다린다 (그때와 판정 호출 중 늦은 쪽부터)
    JUDGE_LATE_MAX_SEC = 8.0  # 호출 뒤 이보다 늦은 판정은 버린다 — 기다림이 호출보다 늦게 시작할 수 있어 넉넉히
    HOLD_MAX_QUIET_SEC = 10.0  # 판정을 기다리지 않는데 모델이 이만큼 조용하면 보류를 푼다 (안전망)
    TRANSCRIPT_LAG_SEC = 1.3  # 사용자 전사는 목소리 시작보다 이만큼 늦게 온다 — 판정 입력은 그때부터의 대화
    LATE_PUSH_SEC = 1.5  # 보류 직전 이 안에 넣은 대사는 사용자가 말하는 도중에 들어갔다 (전사 지연 + 여유)
    PRIOR_MAX = 2  # 판정 입력에 넣는 같은 대사 안 앞 끼어들기 수 (최근 것부터)
    PRIOR_MAX_SEC = 30.0  # 그중 시작한 지 이 안의 것만 — 대사가 비정상적으로 길어져도 한참 전 일은 넣지 않는다
    RESUME_QUIET_SEC = 8.0  # 멈춘 대화가 이만큼 잠잠하면 복귀 판정을 적용한다 (실기에서 조절)
    RESUME_JUDGE_LEAD_SEC = 1.5  # 복귀 판정은 그보다 이만큼 먼저 부른다 — 판정 시간을 숨긴다
    RESUME_LATER_SEC = 20.0  # wait_for_answer 면 잠잠함이 이만큼 됐을 때 다시 판정한다
    RESUME_ATTEMPTS = 3  # 잠잠함 하나에서 복귀 판정 시도 수 — 실패하면 계속 조용할 때 바로 다시

    def __init__(
        self,
        picker: ScriptPicker,
        judge: RadioJudge,
        executor: Executor | None = None,
        resume_judge: RadioResumeJudge | None = None,
    ) -> None:
        """``resume_judge`` 가 없으면 멈춘 뒤 혼자 돌아가지 않는다(이어 달라는 요청으로만)."""
        self._picker = picker
        self._judge = judge
        self._executor = executor
        self._resume_judge = resume_judge
        self.status = "off"
        self.script: RadioScript | None = None
        self.tracker: Tracker | None = None
        self.playing_song = False  # 대본의 곡을 트는 중 (루프의 노래 흐름)
        self._transcripts = _Transcripts()
        self._hold: _Hold | None = None
        self._last_user_text: float | None = None
        self._calls: list[_JudgeCall] = []
        self._last_hold_start: float | None = None  # 가장 최근 끼어들기 — 판정은 이것에 대한 것만 적용한다
        self._notes: list[Command] = []  # 다음 step 에 낼 문구
        # 멈춘 동안 — 복귀 판정의 입력과 잠잠함
        self.auto_resume = False  # 잠잠해지면 복귀 판정을 묻는가
        self._paused_at = 0.0
        self._pause_line = ""  # 멈출 때 읽던 대사
        self._pause_hold = 0.0  # 멈추게 한 끼어들기가 시작된 시각
        self._last_voice: float | None = None  # 마지막 모델 말소리
        self._last_busy: float | None = None  # 백엔드 위임이 마지막으로 진행 중이던 시각
        self._resume_check: _ResumeCheck | None = None
        self._later_lull: float | None = None  # wait_for_answer 를 받은 잠잠함
        self._given_up_lull: float | None = None  # 판정이 거듭 실패해 더 묻지 않는 잠잠함

    # -- 툴 ------------------------------------------------------------------

    def start(self, now: float) -> tuple[str, list[Command]]:
        """``start_radio`` — 새 대본으로 시작한다(멈춘 대본은 버린다). (툴 결과, 할 일)."""
        if self.status == "on":
            return json.dumps({"error": "the radio talk is already on"}), []
        script = self._picker.pick()
        if script is None:
            return json.dumps({"error": "no radio scripts available"}), []
        self._reset_reading()
        self._clear_pause()
        self.status = "on"
        commands: list[Command] = [UpdateBackend(), Think(RADIO_ON_NOTE)]
        commands += self._begin_script(script, now, after_song=False)
        logger.info("Radio started: %s (song %s)", script.key, script.song.key)
        return json.dumps({"status": "on"}), commands

    def resume(self, now: float) -> tuple[str, list[Command]]:
        """``resume_radio`` — 멈춘 곳(아직 안 읽힌 첫 대사)부터 다시 넣는다. (툴 결과, 할 일)."""
        if self.status != "paused" or self.tracker is None:
            return json.dumps({"error": "the radio talk is not paused"}), []
        return json.dumps({"status": "on"}), self._resume_reading(now, auto=False)

    def _resume_reading(self, now: float, *, auto: bool) -> list[Command]:
        """멈춘 곳부터 다시 넣는다. ``auto`` 면 혼자 돌아가는 것 — 첫 대사 앞에 재개 문구를 붙인다."""
        assert self.tracker is not None
        self.status = "on"
        self._clear_pause()
        commands: list[Command] = [Record(RADIO_RESUME), UpdateBackend(), Think(RADIO_ON_NOTE)]
        index = self.tracker.resume(now)
        if index is not None:
            line = self.tracker.lines[index]
            text = f"{RADIO_AUTO_RESUME_OPENING} {line}" if auto else line
            commands += self._push(Push(index, text, "auto" if auto else "resume"), now)
        where = index + 1 if index is not None else "end"
        logger.info("Radio resumed%s at line %s", " on its own" if auto else "", where)
        return commands

    # -- 입력 ----------------------------------------------------------------

    def on_transcript(self, now: float, segment_id: str, speaker: str, text: str) -> None:
        """전사 세그먼트 스냅샷 (닫히지 않은 것 포함).

        구간별 글자는 라디오가 꺼져 있거나 곡 중에도 기록한다 — 시작 요청처럼 그 전에 시작된 구간의 마무리 갱신이 늦게
        와도 새 말로 보지 않게.
        """
        delta = self._transcripts.add(now, segment_id, speaker, text)
        if not delta or self.status == "off" or self.playing_song:
            return
        if speaker == "user":
            if not letters(delta):
                return  # 마무리 갱신에 붙은 문장부호 — 새 말이 아니다
            self._last_user_text = now
            if self.status == "on" and self._hold is None:
                self._start_hold(now)
        elif self.status == "on" and self.tracker is not None:
            self.tracker.add_output(now, delta)  # 멈춘 동안의 대화는 대사를 읽은 것이 아니다 — 재개 지점을 지킨다

    def on_voice(self, now: float) -> None:
        """모델 말소리 조각이 왔다."""
        self._last_voice = now
        if self.tracker is not None:
            self.tracker.add_voice(now)

    def on_user_song(self) -> None:
        """사용자가 곡을 틀었다(대본의 곡이 아니다). 멈춘 동안이면 혼자 돌아가지 않는다 — 이어 달라면 재개."""
        if self.status == "paused" and self.auto_resume:
            self._clear_pause()
            logger.info("Radio: a song was played while paused — staying paused until the user asks")

    def on_song_end(self, now: float, *, completed: bool) -> list[Command]:
        """대본의 곡이 끝났다. 자연 종료면 다음 대본, 사용자가 멈췄으면 라디오를 끝낸다."""
        self.playing_song = False
        if self.status != "on":
            return []
        if not completed:
            logger.info("Radio: song stopped by the user — radio off")
            return self._end()
        script = self._picker.pick()
        if script is None:
            logger.warning("Radio: no next script — radio off")
            return self._end()
        self._reset_reading()
        logger.info("Radio: next script %s (song %s)", script.key, script.song.key)
        return self._begin_script(script, now, after_song=True)

    def reset(self) -> None:
        """세션이 끝난다 — 상태를 지운다 (백엔드·지시는 건드리지 않는다)."""
        self.status = "off"
        self._reset_reading()
        self._transcripts = _Transcripts()
        self.playing_song = False
        self._calls.clear()
        self._clear_pause()
        self._last_voice = self._last_busy = None

    # -- 진행 ----------------------------------------------------------------

    def step(self, now: float, quiet: float, *, backend_busy: bool = False) -> list[Command]:
        """프레임마다. 노래 중이거나 세션이 끝나는 중이면 부르지 않는다.

        Args:
            now: 지금 시각.
            quiet: 마지막 모델 음성 뒤 받은 무음 조각 길이(초).
            backend_busy: 백엔드 위임이 진행 중이다 — 결과를 기다리는 동안 멈춘 대화는 잠잠한 게 아니다.
        """
        commands, self._notes = self._notes, []
        commands += self._poll_judge(now)
        if backend_busy:
            self._last_busy = now
        if self.status == "paused":
            return commands + self._paused_step(now)
        if self.status != "on" or self.tracker is None or self.playing_song:
            return commands
        if self._hold is not None:
            self._hold_step(now, quiet)
            if self._hold is not None and not self._due_while_holding(now, quiet):
                return commands
        action = self.tracker.next_action(now, quiet)
        if isinstance(action, Push):
            commands += self._push(action, now)
        elif isinstance(action, Repush):
            self.tracker.mark_repushed(now)
            logger.info("Radio: quiet after speaking — pushing again: %s", action.text)
            commands.append(Think(RADIO_LINE_TEMPLATE.format(action.text)))
        elif isinstance(action, Closing):
            assert self.script is not None
            self.playing_song = True
            logger.info("Radio: song introduced (%s) — playing %s", action.by, self.script.song.key)
            commands += [Record(f"{RADIO_SONG} {self.script.song.key}"), PlaySong(self.script.song)]
        return commands

    def _begin_script(self, script: RadioScript, now: float, *, after_song: bool) -> list[Command]:
        lines = list(script.lines)
        if after_song:
            lines[0] = f"{RADIO_AFTER_SONG_OPENING} {lines[0]}"
        self.script = script
        self.tracker = Tracker(lines)
        return [Record(f"{RADIO_START} {script.key}"), *self._push(Push(0, lines[0], "start"), now)]

    def _push(self, push: Push, now: float) -> list[Command]:
        assert self.tracker is not None
        self.tracker.mark_pushed(now)
        logger.info("Radio: line %d/%d pushed (%s)", push.index + 1, len(self.tracker.lines), push.by)
        return [Think(RADIO_LINE_TEMPLATE.format(push.text))]

    def _reset_reading(self) -> None:
        """대본 하나의 진행을 지운다. 전사 기록은 세션 동안 남긴다 — 판정 입력은 시각으로 자른다."""
        self.script = None
        self.tracker = None
        self._hold = None
        self._last_user_text = None
        self._notes.clear()

    def _end(self) -> list[Command]:
        self.status = "off"
        self._reset_reading()
        self._clear_pause()
        return [UpdateBackend(), Think(RADIO_ENDED_NOTE)]

    # -- 멈춘 뒤 복귀 ------------------------------------------------------------

    def _paused_step(self, now: float) -> list[Command]:
        """멈춘 대화가 잠잠해지면 복귀 판정을 묻고 적용한다.

        잠잠함은 멈춘 뒤 마지막 사용자 전사·모델 말소리·백엔드 위임 중 늦은 것부터 센다. 판정은
        :attr:`RESUME_QUIET_SEC` 보다 :attr:`RESUME_JUDGE_LEAD_SEC` 먼저 부르고 그 시점이 되면 적용한다 — 그 사이 누가
        말하면 버리고 다음 잠잠함에 다시 묻는다. 판정이 실패하면 조용한 동안 바로 다시 묻는다(잠잠함마다
        :attr:`RESUME_ATTEMPTS` 번까지).
        """
        if not self.auto_resume or self._resume_judge is None or self.tracker is None:
            return []
        marks = (self._paused_at, self._last_user_text, self._last_voice, self._last_busy)
        lull = max(t for t in marks if t is not None)
        check = self._resume_check
        if check is not None and check.lull != lull:
            self._resume_check = check = None
            logger.info("Radio: new speech while asking whether to go back — judgment dropped")
        if self._given_up_lull == lull:
            return []
        later = self._later_lull == lull
        due = self.RESUME_LATER_SEC if later else self.RESUME_QUIET_SEC
        if check is None:
            if now - lull >= due - self.RESUME_JUDGE_LEAD_SEC:
                self._ask_resume(now, lull, second=later, attempt=1)
            return []
        if not check.future.done():
            return []
        if check.future.exception() is not None:
            self._resume_check = None
            if check.attempt < self.RESUME_ATTEMPTS:
                logger.warning("Radio resume judge failed — asking again", exc_info=check.future.exception())
                self._ask_resume(now, lull, second=check.second, attempt=check.attempt + 1)
            else:
                logger.warning("Radio resume judge failed %d times — not going back on its own now", check.attempt)
                self._given_up_lull = lull
            return []
        if now - lull < due:
            return []
        self._resume_check = None
        decision = check.future.result()
        if decision == "go_back":
            return self._resume_reading(now, auto=True)
        if decision == "wait_for_answer":
            self._later_lull = lull
            logger.info("Radio: Ray is waiting for an answer — asking again at %.0f s of quiet", self.RESUME_LATER_SEC)
            return []
        self._clear_pause()
        logger.info("Radio: staying paused until the user asks")
        return []

    def _ask_resume(self, now: float, lull: float, *, second: bool, attempt: int) -> None:
        assert self._resume_judge is not None
        judge = self._resume_judge
        line, interruption = self._pause_line, self._interruption()
        conversation = self._exchange(self._paused_at)
        rows = " | ".join(f"{'User' if who == 'user' else 'Ray'}: {text}" for who, text in conversation)
        logger.info(
            "Radio: quiet %.1f s while paused — asking whether to go back%s | reading «%s» | interruption «%s» | %s",
            now - lull,
            " (again)" if second else "",
            line,
            interruption,
            rows or "-",
        )
        future = self._submit(lambda: judge.decide(line, interruption, conversation, second=second))
        self._resume_check = _ResumeCheck(future, lull, second, attempt)

    def _interruption(self) -> str:
        """낭독을 멈추게 한 사용자 말 — 그 끼어들기의 사용자 전사(목소리 시작 무렵부터 멈출 때까지)."""
        turns = self._exchange(self._pause_hold - self.TRANSCRIPT_LAG_SEC, self._paused_at)
        return " ".join(text for who, text in turns if who == "user")

    def _clear_pause(self) -> None:
        """자동 복귀를 끄고 복귀 판정 상태를 지운다."""
        self.auto_resume = False
        self._resume_check = None
        self._later_lull = self._given_up_lull = None

    # -- 끼어들기 판정 --------------------------------------------------------

    def _start_hold(self, now: float) -> None:
        """사용자 말이 왔다 — 넣기를 멈춘다. 직전에 넣은 대사는 사용자 말 도중 들어갔으니 대답을 먼저 하게 한다."""
        self._hold = _Hold(start=now)
        self._last_hold_start = now
        logger.info("Radio: user spoke — holding lines")
        if self.tracker is None:
            return
        self.tracker.hold(now)
        pushed = self.tracker.last_pushed_at
        if pushed is not None and now - pushed <= self.LATE_PUSH_SEC:
            self._notes.append(Think(RADIO_LATE_LINE_NOTE))
            gap = now - pushed
            logger.info("Radio: line pushed %.1f s before the user's words arrived — asking to answer first", gap)

    def _hold_step(self, now: float, quiet: float) -> None:
        hold = self._hold
        assert hold is not None and self.tracker is not None
        self._maybe_judge(now, hold)
        on = hold.latest_on
        if on is not None and not self._waiting() and not self._stale(on, hold):
            self._release(now, "judge on")
        elif not self._waiting() and quiet >= self.HOLD_MAX_QUIET_SEC:
            self._release(now, "quiet")

    def _due_while_holding(self, now: float, quiet: float) -> bool:
        """판정을 기다리는 중 넣어야 할 순간(넣기·무음 폴백·곡)이 왔으면 판정을 :attr:`DUE_WAIT_SEC` 더 기다리고, 그래도
        없으면 보류를 푼다(on 으로 보고 진행한다). 풀었으면 True.

        보류 중에는 아무것도 넣지 않는다 — 사용자 말 도중 들어간 대사는 대답보다 먼저 읽히고, 조용한 레이는 사용자를
        기다리는 것일 수 있다. 기다림은 그 순간과 판정 호출 중 늦은 쪽부터 센다(사용자가 길게 말하면 판정을 부르기 전에
        무음이 먼저 찬다).
        """
        hold = self._hold
        assert hold is not None and self.tracker is not None
        if self.tracker.next_action(now, quiet) is None:
            hold.due_since = None
            return False
        if hold.due_since is None:
            hold.due_since = now
            logger.info("Radio: due to go on while judging — waiting up to %.0f s", self.DUE_WAIT_SEC)
        if hold.last_sent is None or now - max(hold.due_since, hold.last_sent) < self.DUE_WAIT_SEC:
            return False
        self._release(now, "no judgment")
        return True

    def _release(self, now: float, why: str) -> None:
        self._hold = None
        if self.tracker is not None:
            self.tracker.release(now)
        logger.info("Radio: hold released (%s) — talk goes on", why)

    def _waiting(self) -> bool:
        return any(not c.future.done() and c.hold_start == self._last_hold_start for c in self._calls)

    def _reaction(self, hold: _Hold) -> int:
        assert self.tracker is not None
        return self.tracker.off_script_letters(hold.start)

    def _stale(self, call: _JudgeCall, hold: _Hold) -> bool:
        new_user = self._last_user_text is not None and self._last_user_text > call.upto_user
        return new_user or self._reaction(hold) - call.upto_reaction >= self.RECHECK_LETTERS

    def _maybe_judge(self, now: float, hold: _Hold) -> None:
        if self._last_user_text is None or self._waiting():
            return
        new_user = self._last_user_text > hold.judged_user
        if new_user and now - self._last_user_text < self.USER_QUIET_SEC:
            return  # 사용자 전사가 아직 들어오는 중
        reaction = self._reaction(hold)
        if not new_user and reaction - hold.judged_reaction < self.RECHECK_LETTERS:
            return
        assert self.tracker is not None
        index, line_start = self.tracker.reading()
        line = self.tracker.lines[index]
        exchange = self._exchange(hold.start - self.TRANSCRIPT_LAG_SEC)
        earlier = self._earlier(hold, line_start, now)
        hold.judged_user, hold.judged_reaction = self._last_user_text, reaction

        def joined(turns: list[tuple[str, str]]) -> str:
            return " | ".join(f"{'User' if who == 'user' else 'Ray'}: {text}" for who, text in turns)

        before = f" | earlier: {joined(earlier)} ||" if earlier else ""
        why = "user" if new_user else "reaction"
        logger.info("Radio: judging (%s) — reading «%s»%s | %s", why, line, before, joined(exchange))
        call = _JudgeCall(
            self._submit(lambda: self._judge.classify(line, exchange, earlier)),
            now,
            self._last_user_text,
            reaction,
            hold.start,
        )
        hold.last_sent = now
        self._calls.append(call)

    def _submit(self, fn: Callable[[], str]) -> Future[str]:
        if self._executor is not None:
            return self._executor.submit(fn)
        future: Future[str] = Future()
        try:
            future.set_result(fn())
        except Exception as exc:  # noqa: BLE001 — 판정 실패는 on 으로 본다
            future.set_exception(exc)
        return future

    def _earlier(self, hold: _Hold, line_start: float, now: float) -> list[tuple[str, str]]:
        """같은 대사 안에서 앞서 있었던 끼어들기의 사용자 말·레이 반응 (최근 :attr:`PRIOR_MAX` 개, :attr:`PRIOR_MAX_SEC`
        안). 레이가 대본으로 돌아온 뒤의 낭독은 넣지 않는다 — 낭독이 길게 붙으면 판정기가 사용자 말을 이야기에 대한
        반응으로 읽는 쪽으로 기운다."""
        assert self.tracker is not None
        spans = [
            (a, e)
            for a, e in self.tracker.interruptions()
            if line_start <= a < hold.start and now - a <= self.PRIOR_MAX_SEC
        ][-self.PRIOR_MAX :]
        turns: list[tuple[str, str]] = []
        for k, (a, e) in enumerate(spans):
            nxt = spans[k + 1][0] if k + 1 < len(spans) else hold.start
            turns += self._exchange(a, min(e, nxt - self.TRANSCRIPT_LAG_SEC))
        return turns

    def _exchange(self, since: float, until: float = float("inf")) -> list[tuple[str, str]]:
        """``since`` 부터 ``until`` 전까지의 전사 조각을 화자가 바뀔 때마다 묶는다."""
        turns: list[list[str]] = []
        for t, who, d in self._transcripts.deltas:
            if t < since or t >= until:
                continue
            if turns and turns[-1][0] == who:
                turns[-1][1] += d
            else:
                turns.append([who, d])
        return [(who, text.strip()) for who, text in turns if text.strip()]

    def _poll_judge(self, now: float) -> list[Command]:
        commands: list[Command] = []
        for call in list(self._calls):
            age = now - call.sent_at
            if call.future.done():
                self._calls.remove(call)
                try:
                    state = call.future.result()
                except Exception:
                    logger.warning("Radio judge failed — treating as on", exc_info=True)
                    state = "on"
                if call.hold_start != self._last_hold_start:
                    logger.info("Radio: judgment for an earlier interruption — dropped")  # 사용자가 그 뒤에 또 말했다
                elif state == "paused":
                    commands += self._pause(now)  # 보류가 이미 풀렸어도(넣은 뒤) 적용한다 — 곡 중이면 버린다
                elif self._hold is not None:
                    self._hold.latest_on = call
            elif age >= self.JUDGE_LATE_MAX_SEC:
                self._calls.remove(call)  # 아무도 말하지 않는데 갑자기 멈추지 않게 — 나중에 끝나도 적용하지 않는다
                logger.info("Radio: no judgment in %.0f s — dropped", age)
        return commands

    def _pause(self, now: float) -> list[Command]:
        """paused 적용 — 라디오가 진행 중이면 언제나."""
        if self.status != "on" or self.playing_song:
            return []
        logger.info("Radio: paused by the judge")
        self._hold = None
        if self.tracker is not None:
            self._pause_line = self.tracker.lines[self.tracker.reading()[0]]
            self.tracker.release(now)
        self.status = "paused"
        self._clear_pause()
        self.auto_resume = self._resume_judge is not None
        self._paused_at = now
        self._pause_hold = self._last_hold_start if self._last_hold_start is not None else now
        return [Record(RADIO_PAUSED), UpdateBackend(), Think(RADIO_PAUSED_NOTE)]
