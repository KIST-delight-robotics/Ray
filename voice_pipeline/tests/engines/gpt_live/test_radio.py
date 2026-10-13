"""Tests for voice_pipeline.engines.gpt_live.radio — 대본 로드·검사, 고르기, 진행 추적, 라디오 상태."""

from __future__ import annotations

import json
import random
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voice_pipeline.engines.gpt_live.instructions import (
    RADIO_AFTER_SONG_OPENING,
    RADIO_ENDED_NOTE,
    RADIO_JUDGE_FORMAT,
    RADIO_LINE_TEMPLATE,
    RADIO_PAUSED_NOTE,
    RADIO_RESUMED_NOTE,
)
from voice_pipeline.engines.gpt_live.radio import (
    Closing,
    PlaySong,
    Push,
    Radio,
    RadioJudge,
    RadioScript,
    Repush,
    ScriptLog,
    ScriptPicker,
    Think,
    Tracker,
    UpdateBackend,
    check_script,
    letters,
    load_scripts,
)
from voice_pipeline.engines.gpt_live.songs import Song
from voice_pipeline.types import ILLM

SONGS = {"IAM": Song("IAM", "I AM", "IVE"), "snowman": Song("snowman", "눈사람", "정승환")}

LINES = (
    "오늘은 책상 서랍 이야기예요. 늘 곁에 있는데도 좀처럼 열어 보지 않는 곳이죠.",
    "서랍은 집에서 가장 느리게 바뀌는 자리예요. 책상 위는 매일 달라지는데, "
    "서랍 안은 몇 년이 지나도 그대로일 때가 많아요.",
    "그래서 오늘은 서랍에 넣어 둔 마음에 관한 노래를 골랐어요.",
    "아이브의 아이엠, 들려드릴게요.",
)


def _write(dir_: Path, name: str, song: str = "IAM", topic: str = "주제", body: str = "\n".join(LINES)) -> None:
    (dir_ / f"{name}.md").write_text(f"---\nsong: {song}\ntopic: {topic}\n---\n\n{body}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 대본
# ---------------------------------------------------------------------------


class TestLoadScripts:
    def test_loads_frontmatter_and_one_line_per_line(self, tmp_path: Path) -> None:
        _write(tmp_path, "a", body="\n" + LINES[0] + "\n\n" + LINES[3] + "\n")
        scripts = load_scripts(tmp_path, SONGS)
        assert list(scripts) == ["a"]
        script = scripts["a"]
        assert script.topic == "주제" and script.song is SONGS["IAM"] and script.lines == (LINES[0], LINES[3])

    def test_missing_dir_or_empty_catalog_gives_nothing(self, tmp_path: Path) -> None:
        assert load_scripts(tmp_path / "nope", SONGS) == {}
        _write(tmp_path, "a")
        assert load_scripts(tmp_path, {}) == {}

    def test_unplayable_song_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        _write(tmp_path, "a", song="Nope")
        _write(tmp_path, "b")
        assert list(load_scripts(tmp_path, SONGS)) == ["b"]

    @pytest.mark.parametrize(
        "text", ["no frontmatter", "---\nsong: IAM\n---\n\n줄\n", "---\nsong: IAM\ntopic: t\n---\n\n"]
    )
    def test_bad_file_raises(self, tmp_path: Path, text: str) -> None:
        (tmp_path / "bad.md").write_text(text, encoding="utf-8")
        with pytest.raises(RuntimeError, match="radio script load failed"):
            load_scripts(tmp_path, SONGS)

    def test_rule_violations_only_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write(tmp_path, "a", body="하나. 둘. 셋 문장이 있는 대사예요.\n아이브의 아이엠, 들려드릴게요.")
        assert list(load_scripts(tmp_path, SONGS)) == ["a"]
        assert "3 sentences" in caplog.text


class TestCheckScript:
    def test_clean_script_passes(self) -> None:
        assert check_script(LINES) == []

    def test_reports_each_rule(self) -> None:
        problems = check_script(
            [
                "첫째 문장이에요. 둘째 문장이에요. 셋째 문장이에요.",  # 세 문장
                "마지막 문장이 짧은 대사예요. 짧아요.",  # 마지막 문장 13자 이하
                "다 쓴 볼펜과 어디 건지 모를 열쇠가 나와요.",
                "숫자 3 이 든 대사도 있지요, 정말로 그래요.",
                "어디 건지 모를 열쇠는 더 이상한 물건이에요.",  # 두 대사 앞과 같은 구절
                "아이브의 아이엠, 들려드릴게요.",  # 마지막 대사는 짧아도 된다
            ]
        )
        assert problems == [
            "line 1: 3 sentences",
            "line 1: last sentence is 13 letters or shorter",
            "line 2: last sentence is 13 letters or shorter",
            "lines 3 and 5: repeat '어디건지모를열쇠'",
            "line 4: digits or Latin letters",
        ]


class TestScriptLog:
    def test_played_keys_oldest_first_by_last_play(self) -> None:
        log = ScriptLog(":memory:")
        assert log.played_keys() == []
        for key in ("a", "b", "a"):
            log.record(key)
        assert log.played_keys() == ["b", "a"]


def _scripts(*keys: str) -> dict[str, RadioScript]:
    return {k: RadioScript(k, f"topic {k}", SONGS["IAM"], LINES) for k in keys}


class TestScriptPicker:
    def test_picks_unplayed_at_random_then_oldest(self) -> None:
        log = ScriptLog(":memory:")
        log.record("a")
        picker = ScriptPicker(_scripts("a", "b", "c"), log, rng=random.Random(1))
        first, second = picker.pick(), picker.pick()
        assert first is not None and second is not None and {first.key, second.key} == {"b", "c"}
        third = picker.pick()
        assert third is not None and third.key == "a"
        assert log.played_keys()[-1] == "a"

    def test_session_wraps_around_and_new_session_keeps_log(self) -> None:
        log = ScriptLog(":memory:")
        picker = ScriptPicker(_scripts("a", "b"), log, rng=random.Random(0))
        first = picker.pick()
        picker.new_session()
        second = picker.pick()
        assert first is not None and second is not None and second.key != first.key
        assert ScriptPicker({}, log).pick() is None


# ---------------------------------------------------------------------------
# 진행 추적
# ---------------------------------------------------------------------------


class Model:
    """모델 흉내 — 말을 어절 조각으로 시각을 붙여 추적기에 넣고, 조각마다 추적기의 판단을 받는다."""

    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker
        self.t = 0.0
        self.silent_since = 0.0  # 마지막 말 뒤 — 조용한 시간은 다음 말까지 이어서 센다
        self.actions: list[tuple[float, object]] = []

    def say(self, text: str, rate: float = 6.5) -> None:
        for word in text.split(" "):
            self.tracker.add_voice(self.t)
            self.tracker.add_output(self.t, " " + word)
            self.t += max(0.1, len(letters(word)) / rate)
            if (action := self.tracker.next_action(self.t, 0.0)) is not None:
                self.actions.append((self.t, action))
                if isinstance(action, Push):
                    self.tracker.mark_pushed(self.t)
        self.silent_since = self.t

    def quiet(self, seconds: float) -> object | None:
        """``seconds`` 더 조용히 있는 동안의 판단 (0.1 s 마다 물어 처음 나온 것)."""
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + 0.1, 3)
            if (action := self.tracker.next_action(self.t, self.t - self.silent_since)) is not None:
                return action
        return None


def _started(lines: tuple[str, ...] = LINES) -> Model:
    tracker = Tracker(lines)
    tracker.mark_pushed(0.0)
    return Model(tracker)


class TestTrackerPush:
    def test_next_line_goes_in_inside_the_last_sentence_with_little_left(self) -> None:
        m = _started()
        m.say("오늘은 책상 서랍 이야기예요.")
        assert m.actions == []  # 마지막 문장 전에는 넣지 않는다
        m.say("늘 곁에 있는데도")
        assert m.actions and isinstance(m.actions[-1][1], Push)
        push = m.actions[-1][1]
        assert push.index == 1 and push.text == LINES[1] and push.by == "text"
        left = len(letters("좀처럼 열어 보지 않는 곳이죠"))
        assert left <= Tracker.PUSH_REMAIN < left + len(letters("있는데도"))

    def test_paraphrased_last_sentence_waits_for_quiet_then_moves_on(self) -> None:
        m = _started()
        m.say("오늘은 책상 서랍 이야기예요. 항상 옆에 있지만 잘 안 여는 곳이에요.")
        assert m.actions == []  # 일치가 끊기면 앞서 넣지 않는다
        assert m.quiet(Tracker.STALL_SEC - 0.1) is None
        action = m.quiet(0.2)
        assert action == Push(1, LINES[1], "stall")  # 바꿔 말하며 다 읽었다 — 다시 넣지 않는다

    def test_stop_mid_sentence_repushes_that_sentence_once_then_moves_on(self) -> None:
        m = _started()
        m.say("오늘은 책상 서랍 이야기예요. 늘 곁에")
        action = m.quiet(2.0)
        assert action == Repush("늘 곁에 있는데도 좀처럼 열어 보지 않는 곳이죠.")
        m.tracker.mark_repushed(m.t)
        assert m.quiet(Tracker.REPUSH_WAIT_SEC - 0.2) is None
        assert m.quiet(0.5) == Push(1, LINES[1], "stall")

    def test_only_an_acknowledgement_repushes_the_whole_line(self) -> None:
        m = _started()
        m.say("네, 라디오 시작할게요.")
        assert m.quiet(2.0) == Repush(LINES[0])

    def test_next_line_not_started_is_pushed_again(self) -> None:
        m = _started()
        m.say(LINES[0])
        assert m.tracker.pushed == 2
        assert m.quiet(2.0) == Repush(LINES[1])

    def test_a_few_missing_letters_at_a_sentence_end_are_let_go(self) -> None:
        # long_t #7: "…다음으로 미뤄"에서 멈춤 — 어미 "지죠"는 넘어가고 안 읽은 다음 문장만 다시 넣는다
        line = (
            "그래서 서랍 정리는 늘 다음으로 미뤄지죠. 마음먹고 열었다가도 물건 하나하나에 붙들려 시간이 가 버리니까요."
        )
        m = _started((line, LINES[3]))
        m.say("그래서 서랍 정리는 늘 다음으로 미뤄")
        assert m.quiet(2.0) == Repush("마음먹고 열었다가도 물건 하나하나에 붙들려 시간이 가 버리니까요.")

    def test_quiet_before_any_voice_is_not_a_stop(self) -> None:
        m = _started()
        assert m.quiet(5.0) is None


class TestTrackerClosing:
    def _at_closing(self) -> Model:
        m = _started()
        m.say(LINES[0])
        m.say(LINES[1])
        m.say(LINES[2])
        assert m.tracker.pushed == 4
        return m

    def test_song_after_the_introduction_and_a_short_quiet(self) -> None:
        m = self._at_closing()
        m.say(LINES[3])
        assert m.quiet(Tracker.CLOSING_QUIET_SEC - 0.2) is None
        assert m.quiet(0.3) == Closing("closing")

    def test_cut_introduction_is_repushed_then_song_after_fallback(self) -> None:
        m = self._at_closing()
        m.say("아이브의")
        assert m.quiet(2.0) == Repush(LINES[3])
        m.tracker.mark_repushed(m.t)
        assert m.quiet(Tracker.CLOSING_FALLBACK_SEC + 0.2) == Closing("quiet")

    def test_cap_when_the_model_never_speaks(self) -> None:
        tracker = Tracker(LINES[3:])
        tracker.mark_pushed(0.0)
        assert tracker.next_action(Tracker.CLOSING_CAP_SEC - 1, 10.0) is None
        assert tracker.next_action(Tracker.CLOSING_CAP_SEC, 10.0) == Closing("cap")


class TestTrackerInterruption:
    def test_reply_after_an_on_decision_does_not_count_as_reading(self) -> None:
        m = _started()
        m.say(LINES[0])  # 2번 대사가 들어갔다
        m.tracker.hold(m.t)
        m.say("음, 맞아요.")
        m.tracker.release(m.t)  # on — 맞장구에 대답만 하고 대사로 돌아오지 않았다
        m.say("그런 것들 꺼내 보면 이상하게 그때 생각이 나죠.")
        assert m.quiet(2.0) == Repush(LINES[1])

    def test_reading_on_after_the_reply_is_tracked_again(self) -> None:
        m = _started()
        m.say(LINES[0])
        m.tracker.hold(m.t)
        m.say("네,")
        m.tracker.release(m.t)
        m.say(LINES[1])
        assert m.tracker.pushed == 3  # 대본과 다시 맞으면 평소처럼 넣는다

    def test_off_script_letters_counts_the_reaction(self) -> None:
        m = _started()
        m.say(LINES[0][:20])
        since = m.t
        m.say("네, 저도 그래요")
        assert m.tracker.off_script_letters(since) == len(letters("네저도그래요"))
        m.say(LINES[0][20:])
        assert m.tracker.off_script_letters(since) == len(letters("네저도그래요"))

    def test_resume_forgets_unread_pushes(self) -> None:
        m = _started()
        m.say(LINES[0])
        assert m.tracker.pushed == 2 and m.tracker.done == 1
        assert m.tracker.reading_line() == LINES[1]
        assert m.tracker.resume(m.t) == 1
        assert m.tracker.pushed == 1


# ---------------------------------------------------------------------------
# 판정기
# ---------------------------------------------------------------------------


class TestRadioJudge:
    def test_sends_line_and_exchange_with_the_schema(self) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = iter([json.dumps({"reason": "질문", "state": "paused"})])
        assert RadioJudge(llm).classify("읽던 대사", [("user", "왜?"), ("assistant", "음,")]) == "paused"
        messages = llm.generate.call_args.args[0]
        assert messages[1]["content"] == "Line Ray was reading: 읽던 대사\n\nSince the user spoke:\nUser: 왜?\nRay: 음,"
        assert llm.generate.call_args.kwargs == {"tools": [], "response_format": RADIO_JUDGE_FORMAT}

    @pytest.mark.parametrize("state", ["maybe", "off"])  # 그만두자는 말도 paused — off 는 판정하지 않는다
    def test_unknown_state_raises(self, state: str) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = iter([json.dumps({"reason": "", "state": state})])
        with pytest.raises(RuntimeError):
            RadioJudge(llm).classify("l", [])

    def test_schema_offers_only_on_and_paused(self) -> None:
        assert RADIO_JUDGE_FORMAT["schema"]["properties"]["state"]["enum"] == ["on", "paused"]


# ---------------------------------------------------------------------------
# 라디오 상태
# ---------------------------------------------------------------------------


def _radio(*states: str, executor: object | None = None, keys: tuple[str, ...] = ("a", "b")) -> Radio:
    judge = MagicMock(spec=RadioJudge)
    judge.classify.side_effect = list(states)
    picker = ScriptPicker(_scripts(*keys), ScriptLog(":memory:"), rng=random.Random(0))
    return Radio(picker, judge, executor)  # type: ignore[arg-type]


def _line(text: str) -> Think:
    return Think(RADIO_LINE_TEMPLATE.format(text))


def _read(radio: Radio, text: str, t: float) -> float:
    """모델이 ``text`` 를 말한다 — 어절마다 출력 전사 스냅샷과 음성. 끝난 시각."""
    said = ""
    for word in text.split(" "):
        said += " " + word
        radio.on_voice(t)
        radio.on_transcript(t, "out-1", "assistant", said)
        t += 0.3
    return t


def _user(radio: Radio, text: str, t: float, segment: str = "in-1") -> None:
    radio.on_transcript(t, segment, "user", text)


class TestRadioStart:
    def test_start_pushes_the_first_line_and_reports_on(self) -> None:
        radio = _radio()
        output, commands = radio.start(0.0)
        assert json.loads(output) == {"status": "on"} and radio.status == "on"
        assert commands == [UpdateBackend(), _line(LINES[0])]
        assert json.loads(radio.start(0.1)[0]) == {"error": "the radio talk is already on"}

    def test_no_scripts(self) -> None:
        output, commands = _radio(keys=()).start(0.0)
        assert "no radio scripts" in json.loads(output)["error"] and commands == []

    def test_resume_only_when_paused(self) -> None:
        radio = _radio()
        assert "not paused" in json.loads(radio.resume(0.0)[0])["error"]
        radio.start(0.0)
        assert "not paused" in json.loads(radio.resume(0.1)[0])["error"]

    def test_lines_go_in_while_reading(self) -> None:
        radio = _radio()
        radio.start(0.0)
        t = _read(radio, "오늘은 책상 서랍 이야기예요. 늘 곁에 있는데도", 0.5)
        assert radio.step(t, 0.0) == [_line(LINES[1])]
        assert radio.step(t + 0.1, 0.0) == []


class TestRadioInterruption:
    def _interrupted(self, *states: str, executor: object | None = None) -> tuple[Radio, float]:
        radio = _radio(*states, executor=executor)
        radio.start(0.0)
        t = _read(radio, "오늘은 책상 서랍 이야기예요.", 0.5)
        _user(radio, "근데 서랍에 뭐 넣어 둬?", t)
        return radio, t

    def test_user_speech_holds_lines_until_the_judge_says_on(self) -> None:
        radio, t = self._interrupted("on")
        assert radio.step(t + 0.3, 0.0) == []  # 사용자 전사가 멈춘 지 0.6 s 전 — 아직 판정하지 않는다
        radio._judge.classify.assert_not_called()  # type: ignore[attr-defined]
        t = _read(radio, "늘 곁에 있는데도", t + 0.3)  # 보류 중에는 마지막 문장에 들어서도 넣지 않는다
        assert radio.step(t, 0.0) == []  # 판정을 보낸다
        radio._judge.classify.assert_called_once()  # type: ignore[attr-defined]
        assert radio.step(t + 0.1, 0.0) == [_line(LINES[1])]  # on — 보류를 풀고 바로 다음 대사

    def test_paused_stops_the_talk_and_resume_continues_from_the_unread_line(self) -> None:
        radio, t = self._interrupted("paused")
        assert radio.step(t + 0.7, 0.0) == []  # 판정을 보낸다
        assert radio.step(t + 0.8, 0.0) == [UpdateBackend(), Think(RADIO_PAUSED_NOTE)]
        assert radio.status == "paused"
        _read(radio, "서랍에는 볼펜 같은 걸 넣어 둬요.", t + 1.0)  # 멈춘 뒤 대화
        output, commands = radio.resume(t + 5.0)
        assert json.loads(output) == {"status": "on"}
        assert commands == [UpdateBackend(), Think(RADIO_RESUMED_NOTE), _line(LINES[0])]  # 1번 대사가 안 끝났다

    def test_conversation_while_paused_does_not_move_the_resume_point(self) -> None:
        radio, t = self._interrupted("paused")
        radio.step(t + 0.7, 0.0)
        radio.step(t + 0.8, 0.0)
        _read(radio, "늘 곁에 있는데도 좀처럼 열어 보지 않는 곳이죠. 그런 데가 있잖아요.", t + 1.0)  # 대사와 같은 말
        assert radio.resume(t + 5.0)[1][-1] == _line(LINES[0])  # 1번 대사부터 다시

    def test_new_user_words_make_an_on_stale(self) -> None:
        radio, t = self._interrupted("on", "paused")
        radio.step(t + 0.7, 0.0)
        _user(radio, "근데 서랍에 뭐 넣어 둬? 진짜 궁금해", t + 0.75)  # 판정 결과를 받기 전에 더 말했다
        assert radio.step(t + 0.8, 0.0) == []  # on 이지만 낡았다 — 보류 유지
        assert radio.step(t + 1.5, 0.0) == []  # 다시 판정
        assert radio.step(t + 1.6, 0.0) == [UpdateBackend(), Think(RADIO_PAUSED_NOTE)]

    def test_slow_judge_counts_as_on_but_a_late_pause_still_applies(self) -> None:
        pending: Future[str] = Future()
        executor = MagicMock()
        executor.submit.return_value = pending
        radio, t = self._interrupted(executor=executor)
        radio.step(t + 0.7, 0.0)
        assert radio.step(t + 0.7 + Radio.JUDGE_TIMEOUT_SEC - 0.1, 0.0) == []
        radio.step(t + 0.7 + Radio.JUDGE_TIMEOUT_SEC, 0.0)  # 시간 초과 → on 으로 보고 보류를 푼다
        t2 = _read(radio, "늘 곁에 있는데도", t + 3.0)
        assert radio.step(t2, 0.0) == [_line(LINES[1])]
        pending.set_result("paused")
        assert radio.step(t2 + 0.1, 0.0) == [UpdateBackend(), Think(RADIO_PAUSED_NOTE)]

    def test_start_while_paused_begins_a_new_script(self) -> None:
        radio, t = self._interrupted("paused")
        radio.step(t + 0.7, 0.0)
        radio.step(t + 0.8, 0.0)
        first = radio.script
        output, commands = radio.start(t + 3.0)
        assert json.loads(output) == {"status": "on"} and radio.script is not first
        assert commands == [UpdateBackend(), Think(RADIO_RESUMED_NOTE), _line(LINES[0])]


class TestRadioSong:
    def _at_song(self) -> tuple[Radio, float]:
        radio = _radio()
        radio.start(0.0)
        t = 0.5
        for i, line in enumerate(LINES):
            t = _read(radio, line, t)
            for _ in range(3):
                radio.step(t, 0.0)
                t += 0.1
            assert radio.tracker is not None and radio.tracker.pushed == min(i + 2, len(LINES))
        return radio, t

    def test_introduction_read_then_song_then_next_script_after_the_song(self) -> None:
        radio, t = self._at_song()
        assert radio.step(t, Tracker.CLOSING_QUIET_SEC) == [PlaySong(SONGS["IAM"])]
        assert radio.playing_song
        _user(radio, "노래 좋다", t + 1.0)  # 곡 중 말은 보지 않는다
        assert radio.step(t + 2.0, 5.0) == []
        commands = radio.on_song_end(t + 10.0, completed=True)
        assert commands == [_line(f"{RADIO_AFTER_SONG_OPENING} {LINES[0]}")]
        assert radio.status == "on" and not radio.playing_song

    def test_user_stopping_the_song_ends_the_radio(self) -> None:
        radio, t = self._at_song()
        radio.step(t, Tracker.CLOSING_QUIET_SEC)
        assert radio.on_song_end(t + 3.0, completed=False) == [UpdateBackend(), Think(RADIO_ENDED_NOTE)]
        assert radio.status == "off"

    def test_reset_clears_everything(self) -> None:
        radio, t = self._at_song()
        radio.reset()
        assert radio.status == "off" and radio.tracker is None and not radio.playing_song
