"""Tests for voice_pipeline.engines.gpt_live.radio — 대본 로드·검사, 고르기, 진행 추적, 라디오 상태."""

from __future__ import annotations

import json
import logging
import random
import re
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voice_pipeline.engines.gpt_live.instructions import (
    RADIO_AFTER_SONG_OPENING,
    RADIO_AUTO_RESUME_OPENING,
    RADIO_ENDED_NOTE,
    RADIO_JUDGE_FORMAT,
    RADIO_LATE_LINE_NOTE,
    RADIO_LINE_TEMPLATE,
    RADIO_PAUSED_NOTE,
    RADIO_RESUME_JUDGE_INSTRUCTIONS,
    RADIO_RESUME_JUDGE_SECOND,
    RADIO_RESUMED_NOTE,
    radio_resume_judge_format,
)
from voice_pipeline.engines.gpt_live.radio import (
    Closing,
    PlaySong,
    Push,
    Radio,
    RadioJudge,
    RadioResumeJudge,
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
        assert m.tracker.reading() == (0, 0.0)  # 2번 대사는 넣어 두기만 했다 — 대사 사이엔 방금 끝난 1번 대사
        assert m.tracker.resume(m.t) == 1
        assert m.tracker.pushed == 1

    def test_reading_moves_to_the_next_line_once_the_model_starts_it(self) -> None:
        m = _started()
        m.say(LINES[0])
        m.say("서랍은 집에서 가장 느리게")  # 2번 대사에 들어섰다
        index, start = m.tracker.reading()
        assert index == 1 and start == m.tracker._done_at[0]  # 미리 넣은 때가 아니라 1번 대사가 끝난 때부터

    def test_reading_after_resume_starts_at_the_resume(self) -> None:
        m = _started()
        m.say(LINES[0])
        m.tracker.resume(m.t + 5.0)
        m.tracker.mark_pushed(m.t + 5.0)  # 안 읽힌 2번 대사를 다시 넣었다
        assert m.tracker.reading() == (1, m.t + 5.0)


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

    def test_records_the_call_and_logs_the_time(self, call_log, caplog: pytest.LogCaptureFixture) -> None:
        llm = MagicMock(spec=ILLM)
        llm.model = "gpt-5.4-mini"
        llm.generate.return_value = iter([json.dumps({"reason": "질문", "state": "paused"})])
        with caplog.at_level(logging.INFO, logger="voice_pipeline.radio"):
            RadioJudge(llm).classify("읽던 대사", [("user", "왜?")])
        (rec,) = [r for r in call_log.records if r.operation == "judge"]
        assert (rec.module, rec.model, rec.status) == ("radio", "gpt-5.4-mini", "ok")
        assert json.loads(rec.metadata) == {"state": "paused", "reason": "질문"}
        assert re.search(r"Radio judge: paused \d+\.\d\ds \(질문\)", caplog.text)

    def test_failed_call_is_recorded_as_an_error(self, call_log) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.side_effect = RuntimeError("timeout")
        with pytest.raises(RuntimeError):
            RadioJudge(llm).classify("l", [])
        assert [r.status for r in call_log.records if r.operation == "judge"] == ["error"]

    def test_schema_offers_only_on_and_paused(self) -> None:
        assert RADIO_JUDGE_FORMAT["schema"]["properties"]["state"]["enum"] == ["on", "paused"]


class TestRadioResumeJudge:
    def _judge(self, decision: str) -> tuple[RadioResumeJudge, MagicMock]:
        llm = MagicMock(spec=ILLM)
        llm.model = "gpt-5.4-mini"
        llm.generate.return_value = iter([json.dumps({"reason": "이유", "decision": decision})])
        return RadioResumeJudge(llm), llm

    def test_sends_three_tagged_parts_with_ray_cut_to_its_last_sentence(self) -> None:
        judge, llm = self._judge("go_back")
        conversation = [
            ("assistant", "질경이예요.길가에서 자주 보여요."),
            ("user", "그만, 알았어"),
            ("assistant", "네."),
        ]
        assert judge.decide("읽던 대사", "그 풀 이름이 뭐야?", conversation) == "go_back"
        system, user = (m["content"] for m in llm.generate.call_args.args[0])
        assert system == RADIO_RESUME_JUDGE_INSTRUCTIONS
        assert user == (
            "<radio_line>읽던 대사</radio_line>\n<interruption>그 풀 이름이 뭐야?</interruption>\n"
            "<conversation>\nRay: 길가에서 자주 보여요.\nUser: 그만, 알았어\nRay: 네.\n</conversation>"
        )
        assert llm.generate.call_args.kwargs == {"tools": [], "response_format": radio_resume_judge_format()}

    def test_second_judgment_has_no_wait_for_answer(self) -> None:
        judge, llm = self._judge("go_back")
        judge.decide("l", "잠깐만", [], second=True)
        system, user = (m["content"] for m in llm.generate.call_args.args[0])
        assert system == RADIO_RESUME_JUDGE_INSTRUCTIONS + RADIO_RESUME_JUDGE_SECOND
        assert user.endswith("<conversation></conversation>")
        schema = llm.generate.call_args.kwargs["response_format"]["schema"]
        assert schema["properties"]["decision"]["enum"] == ["go_back", "stay_paused"]
        judge, _ = self._judge("wait_for_answer")
        with pytest.raises(RuntimeError):
            judge.decide("l", "잠깐만", [], second=True)

    def test_records_the_call(self, call_log, caplog: pytest.LogCaptureFixture) -> None:
        judge, _ = self._judge("stay_paused")
        with caplog.at_level(logging.INFO, logger="voice_pipeline.radio"):
            judge.decide("l", "그만하자", [])
        (rec,) = [r for r in call_log.records if r.operation == "resume_judge"]
        assert (rec.module, rec.model, rec.status) == ("radio", "gpt-5.4-mini", "ok")
        assert json.loads(rec.metadata) == {"decision": "stay_paused", "reason": "이유"}
        assert re.search(r"Radio resume judge: stay_paused \d+\.\d\ds \(이유\)", caplog.text)


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


def _read(radio: Radio, text: str, t: float, segment: str = "out-1") -> float:
    """모델이 ``text`` 를 말한다 — 어절마다 출력 전사 스냅샷과 음성. 끝난 시각."""
    said = ""
    for word in text.split(" "):
        said += " " + word
        radio.on_voice(t)
        radio.on_transcript(t, segment, "assistant", said)
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

    def test_judging_logs_the_judge_input(self, caplog: pytest.LogCaptureFixture) -> None:
        radio, t = self._interrupted("on")
        with caplog.at_level(logging.INFO, logger="voice_pipeline.radio"):
            radio.step(t + 0.7, 0.0)
        assert f"Radio: judging (user) — reading «{LINES[0]}» | Ray: 오늘은 책상 서랍 이야기예요." in caplog.text
        assert "| User: 근데 서랍에 뭐 넣어 둬?" in caplog.text

    def test_a_line_pushed_just_before_the_user_words_asks_to_answer_first(self) -> None:
        radio = _radio("on")
        radio.start(0.0)  # 첫 대사를 0.0 에 넣었다
        _read(radio, "오늘은 책상", 0.2)
        _user(radio, "잠깐만", 1.0)  # 전사는 목소리보다 늦다 — 대사는 사용자가 말하는 도중 들어갔다
        assert radio.step(1.0, 0.0) == [Think(RADIO_LATE_LINE_NOTE)]
        assert radio.step(1.1, 0.0) == []  # 한 번만

    def test_no_answer_first_note_when_the_line_went_in_earlier(self) -> None:
        radio = _radio("on")
        radio.start(0.0)
        _user(radio, "잠깐만", Radio.LATE_PUSH_SEC + 0.1)
        assert radio.step(Radio.LATE_PUSH_SEC + 0.1, 0.0) == []

    def test_the_start_request_closing_late_is_not_new_speech(self) -> None:
        radio = _radio("paused")
        _user(radio, "라디오 해줘", 0.0)  # 라디오가 꺼져 있을 때 들어온 시작 요청
        radio.start(0.5)
        _user(radio, "라디오 해줘", 0.9)  # 같은 구간의 마무리 갱신이 시작 뒤에 온다
        _user(radio, "라디오 해줘.", 1.0)  # 문장부호만 붙은 갱신
        assert radio._hold is None
        radio.step(2.0, 0.0)
        radio._judge.classify.assert_not_called()  # type: ignore[attr-defined]
        _user(radio, "근데 잠깐만", 2.5, segment="in-2")  # 시작 뒤 새로 한 말은 보류한다
        assert radio._hold is not None

    def test_speech_during_the_song_closing_late_is_not_new_speech(self) -> None:
        radio = _radio("paused")
        radio.start(0.0)
        radio.playing_song = True
        _user(radio, "노래 좋다", 5.0, segment="in-song")  # 곡 중에 한 말
        radio.on_song_end(9.0, completed=True)  # 다음 대본
        _user(radio, "노래 좋다.", 9.2, segment="in-song")  # 그 구간의 마무리 갱신이 곡 뒤에 온다
        assert radio.status == "on" and radio._hold is None

    def _judge_input(self, radio: Radio) -> tuple:
        return radio._judge.classify.call_args.args  # type: ignore[attr-defined]

    def test_the_judge_sees_an_earlier_exchange_in_the_same_line(self) -> None:
        radio = _radio("on", "paused")
        radio.start(0.0)
        t = _read(radio, "오늘은 책상 서랍 이야기예요.", 0.5)
        _user(radio, "배고프다.", t)  # 첫 끼어들기
        radio.step(t + 0.7, 0.0)  # 판정
        radio.step(t + 0.8, 0.0)  # on — 보류를 푼다
        assert radio._hold is None
        _read(radio, "아, 배고프시구나.", 2.6, segment="out-2")  # 레이의 반응 (대본 밖)
        _read(radio, "늘 곁에 있는데도 좀처럼", 3.4, segment="out-3")  # 대본으로 돌아왔다
        _user(radio, "응, 너무 배고파.", 4.6, segment="in-2")  # 같은 대사 안 두 번째 끼어들기
        radio.step(5.3, 0.0)
        line, exchange, earlier = self._judge_input(radio)
        assert line == LINES[0]
        assert earlier == [("user", "배고프다."), ("assistant", "아, 배고프시구나.")]  # 이어 읽은 낭독은 빼고
        assert exchange == [("assistant", "늘 곁에 있는데도 좀처럼"), ("user", "응, 너무 배고파.")]

    def test_an_interruption_in_an_earlier_line_is_left_out(self) -> None:
        radio = _radio("on", "paused")
        radio.start(0.0)
        _read(radio, "오늘은 책상", 0.5)
        _user(radio, "배고프다.", 1.2)  # 1번 대사에서 끼어듦
        radio.step(1.9, 0.0)
        radio.step(2.0, 0.0)  # on
        t = _read(radio, "서랍 이야기예요. 늘 곁에 있는데도 좀처럼 열어 보지 않는 곳이죠.", 2.2)
        radio.step(t, 0.0)  # 2번 대사를 넣는다
        t = _read(radio, "서랍은 집에서 가장 느리게", t + 1.0, segment="out-2")
        _user(radio, "너는 그런 거 있어?", t, segment="in-2")
        radio.step(t + 0.7, 0.0)
        line, _, earlier = self._judge_input(radio)
        assert line == LINES[1] and earlier == []

    def test_an_interruption_before_a_resume_is_left_out(self) -> None:
        radio, t = self._interrupted("paused", "paused")
        radio.step(t + 0.7, 0.0)
        radio.step(t + 0.8, 0.0)  # paused
        radio.resume(t + 5.0)  # 1번 대사를 다시 넣는다
        t2 = _read(radio, "오늘은 책상 서랍 이야기예요.", t + 5.5, segment="out-2")
        _user(radio, "잠깐만", t2, segment="in-2")
        radio.step(t2 + 0.7, 0.0)
        assert self._judge_input(radio)[2] == []  # 멈추기 전의 끼어들기는 넣지 않는다

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

    def _slow(self, *futures: Future[str]) -> tuple[Radio, float]:
        executor = MagicMock()
        executor.submit.side_effect = list(futures)
        return self._interrupted(executor=executor)

    def test_a_slow_judge_holds_until_a_line_is_due_then_waits_before_going_on(self) -> None:
        pending: Future[str] = Future()
        radio, t = self._slow(pending)
        radio.step(t + 0.7, 0.0)  # 판정을 보낸다
        assert radio.step(t + 3.0, 0.0) == []  # 넣을 일이 없으면 판정이 늦어도 보류를 유지한다
        t2 = _read(radio, "늘 곁에 있는데도", t + 3.0)  # 레이가 이어 읽어 넣기 지점에 닿았다
        assert radio.step(t2, 0.0) == []  # 판정이 없으니 기다린다
        assert radio.step(t2 + Radio.DUE_WAIT_SEC - 0.1, 0.0) == []
        assert radio.step(t2 + Radio.DUE_WAIT_SEC, 0.0) == [_line(LINES[1])]  # 그래도 없으면 on 으로 보고 넣는다
        pending.set_result("paused")  # 늦게 온 paused 도 적용한다
        assert radio.step(t2 + Radio.DUE_WAIT_SEC + 0.1, 0.0) == [UpdateBackend(), Think(RADIO_PAUSED_NOTE)]

    def test_a_pause_while_waiting_stops_before_the_due_line(self) -> None:
        pending: Future[str] = Future()
        radio, t = self._slow(pending)
        radio.step(t + 0.7, 2.0)  # 판정을 보낸다 — 레이가 조용해 무음 폴백을 할 때가 됐다
        assert radio.step(t + 2.0, 3.0) == []  # 판정을 기다린다
        pending.set_result("paused")
        assert radio.step(t + 2.1, 3.0) == [UpdateBackend(), Think(RADIO_PAUSED_NOTE)]  # 다시 넣지 않고 멈춘다

    def test_the_wait_counts_from_the_judge_call_when_the_user_talks_long(self) -> None:
        pending: Future[str] = Future()
        radio, t = self._slow(pending)
        assert radio.step(t + 0.3, 2.0) == []  # 무음이 찼지만 사용자가 아직 말하는 중 — 판정 전
        _user(radio, "근데 서랍에 뭐 넣어 둬? 진짜 궁금해", t + 2.0)
        radio.step(t + 2.6, 4.0)  # 사용자 전사가 멈췄다 — 이제 판정을 부른다
        assert radio.step(t + 2.6 + Radio.DUE_WAIT_SEC - 0.1, 6.0) == []  # 기다림은 판정 호출부터
        rest = "늘 곁에 있는데도 좀처럼 열어 보지 않는 곳이죠."  # 1번 대사의 안 읽은 문장
        assert radio.step(t + 2.6 + Radio.DUE_WAIT_SEC, 6.0) == [_line(rest)]

    def test_a_judgment_for_an_earlier_interruption_is_dropped(self) -> None:
        first: Future[str] = Future()
        second: Future[str] = Future()
        radio, t = self._slow(first, second)
        radio.step(t + 0.7, 2.0)  # 첫 끼어들기 판정 — 레이가 조용하다
        radio.step(t + 0.7 + Radio.DUE_WAIT_SEC, 4.0)  # 판정 없이 기다림이 끝나 보류를 풀고 진행
        assert radio._hold is None
        _user(radio, "아 그렇구나", t + 5.0, segment="in-2")  # 새 끼어들기
        radio.step(t + 5.7, 0.0)  # 앞 판정이 아직 없어도 새 판정을 부른다
        assert len(radio._calls) == 2
        first.set_result("paused")
        assert radio.step(t + 5.8, 0.0) == []  # 앞 끼어들기의 판정은 버린다
        assert radio.status == "on"

    def test_a_judgment_later_than_the_cap_is_dropped(self) -> None:
        pending: Future[str] = Future()
        radio, t = self._slow(pending)
        radio.step(t + 0.7, 0.0)
        radio.step(t + 0.7 + Radio.JUDGE_LATE_MAX_SEC, 0.0)  # 상한 — 이 판정은 더 기다리지 않는다
        pending.set_result("paused")
        assert radio.step(t + 0.8 + Radio.JUDGE_LATE_MAX_SEC, 0.0) == []
        assert radio.status == "on"

    def test_start_while_paused_begins_a_new_script(self) -> None:
        radio, t = self._interrupted("paused")
        radio.step(t + 0.7, 0.0)
        radio.step(t + 0.8, 0.0)
        first = radio.script
        output, commands = radio.start(t + 3.0)
        assert json.loads(output) == {"status": "on"} and radio.script is not first
        assert commands == [UpdateBackend(), Think(RADIO_RESUMED_NOTE), _line(LINES[0])]


def _auto_radio(*decisions: object, executor: object | None = None) -> Radio:
    """멈추는 끼어들기 하나와 복귀 판정기가 있는 라디오. 판정은 ``decisions`` 를 차례로 (예외면 실패)."""
    judge = MagicMock(spec=RadioJudge)
    judge.classify.side_effect = ["paused"]
    resume_judge = MagicMock(spec=RadioResumeJudge)
    resume_judge.decide.side_effect = list(decisions)
    picker = ScriptPicker(_scripts("a", "b"), ScriptLog(":memory:"), rng=random.Random(0))
    return Radio(picker, judge, executor, resume_judge)  # type: ignore[arg-type]


class TestRadioAutoResume:
    QUIET = Radio.RESUME_QUIET_SEC
    ASK = Radio.RESUME_QUIET_SEC - Radio.RESUME_JUDGE_LEAD_SEC

    def _paused(self, *decisions: object, executor: object | None = None) -> tuple[Radio, float]:
        """멈추고 레이가 대답한 라디오와 레이 말소리가 마지막으로 난 시각."""
        radio = _auto_radio(*decisions, executor=executor)
        radio.start(0.0)
        t = _read(radio, "오늘은 책상 서랍 이야기예요.", 0.5)
        _user(radio, "근데 서랍에 뭐 넣어 둬?", t)
        radio.step(t + 0.7, 0.0)  # 끼어들기 판정
        radio.step(t + 0.8, 0.0)  # paused
        assert radio.status == "paused" and radio.auto_resume
        t = _read(radio, "저는 메모지를 넣어 둘 것 같아요.", t + 1.0, segment="out-2")  # 레이의 대답
        return radio, t - 0.3

    def _decide(self, radio: Radio) -> MagicMock:
        return radio._resume_judge.decide  # type: ignore[attr-defined,return-value]

    def test_goes_back_on_its_own_with_the_opening_after_a_quiet(self) -> None:
        radio, last = self._paused("go_back")
        assert radio.step(last + self.ASK - 0.1, 0.0) == []
        self._decide(radio).assert_not_called()
        assert radio.step(last + self.ASK, 0.0) == []  # 판정을 미리 부른다
        line, interruption, conversation = self._decide(radio).call_args.args
        assert (line, interruption) == (LINES[0], "근데 서랍에 뭐 넣어 둬?")
        assert conversation == [("assistant", "저는 메모지를 넣어 둘 것 같아요.")]
        assert self._decide(radio).call_args.kwargs == {"second": False}
        assert radio.step(last + self.QUIET - 0.1, 0.0) == []  # 결과는 잠잠함이 찼을 때 적용한다
        opening = _line(f"{RADIO_AUTO_RESUME_OPENING} {LINES[0]}")  # 1번 대사가 안 끝났다
        assert radio.step(last + self.QUIET, 0.0) == [UpdateBackend(), Think(RADIO_RESUMED_NOTE), opening]
        assert radio.status == "on" and not radio.auto_resume

    def test_new_speech_drops_the_judgment_and_the_next_quiet_asks_again(self) -> None:
        radio, last = self._paused("stay_paused", "go_back")
        radio.step(last + self.ASK, 0.0)
        _user(radio, "아 그렇구나", last + self.QUIET - 0.5, segment="in-2")
        assert radio.step(last + self.QUIET, 0.0) == []  # 판정 뒤 사용자가 말했다 — 버린다
        assert radio.status == "paused" and radio.auto_resume
        lull = last + self.QUIET - 0.5
        radio.step(lull + self.ASK, 0.0)
        assert self._decide(radio).call_count == 2
        assert radio.step(lull + self.QUIET, 0.0)[-1] == _line(f"{RADIO_AUTO_RESUME_OPENING} {LINES[0]}")

    def test_a_running_delegation_is_not_a_quiet(self) -> None:
        radio, last = self._paused("go_back")
        radio.step(last + 3.0, 0.0, backend_busy=True)  # 백엔드가 아직 답을 만드는 중
        radio.step(last + self.ASK + 1.0, 0.0)
        self._decide(radio).assert_not_called()
        radio.step(last + 3.0 + self.ASK, 0.0)
        self._decide(radio).assert_called_once()

    def test_stay_paused_stops_asking(self) -> None:
        radio, last = self._paused("stay_paused")
        radio.step(last + self.ASK, 0.0)
        assert radio.step(last + self.QUIET, 0.0) == []
        assert radio.status == "paused" and not radio.auto_resume
        _user(radio, "근데 있잖아", last + 20.0, segment="in-2")
        radio.step(last + 40.0, 0.0)
        self._decide(radio).assert_called_once()  # 이어 달라는 요청으로만 돌아간다
        assert radio.resume(last + 41.0)[1][-1] == _line(LINES[0])  # 재개는 그대로 — 문구 없이

    def test_waiting_for_an_answer_asks_once_more_later_without_that_option(self) -> None:
        radio, last = self._paused("wait_for_answer", "go_back")
        radio.step(last + self.ASK, 0.0)
        assert radio.step(last + self.QUIET, 0.0) == []
        later = Radio.RESUME_LATER_SEC - Radio.RESUME_JUDGE_LEAD_SEC
        radio.step(last + later - 0.1, 0.0)
        assert self._decide(radio).call_count == 1
        radio.step(last + later, 0.0)
        assert self._decide(radio).call_args.kwargs == {"second": True}
        assert radio.step(last + Radio.RESUME_LATER_SEC, 0.0)[0] == UpdateBackend()
        assert radio.status == "on"

    def test_a_failed_judgment_is_asked_again_right_away_while_quiet(self) -> None:
        radio, last = self._paused(RuntimeError("timeout"), "go_back")
        radio.step(last + self.ASK, 0.0)
        radio.step(last + self.ASK + 0.1, 0.0)  # 실패를 보고 바로 다시
        assert self._decide(radio).call_count == 2
        assert radio.step(last + self.QUIET, 0.0)[0] == UpdateBackend()

    def test_repeated_failures_give_up_until_someone_speaks(self) -> None:
        failures = [RuntimeError("timeout")] * Radio.RESUME_ATTEMPTS
        radio, last = self._paused(*failures, "go_back")
        for k in range(Radio.RESUME_ATTEMPTS + 2):
            radio.step(last + self.ASK + 0.1 * k, 0.0)
        assert self._decide(radio).call_count == Radio.RESUME_ATTEMPTS
        assert radio.step(last + 30.0, 0.0) == [] and radio.status == "paused"
        _user(radio, "음", last + 31.0, segment="in-2")
        radio.step(last + 31.0 + self.ASK, 0.0)
        assert radio.step(last + 31.0 + self.QUIET, 0.0)[0] == UpdateBackend()

    def test_a_user_song_while_paused_turns_it_off(self) -> None:
        radio, last = self._paused("go_back")
        radio.on_user_song()
        radio.step(last + 30.0, 0.0)
        self._decide(radio).assert_not_called()
        assert radio.status == "paused" and not radio.auto_resume

    def test_a_resume_request_while_judging_drops_the_judgment(self) -> None:
        pending: Future[str] = Future()
        executor = MagicMock()
        judged: Future[str] = Future()
        judged.set_result("paused")
        executor.submit.side_effect = [judged, pending]
        radio, last = self._paused(executor=executor)
        radio.step(last + self.ASK, 0.0)
        assert radio._resume_check is not None
        radio.resume(last + self.ASK + 0.5)
        pending.set_result("go_back")
        assert radio.step(last + self.QUIET, 0.0) == [] and radio._resume_check is None

    def test_no_resume_judge_never_goes_back_on_its_own(self) -> None:
        radio = _radio("paused")
        radio.start(0.0)
        t = _read(radio, "오늘은 책상 서랍 이야기예요.", 0.5)
        _user(radio, "근데 서랍에 뭐 넣어 둬?", t)
        radio.step(t + 0.7, 0.0)
        radio.step(t + 0.8, 0.0)
        assert radio.status == "paused" and not radio.auto_resume
        assert radio.step(t + 60.0, 0.0) == [] and radio.status == "paused"


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

    def test_the_song_waits_for_a_slow_judgment_too(self) -> None:
        pending: Future[str] = Future()
        executor = MagicMock()
        executor.submit.return_value = pending
        radio = _radio(executor=executor)
        radio.start(0.0)
        t = 0.5
        for line in LINES:
            t = _read(radio, line, t)
            for _ in range(3):
                radio.step(t, 0.0)
                t += 0.1
        t += Radio.LATE_PUSH_SEC  # 곡 소개 대사를 넣은 지 한참 뒤
        _user(radio, "잠깐만", t)  # 곡 소개를 다 읽은 직후 끼어들었다
        assert radio.step(t + 0.7, 1.0) == []  # 판정을 보낸다 — 곡을 틀 때가 됐지만 기다린다
        assert radio.step(t + 0.7 + Radio.DUE_WAIT_SEC - 0.1, 4.0) == []
        assert radio.step(t + 0.7 + Radio.DUE_WAIT_SEC, 4.0) == [PlaySong(SONGS["IAM"])]  # 그래도 없으면 곡

    def test_user_stopping_the_song_ends_the_radio(self) -> None:
        radio, t = self._at_song()
        radio.step(t, Tracker.CLOSING_QUIET_SEC)
        assert radio.on_song_end(t + 3.0, completed=False) == [UpdateBackend(), Think(RADIO_ENDED_NOTE)]
        assert radio.status == "off"

    def test_reset_clears_everything(self) -> None:
        radio, t = self._at_song()
        radio.reset()
        assert radio.status == "off" and radio.tracker is None and not radio.playing_song
