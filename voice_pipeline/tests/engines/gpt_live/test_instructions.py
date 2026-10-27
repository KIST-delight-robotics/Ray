"""Tests for voice_pipeline.engines.gpt_live.instructions."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from voice_pipeline.engines.gpt_live.instructions import (
    DEFAULT_BACKEND_INSTRUCTIONS,
    DEFAULT_LIVE_INSTRUCTIONS,
    RADIO_ENDED_NOTE,
    RADIO_LINE_TEMPLATE,
    RADIO_ON_NOTE,
    RADIO_PAUSED_NOTE,
    SONG_BACKEND_INSTRUCTIONS,
    build_backend_instructions,
    build_live_instructions,
    build_radio_judge_prompt,
    build_radio_section,
    build_songs_section,
)
from voice_pipeline.engines.gpt_live.songs import Song

_SONGS = {
    "IAM": Song("IAM", "I AM", "IVE", artist_aliases=("아이브",)),
    "Butter_BTS": Song("Butter_BTS", "Butter", "BTS"),
}

_NOW = datetime(2026, 9, 13, 9, 0, tzinfo=ZoneInfo("Asia/Seoul"))
_MARKER = "[Current session — started 2026-09-13 (Sun) 09:00 KST]"


class TestBuildLiveInstructions:
    def test_build_live_instructions_without_context_is_base_and_current_session(self) -> None:
        text = build_live_instructions(now=_NOW)
        assert text.startswith("# Voice model (Ray) instructions\nYou are Ray")
        assert "already know about the user, from earlier sessions" not in text
        assert text.rstrip().endswith(_MARKER)

    def test_build_live_instructions_appends_profile_and_recent_sessions(self) -> None:
        text = build_live_instructions(
            "[User Profile]\ninterest::movie: SF",
            ["[Session — 2026-09-10 (Thu) 10:00]\n- User saw Dune 2.", "[Session — 2026-09-12 (Sat) 20:00]\n- OST."],
            now=_NOW,
        )
        base_end = text.index("Do not guess the result while waiting.")
        assert text.index("already know about the user, from earlier sessions:") > base_end
        assert text.index("[User Profile]") < text.index("2026-09-10") < text.index("2026-09-12") < text.index(_MARKER)
        assert text.rstrip().endswith(_MARKER)

    def test_policies_follow_the_template_order(self) -> None:
        text = DEFAULT_LIVE_INSTRUCTIONS
        order = [
            "Backchannel policy:",
            "Interruption policy:",
            "Delegation policy:",
            "Backend tools:",
            "Delegate to the backend when:",
            "Do not delegate to the backend when:",
        ]
        positions = [text.index(heading) for heading in order]
        assert positions == sorted(positions)
        assert text.count("Delegate to the backend when:") == 1

    def test_radio_is_one_tool_line_and_one_delegation_line(self) -> None:
        # 진행 규칙은 대화 모델이 모른다 — 앱이 대사를 넣는다. Radio 줄은 레이가 진행하는 것이라는 정의
        text = DEFAULT_LIVE_INSTRUCTIONS
        tools = text[text.index("Backend tools:") : text.index("Delegate to the backend when:")]
        delegate = text[text.index("Delegate to the backend when:") : text.index("Do not delegate")]
        assert "- Radio: your own radio talk, which you host and speak yourself." in tools
        assert "- The user asks to start the radio, or to continue a paused radio talk." in delegate
        assert text.count("radio") == 3 and "DJ" not in text


class TestBackendInstructions:
    def test_default_backend_has_play_song_but_not_stop_song(self) -> None:
        text = DEFAULT_BACKEND_INSTRUCTIONS
        # OpenAI 백엔드 템플릿의 세 섹션 순서, 툴 규칙은 Task instructions 안에
        assert text.startswith("# Backend instructions\n")
        assert text.index("## Voice conversation context") < text.index("## Task instructions")
        assert text.index("## Task instructions") < text.index("play_song") < text.index("## Return the result")
        assert "stop_song" not in text  # 정지는 재생 중 교체되는 백엔드(SONG_BACKEND_INSTRUCTIONS)만 안다
        assert "stop_song" in SONG_BACKEND_INSTRUCTIONS

    def test_default_backend_returns_empty_after_actions(self) -> None:
        line = "After start_radio, resume_radio, play_song or end_conversation, return an empty message"
        assert line in DEFAULT_BACKEND_INSTRUCTIONS

    def test_default_backend_ends_only_on_the_users_own_goodbye(self) -> None:
        assert "in their own words in the transcript: end_conversation" in DEFAULT_BACKEND_INSTRUCTIONS
        assert "Never call it for any other reason" in DEFAULT_BACKEND_INSTRUCTIONS

    def test_default_backend_radio_block_sits_in_task_instructions(self) -> None:
        task = DEFAULT_BACKEND_INSTRUCTIONS[DEFAULT_BACKEND_INSTRUCTIONS.index("## Task instructions") :]
        block = task[task.index("Radio (see Radio talk)") : task.index("Ending")]
        assert "start_radio" in block and "resume_radio" in block and "디제이" in block
        assert "also while a talk is paused" in block  # 멈춘 뒤 "라디오 해 줘" 는 새로 시작
        assert "이야기 계속해 줘" in block and "Do not call start_radio or resume_radio in any other case" in block

    def test_backend_builder_has_no_voice_section(self) -> None:
        # 말투는 백엔드가 함께 받는 대화 모델 세션 지시문에 있다 (FINDINGS §4) — 중복 절을 두지 않는다
        text = build_backend_instructions()
        assert text.startswith(DEFAULT_BACKEND_INSTRUCTIONS) and "Ray's voice" not in text
        assert "## Songs on Ray" not in text and "## Radio talk" not in text

    def test_backend_builder_appends_songs_section(self) -> None:
        text = build_backend_instructions(_SONGS)
        assert text.rstrip().endswith("IAM: I AM — IVE (아이브)")
        assert build_songs_section({}) == ""

    def test_default_backend_lists_songs_instead_of_playing(self) -> None:
        task = DEFAULT_BACKEND_INSTRUCTIONS[DEFAULT_BACKEND_INSTRUCTIONS.index("## Task instructions") :]
        assert "asks which songs Ray has" in task and "do not play" in task
        assert "Never choose for them unless asked" in task


class TestRadioInstructions:
    def test_backend_builder_appends_radio_status_last(self) -> None:
        text = build_backend_instructions(_SONGS, "paused")
        assert text.index("## Songs on Ray") < text.index("## Radio talk")
        assert text.rstrip().endswith("## Radio talk\nRadio talk status: paused")
        with pytest.raises(ValueError):
            build_radio_section("maybe")

    def test_line_template_says_only_what_ray_is_saying(self) -> None:
        # "radio"·"next line" 을 넣으면 모델이 틀을 말로 옮겼다 (dj-mode-worknote.md)
        assert RADIO_LINE_TEMPLATE.format("대사") == "You are now saying: «대사»"
        assert "radio" not in RADIO_LINE_TEMPLATE.lower() and "next" not in RADIO_LINE_TEMPLATE.lower()

    def test_status_notes(self) -> None:
        assert "Do not continue the talk yourself" in RADIO_PAUSED_NOTE
        assert "delegate to the backend to resume it" in RADIO_PAUSED_NOTE and "이야기 계속해 줘" in RADIO_PAUSED_NOTE
        notes = (RADIO_ON_NOTE, RADIO_PAUSED_NOTE, RADIO_ENDED_NOTE)
        assert len(set(notes)) == 3 and all(n.startswith("Radio talk: ") for n in notes)
        # 시작·재개 문구는 말투만 — 낭독 어휘가 말로 새어 나왔다
        assert not any(w in RADIO_ON_NOTE.lower() for w in ("read", "line", "script", "application"))
        assert all(len(n) < 400 for n in notes)  # thinking.append 한도 500 토큰에 한참 못 미친다

    def test_judge_prompt_lists_the_exchange(self) -> None:
        text = build_radio_judge_prompt("읽던 대사", [("user", "왜?"), ("assistant", "음,")])
        assert text == "Line Ray was reading: 읽던 대사\n\nSince the user spoke:\nUser: 왜?\nRay: 음,"

    def test_judge_prompt_puts_an_earlier_exchange_first(self) -> None:
        earlier = [("user", "배고프다."), ("assistant", "아.")]
        text = build_radio_judge_prompt("읽던 대사", [("user", "응, 너무 배고파.")], earlier)
        assert text == (
            "Line Ray was reading: 읽던 대사\n\nEarlier during this line:\nUser: 배고프다.\nRay: 아.\n\n"
            "Since the user spoke:\nUser: 응, 너무 배고파."
        )
