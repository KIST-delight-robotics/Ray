"""Tests for voice_pipeline.engines.gpt_live.instructions."""

from __future__ import annotations

from voice_pipeline.engines.gpt_live.dj import Segment
from voice_pipeline.engines.gpt_live.instructions import (
    DEFAULT_BACKEND_INSTRUCTIONS,
    DEFAULT_LIVE_INSTRUCTIONS,
    DJ_BEGIN_INSTRUCTIONS,
    DJ_ENDED_INSTRUCTIONS,
    DJ_RETURN_INSTRUCTIONS,
    DJ_SONG_ENDED_INSTRUCTIONS,
    SONG_BACKEND_INSTRUCTIONS,
    build_backend_instructions,
    build_current_segment_section,
    build_live_instructions,
    build_segment_thinking,
    build_songs_section,
    build_voice_section,
)
from voice_pipeline.engines.gpt_live.songs import Song

_SONGS = {
    "IAM": Song("IAM", "I AM", "IVE", artist_aliases=("아이브",)),
    "Butter_BTS": Song("Butter_BTS", "Butter", "BTS"),
}


class TestBuildLiveInstructions:
    def test_build_live_instructions_without_context_is_base_only(self) -> None:
        text = build_live_instructions()
        assert text.startswith("# Ray\nYou are Ray")
        assert "already know about the user, from earlier sessions" not in text

    def test_build_live_instructions_appends_profile_and_recent_sessions(self) -> None:
        text = build_live_instructions(
            "[User Profile]\ninterest::movie: SF",
            ["[2026-09-10 10:00 session]\n- User saw Dune 2.", "[2026-09-12 20:00 session]\n- User liked the OST."],
        )
        base_end = text.index("Do not mention the backend.")
        assert text.index("already know about the user, from earlier sessions:") > base_end
        assert text.index("[User Profile]") < text.index("[2026-09-10 10:00 session]") < text.index("[2026-09-12")

    def test_mode_sections_in_order_with_delegation_lists(self) -> None:
        # 모드별 구성: 상황마다 한 절만 보면 된다. 위임 조건은 모드마다 명시적인 목록으로(FINDINGS §4)
        text = DEFAULT_LIVE_INSTRUCTIONS
        assert text.index("# Modes") < text.index("# Conversation mode") < text.index("# DJ mode")
        conv = text[text.index("# Conversation mode") : text.index("# DJ mode")]
        dj = text[text.index("# DJ mode") :]
        assert "play or stop music" in conv[conv.index("Delegate to the backend when:") : conv.index("Do not delegate")]
        assert "start DJ mode" in conv and "do not start hosting yourself" in conv
        assert dj.index("## Hosting") < dj.index("## Listener interruption") < dj.index("## Delegation")
        assert "reached its song introduction" in dj[dj.index("Delegate to the backend when:") :]

    def test_dj_hosting_rules_cover_experiment_findings(self) -> None:
        # 하네스 실험(docs/dj-mode-worknote.md): 소재만 받고 조기 발화, 답하자마자 이어가기, 노트 밖 사실
        dj = DEFAULT_LIVE_INSTRUCTIONS[DEFAULT_LIVE_INSTRUCTIONS.index("# DJ mode") :]
        assert "Do not start until the application tells you to begin" in dj
        assert "Do not return to the talk on your own" in dj
        assert "Do not state facts that are not in the notes" in dj
        assert "If you already asked and the user stayed silent, continue" in dj


class TestBackendInstructions:
    def test_default_backend_has_play_song_but_not_stop_song(self) -> None:
        text = DEFAULT_BACKEND_INSTRUCTIONS
        # OpenAI 백엔드 템플릿의 세 섹션 순서, 툴 규칙은 Task instructions 안에
        assert text.index("## Voice conversation context") < text.index("## Task instructions")
        assert text.index("## Task instructions") < text.index("play_song") < text.index("## Return the result")
        assert "stop_song" not in text  # 정지는 재생 중 교체되는 백엔드(SONG_BACKEND_INSTRUCTIONS)만 안다
        assert "stop_song" in SONG_BACKEND_INSTRUCTIONS

    def test_default_backend_returns_empty_after_actions(self) -> None:
        line = "After start_dj, stop_dj, play_dj_song, play_song or end_conversation, return an empty message"
        assert line in DEFAULT_BACKEND_INSTRUCTIONS

    def test_default_backend_ends_only_on_the_users_own_goodbye(self) -> None:
        assert "in their own words in the transcript: end_conversation" in DEFAULT_BACKEND_INSTRUCTIONS
        assert "Never call it for any other reason" in DEFAULT_BACKEND_INSTRUCTIONS

    def test_default_backend_dj_block_sits_in_task_instructions(self) -> None:
        task = DEFAULT_BACKEND_INSTRUCTIONS[DEFAULT_BACKEND_INSTRUCTIONS.index("## Task instructions") :]
        block = task[task.index("DJ mode (see Current segment)") : task.index("Ending")]
        assert "start_dj" in block and "play_dj_song" in block and "stop_dj" in block
        assert "Do not write an opening or a talk" in block
        assert "A question alone does not end DJ mode" in block

    def test_voice_section_default_and_profile_override(self) -> None:
        plain = build_voice_section()
        assert plain.startswith("## Ray's voice") and "해요체" in plain and "overrides" not in plain
        with_pref = build_voice_section("prefers formal tone")
        assert "overrides the default: prefers formal tone" in with_pref

    def test_backend_builder_appends_voice_section(self) -> None:
        text = build_backend_instructions("formal")
        assert text.startswith(DEFAULT_BACKEND_INSTRUCTIONS) and text.rstrip().endswith("formal")
        assert "## Songs on Ray" not in text

    def test_backend_builder_appends_songs_section_after_voice(self) -> None:
        text = build_backend_instructions("formal", _SONGS)
        assert text.index("## Ray's voice") < text.index("## Songs on Ray")
        assert text.rstrip().endswith("IAM: I AM — IVE (아이브)")
        assert build_songs_section({}) == ""

    def test_default_backend_lists_songs_instead_of_playing(self) -> None:
        task = DEFAULT_BACKEND_INSTRUCTIONS[DEFAULT_BACKEND_INSTRUCTIONS.index("## Task instructions") :]
        assert "asks which songs Ray has" in task and "do not play" in task
        assert "Never choose for them unless asked" in task


_SEGMENT = Segment(key="drawer", topic="서랍 속에 남아 있는 것들", song=_SONGS["IAM"], notes="- 첫 줄\n- 둘째 줄")


class TestDJInstructions:
    def test_backend_builder_appends_current_segment_last(self) -> None:
        text = build_backend_instructions("formal", _SONGS, _SEGMENT)
        assert text.startswith(DEFAULT_BACKEND_INSTRUCTIONS)
        assert text.index("## Ray's voice") < text.index("## Songs on Ray") < text.index("## Current segment")
        assert "Current topic: 서랍 속에 남아 있는 것들" in text and "Assigned song: I AM — IVE" in text
        assert text.rstrip().endswith("- 첫 줄\n- 둘째 줄")
        assert "## Current segment" not in build_backend_instructions("formal", _SONGS)
        assert build_current_segment_section(_SEGMENT).startswith("## Current segment\n")

    def test_segment_thinking_marks_replacement_and_carries_notes(self) -> None:
        text = build_segment_thinking(_SEGMENT)
        assert text.startswith("DJ segment (current; replaces any earlier segment).")
        assert "Topic: 서랍 속에 남아 있는 것들" in text and "Assigned song: I AM — IVE" in text
        assert text.rstrip().endswith("- 첫 줄\n- 둘째 줄")

    def test_app_signals_are_short_and_distinct(self) -> None:
        signals = (DJ_BEGIN_INSTRUCTIONS, DJ_SONG_ENDED_INSTRUCTIONS, DJ_RETURN_INSTRUCTIONS, DJ_ENDED_INSTRUCTIONS)
        assert len(set(signals)) == 4
        assert all(len(s) < 200 for s in signals)  # instructions.append 한도 500 토큰에 한참 못 미친다
        assert "Begin now" in DJ_BEGIN_INSTRUCTIONS and "Begin now" in DJ_SONG_ENDED_INSTRUCTIONS
