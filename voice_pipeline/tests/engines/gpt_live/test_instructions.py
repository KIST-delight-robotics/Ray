"""Tests for voice_pipeline.engines.gpt_live.instructions."""

from __future__ import annotations

from voice_pipeline.engines.gpt_live.instructions import (
    DEFAULT_BACKEND_INSTRUCTIONS,
    SONG_BACKEND_INSTRUCTIONS,
    STORY_MODE_INSTRUCTIONS,
    build_live_instructions,
    build_story_backend_instructions,
)


class TestBuildLiveInstructions:
    def test_build_live_instructions_without_context_is_base_only(self) -> None:
        text = build_live_instructions()
        assert text.startswith("You are Ray")
        assert "already know about the user, from earlier sessions" not in text

    def test_build_live_instructions_appends_profile_and_recent_sessions(self) -> None:
        text = build_live_instructions(
            "[User Profile]\ninterest::movie: SF",
            ["[2026-09-10 10:00 session]\n- User saw Dune 2.", "[2026-09-12 20:00 session]\n- User liked the OST."],
        )
        base_end = text.index("Do not mention the backend.")
        assert text.index("already know about the user, from earlier sessions:") > base_end
        assert text.index("[User Profile]") < text.index("[2026-09-10 10:00 session]") < text.index("[2026-09-12")

    def test_music_delegation_lines_sit_inside_the_policy_lists(self) -> None:
        # 위임 규칙은 목록 항목으로 있어야 위임률이 유지된다(FINDINGS §4) — 툴 목록과 위임 조건 목록에 각각 한 줄
        text = build_live_instructions()
        tools_block = text[text.index("Backend tools:") : text.index("Delegate to the backend when:")]
        when_block = text[text.index("Delegate to the backend when:") : text.index("Do not delegate")]
        assert "- Music:" in tools_block
        assert "play or stop music" in when_block


class TestBackendInstructions:
    def test_default_backend_has_play_song_but_not_stop_song(self) -> None:
        text = DEFAULT_BACKEND_INSTRUCTIONS
        # OpenAI 백엔드 템플릿의 세 섹션 순서, 툴 규칙은 Task instructions 안에
        assert text.index("## Voice conversation context") < text.index("## Task instructions")
        assert text.index("## Task instructions") < text.index("play_song") < text.index("## Return the result")
        assert "stop_song" not in text  # 정지는 재생 중 교체되는 백엔드(SONG_BACKEND_INSTRUCTIONS)만 안다
        assert "stop_song" in SONG_BACKEND_INSTRUCTIONS


class TestStoryInstructions:
    def test_story_delegation_lines_sit_inside_the_policy_lists(self) -> None:
        text = build_live_instructions()
        tools_block = text[text.index("Backend tools:") : text.index("Delegate to the backend when:")]
        when_block = text[text.index("Delegate to the backend when:") : text.index("Do not delegate")]
        assert "- Stories:" in tools_block
        assert "tell a story" in when_block

    def test_default_backend_orders_search_before_start(self) -> None:
        task = DEFAULT_BACKEND_INSTRUCTIONS[DEFAULT_BACKEND_INSTRUCTIONS.index("## Task instructions") :]
        assert task.index("search_stories") < task.index("start_story")

    def test_story_backend_embeds_script_and_song(self) -> None:
        text = build_story_backend_instructions("## 단락 1 — 훅\n첫 문장.\n", "snowman", "눈사람 — 정승환")
        assert text.index("## Script") < text.index("## 단락 1 — 훅") < text.index("## Task instructions")
        assert 'play_song with song id "snowman" (눈사람 — 정승환)' in text
        assert "finish_story" in text

    def test_story_mode_append_stays_short(self) -> None:
        assert len(STORY_MODE_INSTRUCTIONS) < 1500  # append 한도 500 토큰 안에 들도록
