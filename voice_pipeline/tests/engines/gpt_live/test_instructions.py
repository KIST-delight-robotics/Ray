"""Tests for voice_pipeline.engines.gpt_live.instructions."""

from __future__ import annotations

from voice_pipeline.engines.gpt_live.instructions import (
    DEFAULT_BACKEND_INSTRUCTIONS,
    SONG_BACKEND_INSTRUCTIONS,
    STORY_MODE_INSTRUCTIONS,
    build_backend_instructions,
    build_live_instructions,
    build_story_backend_instructions,
    build_voice_section,
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

    def test_default_backend_rewrites_first_part_in_rays_voice(self) -> None:
        assert "first_part rewritten in Ray's voice" in DEFAULT_BACKEND_INSTRUCTIONS

    def test_voice_section_default_and_profile_override(self) -> None:
        plain = build_voice_section()
        assert plain.startswith("## Ray's voice") and "해요체" in plain and "overrides" not in plain
        with_pref = build_voice_section("prefers formal tone")
        assert "overrides the default: prefers formal tone" in with_pref

    def test_backend_builder_appends_voice_section(self) -> None:
        text = build_backend_instructions("formal")
        assert text.startswith(DEFAULT_BACKEND_INSTRUCTIONS) and text.rstrip().endswith("formal")

    def test_story_mode_hands_pacing_and_interruption_to_the_model(self) -> None:
        assert "in full" not in STORY_MODE_INSTRUCTIONS
        # 끊긴 이야기는 계속할지 물은 뒤 위임 없이 이어 말하고, 답에는 스스로 반응하고 군말 없이 위임
        assert "ask whether they want you to continue" in STORY_MODE_INSTRUCTIONS
        assert "stop and talk with them" in STORY_MODE_INSTRUCTIONS and "잠깐만요" in STORY_MODE_INSTRUCTIONS
        assert "continuously" not in STORY_MODE_INSTRUCTIONS  # 끼어들기를 무시하게 만들었던 표현 (실기 2026-09-28)

    def test_story_backend_embeds_script_and_song(self) -> None:
        text = build_story_backend_instructions("## 단락 1 — 훅\n첫 문장.\n", "snowman", "눈사람 — 정승환", "formal")
        assert text.index("## Script") < text.index("## 단락 1 — 훅") < text.index("## What to hand over")
        assert 'play_song with song id "snowman" (눈사람 — 정승환)' in text
        assert "finish_story" in text
        assert text.index("## Return the result") < text.index("## Ray's voice") and "formal" in text
        assert "Rewrite it in Ray's voice" in text
        assert text.index("## Deciding whether the story goes on") < text.index("## Return the result")
        assert "Call finish_story only when" in text
        # 다음 질문 지점까지 통째로 넘기고, 사용자에게 답하지 않는다 (반응은 대화 모델 몫)
        assert "up to and including the next stop" in text and "Never call it before Ray has told the bridge" in text
        assert "never answer the user yourself" in text

    def test_story_mode_append_stays_within_append_limit(self) -> None:
        import tiktoken

        tokens = len(tiktoken.get_encoding("o200k_base").encode(STORY_MODE_INSTRUCTIONS))
        assert tokens <= 450  # instructions.append 한도 500 토큰, 토크나이저 차이를 감안한 여유
