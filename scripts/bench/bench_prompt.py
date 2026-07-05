"""Prompt style benchmark — test LLM output naturalness.

Runs the production prompt through the real LLM with stubbed
TTS/Bridge/ASR/VAP/TurnGPT.  Prints raw LLM output for each
test input so you can evaluate conversational quality.

Usage::

    uv run python -m scripts.bench.bench_prompt

Optionally override the system prompt to compare::

    uv run python -m scripts.bench.bench_prompt --prompt "You are Ray, ..."
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from scripts.sandbox import (
    CaptureTTS,
    FakeTurnGPT,
    FakeVAP,
    ObservableLLM,
    SandboxBridge,
    ScriptedASR,
    run_pipeline,
    setup_sandbox,
)
from voice_pipeline.llm.llm import OpenAILLM

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

# (label, history_turns, user_input)
SCENARIOS_KO: list[tuple[str, list[tuple[str, str]], str]] = [
    (
        "ko-greeting",
        [],
        "안녕 레이!",
    ),
    (
        "ko-simple question",
        [],
        "세계에서 제일 높은 산이 뭐야?",
    ),
    (
        "ko-follow-up",
        [
            ("user", "어제 인터스텔라 봤어."),
            ("assistant", "오 진짜? 어땠어?"),
        ],
        "도킹 장면 진짜 미쳤어.",
    ),
    (
        "ko-opinion",
        [],
        "파인애플 피자 어떻게 생각해?",
    ),
    (
        "ko-emotional",
        [],
        "오늘 회사에서 진짜 힘든 하루였어.",
    ),
    (
        "ko-ambiguous",
        [
            ("user", "새로운 취미를 시작해볼까 생각 중이야."),
            ("assistant", "좋은데! 어떤 거에 관심 있어?"),
        ],
        "음, 잘 모르겠어.",
    ),
    (
        "ko-deep",
        [
            ("user", "AI가 대부분의 직업을 대체할 거라고 생각해?"),
            ("assistant", "많은 직업이 바뀌긴 하겠지만, 사람만이 할 수 있는 게 있을 거야."),
            ("user", "예를 들면?"),
            ("assistant", "공감이나 창의성 같은, 인간적인 부분들."),
        ],
        "근데 AI도 공감을 배울 수 있지 않을까?",
    ),
    (
        "ko-list request",
        [],
        "볼 만한 영화 다섯 개만 추천해줘.",
    ),
    (
        "ko-recipe",
        [],
        "스크램블 에그 어떻게 만들어?",
    ),
    (
        "ko-tips",
        [],
        "잠을 잘 자려면 어떻게 해야 돼?",
    ),
]

# -- TTS edge cases: numbers, abbreviations, symbols, URLs, etc. --
SCENARIOS_TTS: list[tuple[str, list[tuple[str, str]], str]] = [
    (
        "tts-large numbers",
        [],
        "How far is the sun from Earth?",
    ),
    (
        "tts-percentages",
        [],
        "What's the unemployment rate in the US right now?",
    ),
    (
        "tts-abbreviations",
        [],
        "What does NASA stand for?",
    ),
    (
        "tts-dates",
        [],
        "When did World War II end?",
    ),
    (
        "tts-currency",
        [],
        "How much does a Tesla Model 3 cost?",
    ),
    (
        "tts-math",
        [],
        "What's the square root of 144?",
    ),
    (
        "tts-units",
        [],
        "How hot is the surface of the sun?",
    ),
    (
        "tts-acronyms in context",
        [],
        "Can you explain what an API is?",
    ),
    (
        "tts-mixed numbers and text",
        [
            ("user", "I'm training for a marathon."),
            ("assistant", "That's awesome! How's it going?"),
        ],
        "I ran 5k yesterday in 23 minutes.",
    ),
    (
        "tts-url and email",
        [],
        "How do I sign up for ChatGPT?",
    ),
]

# -- Web search scenarios: queries that trigger web_search tool --
SCENARIOS_SEARCH: list[tuple[str, list[tuple[str, str]], str]] = [
    (
        "search-today weather",
        [],
        "오늘 서울 날씨 어때?",
    ),
    (
        "search-recent news",
        [],
        "요즘 가장 핫한 뉴스가 뭐야?",
    ),
    (
        "search-stock price",
        [],
        "애플 주가 지금 얼마야?",
    ),
    (
        "search-sports result",
        [],
        "어제 프리미어리그 경기 결과 알려줘.",
    ),
    (
        "search-release date",
        [],
        "GTA 6 출시일이 언제야?",
    ),
    (
        "search-product info",
        [],
        "아이폰 17 스펙이 어떻게 돼?",
    ),
    (
        "search-event",
        [],
        "다음 올림픽 어디서 해?",
    ),
    (
        "search-comparison",
        [],
        "ChatGPT랑 Gemini 중에 뭐가 더 좋아?",
    ),
]

SCENARIOS: list[tuple[str, list[tuple[str, str]], str]] = [
    (
        "cold-open greeting",
        [],
        "Hey Ray!",
    ),
    (
        "simple question",
        [],
        "What's the tallest mountain in the world?",
    ),
    (
        "follow-up in context",
        [
            ("user", "I watched Interstellar last night."),
            ("assistant", "Oh nice! How did you like it?"),
        ],
        "The docking scene was insane.",
    ),
    (
        "opinion request",
        [],
        "Do you think pineapple belongs on pizza?",
    ),
    (
        "emotional support",
        [],
        "I had a really rough day at work today.",
    ),
    (
        "short ambiguous input",
        [
            ("user", "I'm thinking about getting a new hobby."),
            ("assistant", "That sounds fun! What kind of thing are you into?"),
        ],
        "Hmm, I dunno.",
    ),
    (
        "multi-turn deep conversation",
        [
            ("user", "Do you think AI will replace most jobs?"),
            ("assistant", "Some jobs will change a lot, but I think humans will always bring something unique."),
            ("user", "Like what?"),
            ("assistant", "Empathy, creativity, the messy stuff that makes us human."),
        ],
        "But couldn't AI learn empathy too?",
    ),
    (
        "playful/humor",
        [],
        "Tell me something weird.",
    ),
    # -- Stress tests: scenarios that tempt list/markdown/numbering --
    (
        "explicit list request",
        [],
        "Give me a list of five good movies to watch.",
    ),
    (
        "recipe request",
        [],
        "How do I make scrambled eggs?",
    ),
    (
        "compare two things",
        [],
        "What's the difference between Python and JavaScript?",
    ),
    (
        "step-by-step request",
        [],
        "How do I change a flat tire?",
    ),
    (
        "pros and cons",
        [],
        "What are the pros and cons of working from home?",
    ),
    (
        "multiple tips",
        [],
        "Any tips for getting better sleep?",
    ),
    (
        "ranking request",
        [],
        "What are the top three programming languages to learn right now?",
    ),
    (
        "explain with examples",
        [],
        "Can you explain what machine learning is?",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt style benchmark")
    parser.add_argument("--prompt", type=str, default=None, help="Override system prompt")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model (default: config default)")
    parser.add_argument(
        "--tools",
        type=str,
        nargs="*",
        default=None,
        help="LLM tools to enable (e.g. --tools web_search). Default: no tools",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        default=None,
        help="Run only these scenario labels (substring match)",
    )
    args = parser.parse_args()

    all_scenarios = SCENARIOS + SCENARIOS_KO + SCENARIOS_TTS + SCENARIOS_SEARCH
    selected = all_scenarios
    if args.scenarios:
        selected = [s for s in all_scenarios if any(filt in s[0] for filt in args.scenarios)]
        if not selected:
            print(f"No scenarios matched: {args.scenarios}")
            sys.exit(1)

    tools: list[str] = args.tools if args.tools is not None else []
    llm_kwargs: dict[str, Any] = {"tools": tools}
    if args.model:
        llm_kwargs["model"] = args.model

    print("=" * 60)
    print("PROMPT STYLE BENCHMARK")
    print("=" * 60)

    prompt_text = args.prompt
    if prompt_text:
        print(f"\n[Custom prompt]\n{prompt_text}\n")
    else:
        from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT

        print(f"\n[Default prompt]\n{DEFAULT_SYSTEM_PROMPT}\n")
        prompt_text = None  # use default

    print(f"[tools] {tools or '(none)'}")
    print("=" * 60)

    for label, history, user_input in selected:
        llm = ObservableLLM(OpenAILLM(**llm_kwargs))
        setup = setup_sandbox(
            asr=ScriptedASR([]),
            vap=FakeVAP(),
            turngpt=FakeTurnGPT(),
            tts=CaptureTTS(),
            bridge=SandboxBridge(),
            llm=llm,
            history_turns=history or None,
            system_prompt=prompt_text,
        )
        try:
            t0 = time.monotonic()
            result = run_pipeline(setup, user_input)
            elapsed = time.monotonic() - t0

            print(f"\n--- {label} ---")
            if history:
                print("[history]")
                for role, text in history:
                    print(f"  {role}: {text}")
            print(f"[user] {user_input}")
            print(f"[ray]  {result.clean_text}")
            if result.raw_llm_output != result.clean_text:
                print(f"[raw]  {result.raw_llm_output}")
            print(f"[time] {elapsed:.1f}s")
            if result.error:
                print(f"[error] {result.error}")
        finally:
            setup.cleanup()

    print("\n" + "=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
