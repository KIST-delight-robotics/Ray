# -*- coding: utf-8 -*-
"""
LLM 대화 수행 능력 테스트 러너 (Adaptive Mode)
테스터 LLM이 Ray의 응답에 따라 후속 질문을 동적으로 생성합니다.

사용법:
  전체 시나리오:
    python test_conversation_log.py

  특정 시나리오:
    python test_conversation_log.py --ids 1-1,2-1

  대화형 모드:
    python test_conversation_log.py --interactive

  모델 변경:
    python test_conversation_log.py --model gpt-5-mini
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# 환경변수 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    print("PowerShell: $env:OPENAI_API_KEY = 'your-key'")
    sys.exit(1)

from openai import OpenAI

# 상위 디렉토리(python/) 모듈 임포트를 위한 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RESPONSES_PRESETS, RESPONSES_MODEL
from llm.prompts import SYSTEM_PROMPT_V0_1
from engine.session import ConversationManager
from test_scenarios import TEST_SCENARIOS

# ─── 설정 ─────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT_NAME = "SYSTEM_PROMPT_V0_1"
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT_V0_1
DEFAULT_MODEL = RESPONSES_MODEL
TESTER_MODEL = "gpt-4.1-mini"  # 테스터 LLM 모델

TOOLS = [
    {
        "type": "web_search",
        "user_location": {"type": "approximate", "country": "KR"},
    },
]


# ─── 테스터 LLM (Adaptive Questioning) ──────────────────────────────────────

TESTER_SYSTEM_PROMPT = """\
너는 AI 대화 로봇의 영화 지식과 대화 능력을 테스트하는 평가자이다.
대화 상대(Ray)의 응답을 분석하고, 다음에 할 질문을 생성해야 한다.

## 핵심 원칙
1. 절대로 답을 미리 알려주지 마라. "칸 영화제에서 상 받았지?" 같은 질문은 금지. 대신 "그 영화 상 받은 적 있어?"처럼 물어라.
2. Ray가 정확하게 답하면 더 깊은 후속 질문을 해라.
3. Ray가 부정확하거나 모호하게 답하면 구체적으로 확인해라.
4. Ray가 모른다고 하면 자연스럽게 다른 관련 주제로 넘어가라.
5. 자연스러운 구어체 한국어로 질문해라. 시험관처럼 딱딱하게 하지 마라.
6. 질문만 출력해라. 설명이나 분석은 하지 마라.
"""


def generate_next_question(client: OpenAI, scenario: dict, conversation_so_far: list) -> str:
    """테스터 LLM을 사용해 다음 질문을 동적으로 생성합니다."""

    # 대화 이력을 텍스트로 구성
    conv_text = ""
    for turn in conversation_so_far:
        conv_text += f"User: {turn['user']}\n"
        conv_text += f"Ray: {turn['ray']}\n\n"

    knowledge_str = "\n".join(f"- {k}" for k in scenario.get("knowledge_to_verify", []))

    user_prompt = f"""## 테스트 시나리오
이름: {scenario['name']}
목표: {scenario['objective']}

## 확인해야 할 지식
{knowledge_str}

## 지금까지의 대화
{conv_text}

## 지시
위 대화를 보고, 다음에 할 자연스러운 후속 질문을 하나만 생성해라.
답을 미리 알려주지 말고, Ray가 스스로 아는지 확인하는 방식으로 물어라."""

    try:
        response = client.responses.create(
            model=TESTER_MODEL,
            input=[
                {"role": "system", "content": TESTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        for item in response.output:
            if item.type == "message":
                return item.content[0].text.strip()
    except Exception as e:
        return f"[TESTER_ERROR] {e}"

    return "그렇구나. 좀 더 자세히 알려줄 수 있어?"


# ─── Ray 대화 호출 ──────────────────────────────────────────────────────────

def chat_turn(client: OpenAI, cm: ConversationManager, user_input: str, model: str) -> str:
    """한 턴의 대화를 수행합니다."""
    cm.add_message({"role": "user", "content": user_input, "type": "message"})
    current_log = cm.get_current_log()

    params = {
        **RESPONSES_PRESETS.get(model, {}),
        "input": current_log,
        "tools": TOOLS,
    }

    try:
        response = client.responses.create(**params)
    except Exception as e:
        return f"[API_ERROR] {e}"

    final_text = ""
    for item in response.output:
        if item.type == "message":
            final_text = item.content[0].text.strip()
            break

    if final_text:
        cm.add_message({"role": "assistant", "content": final_text, "type": "message"})

    return final_text


# ─── 시나리오 실행 ──────────────────────────────────────────────────────────

def run_scenario(client: OpenAI, scenario: dict, prompt_name: str, system_prompt: str, model: str) -> dict:
    """Adaptive 모드로 하나의 시나리오를 실행합니다."""
    sid = scenario["id"]
    name = scenario["name"]
    num_turns = scenario.get("num_turns", 5)

    print(f"\n{'═' * 60}")
    print(f"  [{sid}] {name}")
    print(f"  카테고리: {scenario['category']}")
    print(f"  턴 수: {num_turns}")
    print(f"{'═' * 60}")

    cm = ConversationManager(openai_api_key=OPENAI_API_KEY)
    cm.start_new_session(system_prompt=system_prompt)

    conversation = []

    for turn_num in range(1, num_turns + 1):
        # 첫 턴은 시나리오의 first_message, 이후는 테스터 LLM이 생성
        if turn_num == 1:
            user_text = scenario["first_message"]
        else:
            print(f"  🧪 테스터가 다음 질문 생성 중...")
            user_text = generate_next_question(client, scenario, conversation)

        print(f"\n  [{turn_num}/{num_turns}] User: {user_text}")
        start_t = time.time()
        reply = chat_turn(client, cm, user_text, model)
        elapsed = time.time() - start_t
        print(f"           Ray:  {reply}")
        print(f"           ({elapsed:.2f}s)")

        conversation.append({
            "turn": turn_num,
            "user": user_text,
            "ray": reply,
            "response_time_sec": round(elapsed, 2),
        })

    return {
        "scenario_id": sid,
        "scenario_name": name,
        "category": scenario["category"],
        "check_points": scenario["objective"],
        "knowledge_to_verify": scenario.get("knowledge_to_verify", []),
        "conversation": conversation,
    }


def run_interactive(client: OpenAI, prompt_name: str, system_prompt: str, model: str) -> dict:
    """대화형 모드"""
    print(f"\n{'═' * 60}")
    print(f"  대화형 모드 (종료: 'quit' 또는 'q')")
    print(f"  모델: {model}  |  프롬프트: {prompt_name}")
    print(f"{'═' * 60}\n")

    cm = ConversationManager(openai_api_key=OPENAI_API_KEY)
    cm.start_new_session(system_prompt=system_prompt)

    conversation = []
    turn_num = 0

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "q", "종료"]:
                break

            turn_num += 1
            start_t = time.time()
            reply = chat_turn(client, cm, user_input, model)
            elapsed = time.time() - start_t

            print(f"Ray:  {reply}")
            print(f"({elapsed:.2f}s)\n")

            conversation.append({
                "turn": turn_num,
                "user": user_input,
                "ray": reply,
                "response_time_sec": round(elapsed, 2),
            })
        except KeyboardInterrupt:
            break

    # 대화형 모드에서 시나리오 이름 입력 받기
    scenario_name = input("\n시나리오 이름 (엔터로 건너뛰기): ").strip()
    if not scenario_name:
        scenario_name = "대화형 모드"

    return {
        "scenario_id": "interactive",
        "scenario_name": scenario_name,
        "category": "수동 테스트",
        "check_points": "수동 확인",
        "knowledge_to_verify": [],
        "conversation": conversation,
    }


# ─── 출력 / 저장 ────────────────────────────────────────────────────────────

def build_metadata(prompt_name: str, model: str) -> dict:
    tool_names = [t.get("type") or t.get("name", "unknown") for t in TOOLS]
    return {
        "prompt_name": prompt_name,
        "model": model,
        "model_presets": RESPONSES_PRESETS.get(model, {}),
        "tools": tool_names,
        "timestamp": datetime.now().isoformat(),
    }


def create_run_dir(base_dir: str, model: str) -> str:
    """테스트 런별 폴더를 생성합니다. 예: 20260212_0932_gpt4.1mini/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_short = model.replace("-", "").replace(".", "")
    run_name = f"{timestamp}_{model_short}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_results(results: list, metadata: dict, run_dir: str) -> str:
    """테스트 결과를 JSON으로 저장합니다."""
    output_data = {
        "metadata": metadata,
        "results": results,
    }

    filepath = os.path.join(run_dir, "results.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장: {filepath}")
    return filepath


# ─── 메인 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM 대화 수행 능력 테스트 러너 (Adaptive)")
    parser.add_argument("--ids", type=str, default=None,
                        help="실행할 시나리오 ID (콤마 구분, 예: 1-1,2-1)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"사용할 모델 (기본: {DEFAULT_MODEL})")
    parser.add_argument("--interactive", action="store_true",
                        help="대화형 모드로 실행")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT_NAME,
                        help="프롬프트 이름 (기본: SYSTEM_PROMPT_V0_1)")
    args = parser.parse_args()

    model = args.model
    prompt_name = args.prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT

    client = OpenAI(api_key=OPENAI_API_KEY)
    metadata = build_metadata(prompt_name, model)

    # 테스트 런 폴더 생성
    base_output = os.path.join(os.path.dirname(__file__), "..", "..", "output", "test_logs")
    run_dir = create_run_dir(base_output, model)

    print(f"{'═' * 60}")
    print(f"  LLM 대화 수행 능력 테스트 (Adaptive)")
    print(f"  프롬프트: {prompt_name}")
    print(f"  모델:     {model}")
    print(f"  도구:     {', '.join(metadata['tools'])}")
    print(f"  출력:     {run_dir}")
    print(f"{'═' * 60}")

    results = []

    if args.interactive:
        result = run_interactive(client, prompt_name, system_prompt, model)
        if result["conversation"]:
            results.append(result)
    else:
        if args.ids:
            target_ids = [s.strip() for s in args.ids.split(",")]
            scenarios = [s for s in TEST_SCENARIOS if s["id"] in target_ids]
            if not scenarios:
                print(f"⚠️ 해당 ID의 시나리오를 찾을 수 없습니다: {args.ids}")
                sys.exit(1)
        else:
            scenarios = TEST_SCENARIOS

        print(f"\n  실행 대상: {len(scenarios)}개 시나리오 (Adaptive 모드)")

        for scenario in scenarios:
            result = run_scenario(client, scenario, prompt_name, system_prompt, model)
            results.append(result)

    if results:
        filepath = save_results(results, metadata, run_dir)
        print(f"\n✅ 테스트 완료! 총 {len(results)}개 시나리오")
        print(f"📁 결과 폴더: {run_dir}")
        print(f"📋 PPT 변환:  python format_for_ppt.py \"{filepath}\" --style ppt -o \"{run_dir}\"")


if __name__ == "__main__":
    main()
