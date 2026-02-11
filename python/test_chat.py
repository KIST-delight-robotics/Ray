# -*- coding: utf-8 -*-
"""
RAG 채팅 테스트 스크립트 (CLI Ver.)
state_manager.py의 로직(도구, 프롬프트, 흐름)을 그대로 모사하여 테스트합니다.
(단, 음성/음악 재생 관련 기능은 제외)
"""

import os
import json
import sys
import time

# 환경변수 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    print("PowerShell: $env:OPENAI_API_KEY = 'your-key'")
    sys.exit(1)

from openai import OpenAI
from rag import init_db, search_archive
from rag.retriever import search_archive_debug
from config import RAG_PERSIST_DIR, RAG_TOP_K, RESPONSES_MODEL, RESPONSES_PRESETS
from prompts import SYSTEM_PROMPT_V0_1, SYSTEM_PROMPT_V0_2
from conversation_manager import ConversationManager

# 시스템 프롬프트 선택
SYSTEM_PROMPT = SYSTEM_PROMPT_V0_2

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# RAG DB 초기화
# print("📚 RAG DB 초기화 중...")
# init_db(str(RAG_PERSIST_DIR), OPENAI_API_KEY)
# print("✅ RAG DB 준비 완료!")

# Conversation Manager 초기화
cm = ConversationManager(openai_api_key=OPENAI_API_KEY)
cm.start_new_session(system_prompt=SYSTEM_PROMPT)
print(f"✅ 세션 시작됨 (System Prompt: SYSTEM_PROMPT)")


# 도구 정의 (state_manager.py의 LLMManager와 동일하게 구성, play_music 제외)
tools = [
    {
        "type": "web_search",
        "user_location": {"type": "approximate", "country": "KR"},
    },
    # {
    #     "type": "function",
    #     "name": "consult_archive",
    #     "description": "영화/음악에 대한 정보를 찾거나, 사용자의 기분/상황에 맞는 작품을 연상할 때 사용합니다. 사실 확인, 위로, 공감, 추천이 필요할 때 적극적으로 사용하세요.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "query": {
    #                 "type": "string",
    #                 "description": "검색할 키워드 또는 문장 (예: '비 오는 날의 우울함', '헤어질 결심 해석')"
    #             },
    #             "intent": {
    #                 "type": "string",
    #                 "enum": ["fact", "vibe", "critique"],
    #                 "description": "fact=사실정보(감독/출연진), vibe=분위기/추천, critique=평론/해석"
    #             }
    #         },
    #         "required": ["query", "intent"]
    #     }
    # }
]


def chat(user_input: str) -> str:
    """사용자 입력을 받아 RAG 기반 응답 생성 (state_manager.py 로직 모사)"""
    
    # 1. 사용자 메시지 기록
    cm.add_message({"role": "user", "content": user_input, "type": "message"})
    current_log = cm.get_current_log()
    
    # 2. 1차 API 호출
    params = {
        **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
        "input": current_log,
        "tools": tools,
    }
    
    try:
        response = client.responses.create(**params)
    except Exception as e:
        return f"API 호출 오류: {e}"
    
    final_text = ""
    
    # 3. 결과 처리 루프
    for item in response.output:
        if item.type == "message":
            final_text = item.content[0].text.strip()
            break
            
        elif item.type == "function_call":
            print(f"\n🧠 Function call: {item.name}")
            
            if item.name == "consult_archive":
                args = json.loads(item.arguments)
                query = args.get("query", "")
                intent = args.get("intent", "vibe")
                
                print(f"🔍 RAG 검색: query='{query}', intent='{intent}'")
                
                # RAG 검색 (디버그용 상세 출력)
                docs_info, search_result = search_archive_debug(query, intent, top_k=RAG_TOP_K)
                
                # 검색된 문서 개별 출력
                print("─" * 50)
                print(f"📄 검색된 문서 ({len(docs_info)}개):")
                for doc in docs_info:
                    print(f"\n  [{doc['index']}] {doc['title']} ({doc['category']})")
                    print(f"      ID: {doc['movie_id']}")
                    print(f"      내용: {doc['content_preview']}")
                print("─" * 50)
                
                # 휘발성 기억 패턴 - temp_log 사용
                # (도구 호출 및 결과는 영구 저장소인 cm에 넣지 않고, 이번 턴의 컨텍스트로만 사용)
                temp_log = current_log.copy()
                temp_log.append({
                    "type": "function_call",
                    "name": item.name,
                    "call_id": item.call_id,
                    "arguments": item.arguments
                })
                temp_log.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": search_result
                })
                
                # 2차 API 호출 (검색 결과 포함)
                params_with_context = {
                    **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
                    "input": temp_log,
                    "tools": tools,
                }
                response_2 = client.responses.create(**params_with_context)
                
                if response_2.output:
                    for resp_item in response_2.output:
                        if resp_item.type == "message" and resp_item.content:
                            final_text = resp_item.content[0].text.strip()
                            break
                break
            
            elif item.name == "web_search":
                print("🌐 Web Search 호출됨 (내부 처리)")
                pass

    # 4. 응답 저장 (메시지만 저장)
    if final_text:
        cm.add_message({"role": "assistant", "content": final_text, "type": "message"})
    
    return final_text


def main():
    print("\n" + "="*50)
    print("채팅 테스트 (System Logic Synced)")
    print("="*50)
    print("대화를 시작하세요! (종료: 'quit' 또는 'q')\n")
    print("System Prompt Length:", len(SYSTEM_PROMPT))
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'q', '종료']:
                print("👋 안녕히 가세요!")
                cm.end_session()
                break
            
            response = chat(user_input)
            print(f"\nRay: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            cm.end_session()
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()