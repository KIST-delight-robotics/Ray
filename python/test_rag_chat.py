# -*- coding: utf-8 -*-
"""
RAG 채팅 테스트 스크립트
간단한 CLI로 RAG + LLM 기능을 테스트합니다.
"""

import os
import json
import sys

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

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# RAG DB 초기화
print("📚 RAG DB 초기화 중...")
init_db(str(RAG_PERSIST_DIR), OPENAI_API_KEY)
print("✅ RAG DB 준비 완료!")

# 대화 로그
conversation_log = [
    {
        "role": "system",
        "content": """당신은 영화와 음악에 해박한 친근한 로봇 'Ray'입니다.
        
consult_archive 툴을 통해 제공된 정보는 당신의 '내면의 지식'입니다.
데이터를 인용할 때 "검색 결과에 따르면"이라고 말하지 말고, 
당신이 직접 알고 있는 것처럼 자연스럽게 말하세요.

예: "평론가가 파도 같대요" (X) → "그 영화, 감정이 파도처럼 밀려오지 않나요?" (O)"""
    }
]

# 도구 정의
tools = [
    {
        "type": "function",
        "name": "consult_archive",
        "description": "영화/음악에 대한 정보를 찾거나, 사용자의 기분/상황에 맞는 작품을 연상할 때 사용합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 키워드 또는 문장"
                },
                "intent": {
                    "type": "string",
                    "enum": ["fact", "vibe", "critique"],
                    "description": "fact=사실정보, vibe=분위기/추천, critique=평론/해석"
                }
            },
            "required": ["query", "intent"]
        }
    }
]


def chat(user_input: str) -> str:
    """사용자 입력을 받아 RAG 기반 응답 생성"""
    
    # 사용자 메시지 추가
    conversation_log.append({"role": "user", "content": user_input})
    
    # 1차 API 호출
    params = {
        **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
        "input": conversation_log,
        "tools": tools,
    }
    response = client.responses.create(**params)
    
    final_text = ""
    
    for item in response.output:
        if item.type == "message":
            final_text = item.content[0].text.strip()
            break
            
        elif item.type == "function_call" and item.name == "consult_archive":
            args = json.loads(item.arguments)
            query = args.get("query", "")
            intent = args.get("intent", "vibe")
            
            print(f"\n🔍 RAG 검색: query='{query}', intent='{intent}'")
            
            # RAG 검색 (디버그 모드)
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
            temp_log = conversation_log.copy()
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
            
            # 2차 API 호출
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
    
    # 응답 저장 (Tool Call/Output은 저장하지 않음 - 휘발성)
    if final_text:
        conversation_log.append({"role": "assistant", "content": final_text})
    
    return final_text


def main():
    print("\n" + "="*50)
    print("🎬 RAG 채팅 테스트")
    print("="*50)
    print("영화에 대해 물어보세요! (종료: 'quit' 또는 'q')\n")
    print("예시 질문:")
    print("  - 헤어질 결심 감독이 누구야?")
    print("  - 비 오는 날 볼만한 영화 추천해줘")
    print("  - 기생충은 어떤 영화야?")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'q', '종료']:
                print("👋 안녕히 가세요!")
                break
            
            response = chat(user_input)
            print(f"\nRay: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
