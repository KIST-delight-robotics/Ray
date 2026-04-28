# Context Module

LLM 입력 context 조립. 4-블록 (system / profile / prev sessions / history + current + memory) 우선순위 기반
토큰 예산 배분.

블록 순서는 prefix cache 최대화를 위해 **고정 블록 먼저, 변동 블록 나중**이다 (`context_builder.py`
docstring 참조).

## Usage

```python
from voice_pipeline.context.context_builder import ContextBuilder

cb = ContextBuilder(
    history,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    token_counter=token_counter,
    tools_token_cost=tools_token_cost,
    profiles=profiles,
    session_summaries=session_summaries,
)
messages = cb.build(current_text, memory_result=memory_result)
```

## `ContextBuilder.__init__` 인자

| 인자 | Default | 의미 |
|---|---|---|
| `history` | (필수) | `IConversationHistory` 구현체. `get_turns()`로 현재 세션 history 조회. |
| `system_prompt` | (필수) | LLM system 메시지 내용. 빈 문자열이면 블록 생략. |
| `token_counter` | (필수) | `Callable[[str], int]` 토큰 카운터. 블록별 token cost 산정. |
| `tools_token_cost` | `0` | Tool 정의가 LLM 요청에 차지하는 고정 토큰 수. 전체 예산에서 선차감. |
| `profiles` | `None` | 세션 내 불변 profile 리스트. 생성 시 pre-format + pre-count. |
| `session_summaries` | `None` | 이전 세션 요약 문자열 리스트 (오래된 순). |

## 클래스 변수

| 변수 | 값 | 의미 |
|---|---|---|
| `_MAX_CONTEXT_TOKENS` | `4096` | LLM 입력 전체 토큰 예산 (모든 블록 합산 상한). |
| `_MAX_MEMORY_TOKENS` | `512` | retrieved memory 블록 전용 예산. 초과 시 낮은 salience 순 drop. |
| `_MAX_PROFILE_TOKENS` | `256` | profile 블록 전용 예산. 초과 시 블록 전체 skip. |
| `_MAX_PREV_SESSION_TOKENS` | `512` | previous session summary 블록 전용 예산. 초과 시 오래된 순 drop. |

## Testing

토큰 예산 테스트는 `monkeypatch.setattr(ContextBuilder, "_MAX_CONTEXT_TOKENS", N)` 등으로 class var 덮어쓰기.
`tests/context/test_context_builder.py`의 `_set_budgets(monkeypatch, ...)` 헬퍼 참조.
