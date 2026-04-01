# 다음 세션 핸드오프

## 현재 상태

### 장기기억 로드맵 (`docs/memory-roadmap.md`) 진행 상황

- **Phase 1 (Storage)**: 완료
- **Phase 2 (Read)**: 완료
- **Phase 3 (Write)**: 완료
- **Phase 4 (Integration)**: 로드맵 정의 범위 전부 완료
  - ContextBuilder 4블록 재작성, 기억 인용 파싱, 세션 라이프사이클 훅, 프롬프트 업데이트
  - 추가: utterance 저장, threading.Lock, 이전 세션 요약 = 에피소드 직접 사용

### 로드맵 외 추가 작업 (미커밋)

**Sentence Streaming** (`pipeline_mode="sentence"`):
- `SentenceDetector` — 영어 문장 경계 감지, 약어 처리, min_flush_words
- `_run_pipeline_sentence` — producer-consumer 스레딩, TTS 파이프라이닝
- `_sentence_tts_consumer` — 순서 보장 TTS drain, `_text` 누적 갱신
- 테스트 45개 통과 (기존 31 + 신규 14), 전체 847 통과

### 미커밋 파일

이번 세션:
- `voice_pipeline/generation/sentence_detector.py` (신규)
- `voice_pipeline/tests/generation/test_sentence_detector.py` (신규)
- `voice_pipeline/core/config.py` (min_flush_words 추가)
- `voice_pipeline/generation/speech_generator.py` (dispatch + sentence pipeline)
- `voice_pipeline/tests/generation/test_speech_generator.py` (sentence 모드 테스트)
- `docs/decisions-wip.md` (Phase 4-5 결정 추가)

이전 세션 (출처 미확인):
- `docs/ray-memory/02-session.md`, `03-read.md`, `04-write.md`, `05-storage.md` — 설계 문서를 구현과 동기화한 것으로 보임
- `cpp/config.toml`
- `.claude/settings.json`


## 다음 세션에서 해야 할 일

### 1. 미커밋 변경 커밋 정리

미커밋 파일들을 검토하고 적절히 커밋. 특히 `docs/ray-memory/` 변경은 이전 세션 것인지 확인 필요.

### 2. 남은 작업 재계획

로드맵 Phase 4까지 완료. 이후 작업은 plan 문서가 소실되어 재계획 필요.

**계획 시 참고:**
- `docs/memory-roadmap.md` — Phase 1~4 정의 (Phase 5 이후 없음)
- `docs/ray-memory/01~05` — 설계 문서 (구현과 차이 있을 수 있음, 대조 필요)
- `docs/decisions-wip.md` — Phase 1~4-5 결정 기록 (확정분 decisions.md 이관 검토)

**알려진 미해결 항목** (Phase 4-4에서 넘어온 것):
- 처리 상태 추적: `process_session` 실패 시 재시도/중복 처리 구분 불가
- 임베딩 실패 복구: `embedding=None`으로 저장된 에피소드 재생성 메커니즘
- 서브토픽 중복: LLM이 유사 서브토픽을 다른 이름으로 생성 (movie vs movies)
- `write_executor.shutdown(wait=True)` 블로킹: 종료 시 LLM 호출 대기
- `started_at` 타임스탬프 미세 불일치: SessionManager/ConversationHistory 간 수 ms 차이

**Summarization 관련**: 현재 "에피소드 = 세션 요약"으로 별도 LLM 요약 호출 없이 운용 중. 에피소드가 `max_prev_session_tokens`(512)를 초과하는 사례가 실사용에서 발생하면 재검토.
