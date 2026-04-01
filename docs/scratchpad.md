# Phase 4-5 (Sentence Streaming + Summarization) 사전 메모

## Phase 4-4에서 넘어온 사항

- **처리 상태 추적**: `process_session`이 실패하면 로그만 남기고 빈 리스트 반환. 세션이 "처리됨"으로 마킹되는 메커니즘이 없어 재시도와 중복 처리 구분 불가.
- **임베딩 실패 복구**: 에피소드가 embedding=None으로 저장된 경우, 다음 기회에 재생성할 메커니즘 검토.
- **서브토픽 중복**: LLM이 유사한 서브토픽을 다른 이름으로 만들 수 있음 (movie vs movies). 실사용 데이터에서 문제 빈도 확인 후 대응.
- **write_executor.shutdown(wait=True) 블로킹**: 프로세스 종료 시 MemoryWriter가 LLM 호출 중이면 응답 대기 시간만큼 종료 지연. 필요 시 timeout 또는 cancel_futures 검토.
- **started_at 타임스탬프 미세 불일치**: SessionManager와 ConversationHistory가 각각 datetime.now()를 호출하여 수 밀리초 차이 발생. 기능상 무해하나 인지 필요.
