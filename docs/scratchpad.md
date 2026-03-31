# Phase 4 (Integration) 사전 메모

## Phase 3에서 넘어온 사항

- **처리 상태 추적**: `process_session`이 실패하면 로그만 남기고 빈 리스트 반환. 세션이 "처리됨"으로 마킹되는 메커니즘이 없어 재시도와 중복 처리 구분 불가. Phase 4에서 설계 필요.
- **utterance 저장 연결**: orchestrator에서 `add_utterance()` 호출 시 `token_count`를 함께 전달해야 함.
- **임베딩 인스턴스 공유**: similarity와 memory가 같은 모델이면 하나만 생성해서 주입. 와이어링 시 결정.
- **에피소드 = 세션 요약**: `process_session()`이 반환하는 에피소드 리스트를 히스토리 블록 3의 세션 요약으로 사용. 포맷 변환이 필요한지 확인.
- **임베딩 실패 복구**: 에피소드가 embedding=None으로 저장된 경우, 다음 기회에 재생성할 메커니즘 검토.
- **서브토픽 중복**: LLM이 유사한 서브토픽을 다른 이름으로 만들 수 있음 (movie vs movies). 실사용 데이터에서 문제 빈도 확인 후 대응.
