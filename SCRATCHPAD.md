# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

568 tests pass, ruff 0 errors. 첫 전체 파이프라인 실행 완료.

### 미해결 사항

**배치 내 상태 전이 누락** (실측 후 판단)
- 배치 N프레임에서 VAP 결과 1개로 타이머를 N*30ms 증분 → 배치 중간 발화 상태 변화 반영 안 됨
- VAP 추론 주기(100ms ≈ 3프레임)이고 배치가 주로 2-3프레임이므로 영향 제한적

**awaiting_response 중 `no_robot_audio` interrupt 반복**
- awaiting_response 동안 robot_audio=None → user_is_speaking fallback → 주변 소음에 의해 반복 interrupt → generator cancel → re-prepare 사이클
- 로그에서 LLM 7회 호출, 4.7초 지연 관찰됨
- 근본 해결: AEC(에코 캔슬레이션) 또는 awaiting 중 interrupt 억제 정책 필요

**벤치마크 결과 (Ryzen 5 5600X)**:
- VAP 10Hz: mean 7.5ms, P99 11.3ms (concurrent). Budget 100ms 대비 89% headroom.
- TurnGPT 3Hz: mean 8.5ms, P99 12.0ms (concurrent). Budget 333ms 대비 96% headroom.
- Similarity (all-MiniLM-L6-v2, torch): ~4ms/호출, 모델 로드 ~2.7s
- **RPi 5에서 재측정 필요** — 데스크탑 수치와 큰 차이 예상.

### 해결된 사항 (이번 세션)
- interrupt에 user_is_speaking 전제조건 누락 → 추가 (논문 준수)
- mock 서버 double playback_complete → 수정
- STOP_PENDING watchdog에서 turn_detector.reset() 누락 → 추가
- 유사도 검사 SequenceMatcher → sentence embedding (의미 유사도)
- 파일 저장 구현 (FileStorageBackend)
- 로깅 체계 개선

### Next
- awaiting_response 중 false interrupt 문제 해결 방안 검토
- Phase 7 — Integration tests (Python ↔ C++ 실제 연결 테스트)
- RPi 5에서 벤치마크 재실행 (similarity 모델 포함)
- 이전 세션 대화 내역 load/활용 기능
- TurnGPT: dialog 누적 시 비동기 cache prefill (선택적 최적화)
