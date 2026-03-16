# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

596 tests pass, ruff 0 errors. 첫 전체 파이프라인 실행 완료.

### 미해결 사항

**배치 내 상태 전이 누락** (실측 후 판단)
- 배치 N프레임에서 VAP 결과 1개로 타이머를 N*30ms 증분 → 배치 중간 발화 상태 변화 반영 안 됨
- VAP 추론 주기(100ms ≈ 3프레임)이고 배치가 주로 2-3프레임이므로 영향 제한적

**벤치마크 결과 (Ryzen 5 5600X)**:
- VAP 10Hz: mean 7.5ms, P99 11.3ms (concurrent). Budget 100ms 대비 89% headroom.
- TurnGPT 3Hz: mean 8.5ms, P99 12.0ms (concurrent). Budget 333ms 대비 96% headroom.
- Similarity (all-MiniLM-L6-v2, torch): ~4ms/호출, 모델 로드 ~2.7s
- **RPi 5에서 재측정 필요** — 데스크탑 수치와 큰 차이 예상.

### 해결된 사항 (이전 세션)
- interrupt에 user_is_speaking 전제조건 누락 → 추가 (논문 준수)
- mock 서버 double playback_complete → 수정
- STOP_PENDING watchdog에서 turn_detector.reset() 누락 → 추가
- 유사도 검사 SequenceMatcher → sentence embedding (의미 유사도)
- 파일 저장 구현 (FileStorageBackend)
- 로깅 체계 개선

### 해결된 사항 (이전 세션)
- CppBridge 재연결: `_run_greeting()`에서 `connect()` 호출 → C++ 크래시 후 자동 복구
- Audio starvation timeout: Orchestrator에서 5초간 프레임 미도착 시 세션 종료
- LED 테스트 16개 실패 → `rpi5-ws2812` optional 이동 + CppBridgeConfig 기본값 수정으로 해결

### 해결된 사항 (이전 세션)
- GitHub Actions CI 구축: lint (uvx ruff) + test (uv run pytest), push/PR 자동 실행
- Claude Code pre-commit hook: git commit 시 ruff + pytest 자동 실행, 실패 시 커밋 차단

### 해결된 사항 (이번 세션)
- `no_robot_audio` 무조건 interrupt 제거: robot_audio=None 시 TurnDecision.none() 반환. 논문에 없는 분기였으며 VAP가 로봇 채널 없이 interrupt/backchannel 구분 불가.
- awaiting 중 ASR 텍스트 변경 cancel: turn_shift 후 0.5초 grace 이후 ASR 텍스트 변경 시 generator cancel + USER_TURN 복귀. 유저 추가 발화 처리.
- integration test barge-in 수정: ScriptedBridge.send_audio_end()의 즉시 PLAYBACK_COMPLETE가 실제 C++ 동작과 불일치 → InterruptBridge에서 오버라이드.

### Next
- Phase 7 — Integration tests (Python ↔ C++ 실제 연결 테스트)
- RPi 5에서 벤치마크 재실행 (similarity 모델 포함)
- 이전 세션 대화 내역 load/활용 기능
- TurnGPT: dialog 누적 시 비동기 cache prefill (선택적 최적화)
