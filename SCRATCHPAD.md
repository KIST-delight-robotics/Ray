# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

543 tests pass, ruff 0 errors.

### 미해결 사항

**배치 내 상태 전이 누락** (실측 후 판단)
- 배치 N프레임에서 VAP 결과 1개로 타이머를 N*30ms 증분 → 배치 중간 발화 상태 변화 반영 안 됨
- VAP 추론 주기(100ms ≈ 3프레임)이고 배치가 주로 2-3프레임이므로 영향 제한적
- **실제 파이프라인 실행해서 조기 turn_shift 발생 여부 확인 필요**

**VAP 추론을 별도 스레드로 분리 필요**
- 현재 메인 루프에서 VAP blocking 추론 → ASR 공급 지연, TurnGPT와 직렬 실행
- ONNX Runtime은 GIL 해제하므로 별도 스레드에서 진짜 병렬 실행 가능
- 분리하면 budget 초과 문제도 구조적으로 해소

**벤치마크 결과 (Ryzen 5 5600X)**:
- VAP 10Hz: mean 7.5ms, P99 11.3ms (concurrent). Budget 100ms 대비 89% headroom.
- TurnGPT 3Hz: mean 8.5ms, P99 12.0ms (concurrent). Budget 333ms 대비 96% headroom.
- 동시 실행 간섭: P99 기준 +2-3ms (무시 가능).
- **RPi 5에서 재측정 필요** — 데스크탑 수치와 큰 차이 예상.

### Next
- **VAP 별도 스레드 분리** — orchestrator/turn_detector 구조 변경
- **`uv run ray` 실행 테스트** — 하드웨어 + 외부 서비스 연결 상태에서 전체 파이프라인 검증
- Phase 7 — Integration tests (Python ↔ C++ 실제 연결 테스트)
- RPi 5에서 벤치마크 재실행
- TurnGPT: dialog 누적 시 비동기 cache prefill (선택적 최적화)
