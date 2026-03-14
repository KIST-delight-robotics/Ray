# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

543 tests pass, ruff 0 errors.

### 미해결 사항

**배치 내 상태 전이 누락** (실측 후 판단)
- 배치 N프레임에서 VAP 결과 1개로 타이머를 N*30ms 증분 → 배치 중간 발화 상태 변화 반영 안 됨
- VAP 추론 주기(100ms ≈ 3프레임)이고 배치가 주로 2-3프레임이므로 영향 제한적
- **실제 파이프라인 실행해서 조기 turn_shift 발생 여부 확인 필요**

**VAP/TurnGPT 예산 초과 시 처리 미구현**
- VAP budget 초과 20-25% (P95: 150ms, budget 100ms). 프레임 드롭 or 5Hz 폴백 설계 필요
- 현재는 배치 드레인만 있고 명시적 budget 초과 처리 없음

### Next
- **`uv run ray` 실행 테스트** — 하드웨어 + 외부 서비스 연결 상태에서 전체 파이프라인 검증
- Phase 7 — Integration tests (Python ↔ C++ 실제 연결 테스트)
- VAP budget 초과 20-25% 허용 설계 (프레임 드롭 or 5Hz 폴백)
- TurnGPT: dialog 누적 시 비동기 cache prefill (선택적 최적화)
