"""Ray 음성 대화 파이프라인.

처음 읽는 순서 — 한 턴이 흐르는 뼈대:
  __main__.py       모드 루프 (SLEEP → GREETING → ACTIVE → FAREWELL), 프로세스 수명
  wiring.py         컴포넌트 조립 (프로세스 수준 / 세션 수준)
  session_loop.py   ACTIVE 세션의 프레임 루프 — ASR, 턴 감지, 재생, barge-in
  generator.py      응답 생성 (ContextBuilder → LLM → TTS, 백그라운드)
  prompt.py         LLM 입력 조립 + 롤링 요약

필요할 때 읽는 것:
  turn_detector.py  VAP + TurnGPT + VAD 결합 → turn_shift / prepare / cancel / interrupt 판정
  history.py        세션 히스토리 (SQLite write-through)
  memory/           장기 기억 — 에피소드·프로필 추출/검색. 선택 가능한 서브시스템
  text_session.py   오디오 없이 텍스트로 도는 세션 (eval --text)
  greeting_audio.py 인사/작별 오디오 사전 생성
  trace.py          실행 기록 (관측용, 동작에 영향 없음)

참조용 — 선형으로 읽지 않음:
  types.py          벤더 인터페이스 (IASR / ILLM / ITTS / IEmbedder) + 그 계약 타입, 공통 별칭
  settings.py       오디오 형식, DB 경로, 토큰 예산 등 공유 상수

adapters/           외부 경계 — 벤더·하드웨어·외부 모델 래퍼. 바꿀 때만 연다.
"""
