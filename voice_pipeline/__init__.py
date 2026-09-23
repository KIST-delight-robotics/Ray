"""Ray 음성 대화 파이프라인.

처음 읽는 순서 — 뼈대:
  __main__.py         모드 루프 (SLEEP → GREETING → ACTIVE → FAREWELL), 프로세스 수명
  wiring.py           컴포넌트 조립 (프로세스 수준 / 세션 수준). settings.ENGINE 으로 엔진을 고른다
  engines/            ACTIVE 세션을 도는 대화 엔진 두 구현 — 하나만 실행된다
    gpt_live/         OpenAI GPT-Live 한 모델로 듣기·말하기 (현재 기본)
      loop.py           프레임 루프 — 마이크 → 세션, 출력 → C++, 전사 저장, 툴 실행, 노래 재생 교대, 종료 시퀀스
      instructions.py   세션 지시문 텍스트 (대화 모델 · 백엔드 · 노래 재생 중) + 시작 컨텍스트 붙이기
      tools.py          백엔드 함수 툴 정의·핸들러 (기억 검색, 볼륨·밝기, 노래 재생·정지, 종료)
      songs.py          노래 카탈로그 (assets/songs.json) 로드 · 파일 세트 검증 · LLM 용 목록
    cascade/          ASR → 턴 감지 → LLM → TTS 를 이어 붙인 엔진
      loop.py           프레임 루프 — ASR, 턴 감지, 재생, barge-in
      generator.py      응답 생성 (ContextBuilder → LLM → TTS, 백그라운드)
      context_builder.py  턴마다 LLM 입력을 블록별 토큰 예산으로 조립
      summarizer.py     히스토리 롤링 요약 (백그라운드 LLM)
      turn_detector.py  VAP + TurnGPT + VAD 결합 → turn_shift / prepare / cancel / interrupt 판정
      text_session.py   오디오 없이 텍스트로 도는 세션 (eval --text)

두 엔진이 같이 쓰는 것:
  session_context.py  장기기억에서 프로필 블록·최근 세션 블록 만들기 (엔진이 자기 입력에 넣는다)
  history.py          세션 히스토리 (SQLite write-through)
  memory/             장기 기억 — 에피소드·프로필 추출/검색. 선택 가능한 서브시스템
  device_settings.py  볼륨·밝기 단계 — 상태·저장·시작 시 재적용 (바꾸는 입구는 gpt_live/tools.py)
  greeting_audio.py   인사/작별 오디오 사전 생성
  trace.py            실행 기록 (관측용, 동작에 영향 없음)

참조용 — 선형으로 읽지 않음:
  types.py          벤더 인터페이스 (IASR / ILLM / ITTS / IEmbedder) + 그 계약 타입, 공통 별칭
  settings.py       오디오 형식, DB 경로, 토큰 예산 등 공유 상수

adapters/           외부 경계 — 벤더·하드웨어·외부 모델 래퍼. 바꿀 때만 연다.
"""
