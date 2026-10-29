"""Ray 음성 대화 파이프라인.

처음 읽는 순서 — 뼈대:
  __main__.py         모드 루프 (SLEEP → GREETING → ACTIVE → FAREWELL), 프로세스 수명
  wiring.py           컴포넌트 조립 (프로세스 수준 / 세션 수준)
  engines/gpt_live/   ACTIVE 세션을 도는 대화 엔진 — OpenAI GPT-Live 한 모델로 듣기·말하기
      loop.py           프레임 루프 — 마이크 → 세션, 출력 → C++, 전사 저장, 툴 실행, 노래 재생 교대, 종료 시퀀스
      instructions.py   세션 지시문 텍스트 (대화 모델 · 백엔드 · 노래 재생 중 · 라디오 문구·판정기) + 시작 컨텍스트
      tools.py          백엔드 함수 툴 정의·핸들러 (기억 검색, 볼륨·밝기, 노래 재생·정지, 라디오 시작·재개, 종료)
      songs.py          노래 카탈로그 (assets/songs.json) 로드 · 파일 세트 검증 · LLM 용 목록
      radio.py          라디오 — 대본(assets/radio_scripts) · 진행 추적 · 끼어들기 판정 · 멈춤·재개 (루프가 I/O)

엔진 밖에 두는 것:
  session_context.py  장기기억에서 사용자 노트 블록·최근 대화 원문 만들기 (지시문 조립이 넣는다)
  history.py          세션 히스토리 (SQLite write-through)
  memory/             장기 기억 — 에피소드·프로필 추출/검색. 선택 가능한 서브시스템
  device_settings.py  볼륨·밝기 단계 — 상태·저장·시작 시 재적용 (바꾸는 입구는 gpt_live/tools.py)
  greeting_audio.py   인사 오디오 사전 생성
  audio_level.py      재생 파일 출력 레벨 정규화 — 차임·인사 WAV 를 GPT-Live 말소리 크기에 맞춘다
  trace.py            실행 기록 (관측용, 동작에 영향 없음)

참조용 — 선형으로 읽지 않음:
  types.py          벤더 인터페이스 (ILLM / ITTS / IEmbedder) + 그 계약 타입, 공통 별칭
  settings.py       오디오 형식, DB·자산 경로 등 공유 상수

adapters/           외부 경계 — 벤더·하드웨어·외부 모델 래퍼. 바꿀 때만 연다.
"""
