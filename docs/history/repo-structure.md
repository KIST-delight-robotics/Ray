# 저장소 구조 정리

> 2026-08 ~ 2026-09 저장소 구조 정리 작업의 기록. 완료 후 고정 — 이후 변경은 반영하지 않는다.

## 개요

저장소의 배치 기준을 바꾼 네 번의 정리를 묶었다. 동작은 바꾸지 않고, 파일이 어디에 왜 있는지가 트리에서 보이게
하는 것이 공통 목표였다.

| 날짜 | 정리 | 기준 |
|---|---|---|
| 2026-08-31 | 파이썬 구조 정리 | 패키지마다 같은 규칙 대신 경계 기준으로 |
| 2026-08-31 | 트레이싱 | 생성자 주입 대신 logging 식 모듈 API 로 |
| 2026-09-18 | 저장소 폴더 정리 | "누가 쓰는 파일인가" 로 `assets/` · `var/` · `config/` |
| 2026-09-21 | 엔진 폴더 분리 | 대화 엔진을 `engines/gpt_live/` · `engines/cascade/` 로 |

## 파이썬 구조 정리 — 규칙 기준 분리에서 경계 기준 분리로 (2026-08-31)

- **문제 진단**: 15개 패키지(13.5k줄)에 모듈마다 `__init__` + `exceptions.py` + README + 인터페이스 +
  `tests/<module>/` 를 똑같이 붙이는 규칙이 있었다.
  - 그래서 200줄짜리와 2,000줄짜리가 같은 무게로 보였다.
  - 18개 인터페이스 중 12개가 구현체 1개였다.
  - 예외 18개 중 7개는 raise 조차 안 됐고, `except` 로 잡는 곳은 프로덕션 전체에 1곳뿐이었다(`except Exception` 이
    폴백 정책).
  - 반복된 정리 시도가 효과가 없던 이유는 같은 규칙 아래서 재배치만 했기 때문이다. 규칙 자체를 바꿔야 했다.
- **분리 기준은 "언제 건드리나"**: import 그래프가 이미 진짜 경계를 보여줬다.
  - 외부 것을 감싼 모듈(ASR/TTS/LLM/VAP/TurnGPT/bridge/audio/led)은 `core` 만 의존했다. 반면 내부 로직
    (context/generation/history/memory)은 규칙을 어기며 서로 import 하고 있었다.
  - 그래서 외부 래퍼는 `adapters/` 에 파일 하나씩 둔다. 역할이 아니라 결합 방식으로 묶었다 — "벤더·하드웨어를
    바꿀 때만 연다".
  - 내부 로직은 top-level 파일로 뭉쳤다(`generator`, `prompt`, `history`).
    (→ 이후 엔진 폴더 분리에서 `generator` · `prompt` 는 `engines/` 아래로 — 아래 절)
  - 13k줄에 15개 패키지 트리는 10만 줄짜리 구조다.
- **인터페이스는 벤더 교체 대상만 (IASR/ILLM/ITTS/IEmbedder)**: 나머지는 구체 타입을 주입한다.
  - Python 은 mock 에 ABC 가 필요 없다(`Mock(spec=Class)`).
  - 인터페이스 hop 하나가 처음 읽는 사람에게 파일 하나를 더 열게 한다.
- **커스텀 예외 전부 삭제 → `RuntimeError(원인 메시지) from exc`**:
  - 호출자가 종류별로 처리하지 않고(`except Exception`), 예외에 실린 데이터도 없고, 벤더 예외 은닉은 RuntimeError
    로도 된다.
  - 종류 구분이 실제로 필요해지면 그때 클래스 하나를 추가한다.
- **테스트 더블은 테스트가 소유한다**:
  - 인메모리 저장소 더블은 SQLite `:memory:` 로 대체했다. 두 백엔드 모두 단일 커넥션이라 동작한다.
  - 검증용으로 리스트에 쌓아야 하는 call/trace 스토어는 `tests/fakes.py` 로 옮겼다.
- **공유 상수는 `settings.py`**:
  - 토큰 예산(`_MAX_HISTORY_TOKENS`, `_HARD_CAP_TOKENS`)을 wiring 이 다른 모듈의 private 클래스 변수로 꺼내
    쓰고 있었다.
  - "중앙 설정 없음" 규칙은 벤더 파라미터엔 맞지만, 여러 모듈이 공유하는 값엔 안 맞는다.
  - 클래스 변수는 테스트 monkeypatch 를 위해 남기고, 초기값만 settings 에서 가져온다.
- **타입은 만들어 내는 모듈이 소유**:
  - `core/types.py` 27개 중 17개가 파일 한두 곳에서만 쓰였다(`Phase` 는 session_loop 만, `SystemMode` 는
    __main__ 만). "경계를 넘는 타입은 core 에" 규칙이 소유자가 명확한 타입까지 끌어모은 것이다.
  - `types.py` 에는 세 가지만 남겼다: 벤더 인터페이스, 그 시그니처가 참조하는 계약 타입(스트림·결과·LLMMetrics),
    공통 별칭.
  - 나머지는 생산자로 보냈다(`TurnDecision → turn_detector`, `ResponseData → generator`, `VAPResult → adapters/vap`,
    …). 방향이 전부 소비자 → 생산자라 순환이 없다.
  - 남긴 기준은 "소유자에 두면 어색한 의존이 생기는 것"이다. 예를 들어 `LLMMetrics` 를 `llm_openai` 에 두면,
    history 가 OpenAI 어댑터를 import 하게 된다.
- **트레이싱 주입 정리는 분리**: 생성자마다 들어가는 `session_id` / `call_store` 정리는 동작 리팩터라, 구조
  이동(동작 변경 0)과 섞지 않고 다음 작업(아래 트레이싱 절)으로 넘겼다.
- **함정 — 병합 도구가 코드를 잃는다**: ast 기반으로 파일을 합칠 때 두 가지가 유실됐다.
  - 첫 body 문장의 데코레이터(`@pytest.fixture`)
  - 삭제 범위 사이에 끼어 있던 모듈 상수(`_CALL_COLUMNS`)

  대규모 이동 후에는 반드시 전체 테스트 수를 전후 비교할 것.

## 트레이싱 — 주입에서 logging 식 모듈 API 로 (2026-08-31)

- **문제**: CallRecord 하나를 채우려면 emitter 마다 세 가지가 필요했다.
  - 싱크: `call_store` 생성자 주입 6곳
  - `session_id`: 생성자 3곳 + wiring 이 세션마다 속성 4개에 대입
  - `turn_index`: 스토어를 우체통으로 써서 `set_turn_index` / `current_turn_index` 로 되읽기

  같은 레코드의 두 컨텍스트 필드가 다른 경로로 전파됐고, `if self._call_store is not None` 가드가 15곳이었다.
- **관찰**: 트레이싱은 logging 과 같은 모양이다 — fire-and-forget 이고, 컨텍스트는 "지금 진행 중인 것"이며,
  싱크는 프로세스에 하나다. 아무도 logger 를 주입하지 않는데 트레이싱에만 DI 규칙을 적용해서, 9개 객체에
  `session_id` 를 밀어 넣게 됐다. 명시적 주입이 원인이었다.
- **채택 — 모듈 전역 API**: `trace.py` 가 logging 처럼 프로세스 전역 싱크와 현재 세션·턴 컨텍스트를 가진다.
  싱크가 없으면 no-op 이라 가드가 사라진다. 저장 형식(두 테이블)은 그대로고 기록 경로만 바뀌었다.
- **핫패스 버퍼링은 유지**:
  - VAP(10 Hz)·TurnGPT 추론 스레드는 SQLite 쓰기를 피해야 해서, 기록을 모아 `reset()/stop()` 에서 쓴다.
  - 이때 `turn_index` 는 **캡처 시점** 값이어야 한다. flush 시점 값을 쓰면 세션 전체가 마지막 턴으로 찍힌다.
    그래서 스탬프만 찍는 캡처와 나중에 쓰는 쓰기를 분리했다.
  - 비동기 writer 스레드 대안은 기각했다. 테스트 단언이 경합하게 되고, 얻는 건 균일성뿐이다.
- **`pipeline_traces.turn_index` 추가**: 두 테이블의 조인 키가 `session_id` 뿐이었다. `save_turn` 이 저장 시점의
  턴을 찍는다. `_save_trace` 는 completed/truncated/cancelled 모두 `turn_detector.reset()` 이전(해당 exchange
  안)에서 호출되므로, 그 턴의 call_records 와 같은 값이 된다.
- **전역 상태 트레이드오프**:
  - 프로세스당 세션 하나가 전제다(현 아키텍처·eval 모두 순차).
  - 동시 세션이 필요해지면 내부를 `contextvars` 로 바꾸되 API 는 바꾸지 않는다.
  - 테스트는 autouse `trace.reset()` 으로 격리하고 `call_log` / `turn_log` 픽스처로 검증한다 — logging 의 `caplog`
    과 같은 패턴이다. 결과적으로 이전(생성자에 fake 주입)보다 테스트 코드가 짧아졌다.
- **`HistorySummarizer.session_id` 는 남김**: 트레이싱이 아니라 롤링 요약 영속화(다음 세션 이월)의 키라서 정당한
  인자다. "session_id 가 보이면 전부 트레이싱 잔재"가 아니다.

## 저장소 폴더 정리 — "누가 쓰는 파일인가" 한 축으로 (2026-09-18)

- **문제**:
  - `data/` 에 배포용 세그먼트 npy, 평가 질문, 런타임 DB 가 섞여, 폴더 전체를 gitignore 하고 예외를 파는 구조가 됐다.
  - `assets/` 에도 배포 파일과 생성물(인사 WAV 캐시, 모션 CSV)이 섞여 있었다.
  - "data" 라는 이름이 데이터셋인지 런타임 상태인지 아무것도 말해 주지 않는 것이 원인이었다.
- **기준 — 파일을 누가 쓰느냐 하나로 나눈다**:

  | 누가 쓰나 | 위치 | git |
  |---|---|---|
  | 개발자가 만들어 저장소와 함께 배포 | `assets/` | 전부 추적 |
  | 프로그램이 실행 중에 씀 (유닉스 /var 의 "변하는 상태") | `var/` | 전부 무시 |
  | 운영자가 기기마다 손으로 씀 | `config/ray.env` | 무시, `*.example` 만 추적 |

  - 평가 질문 세트는 코드와 같이 리뷰되므로 `evaluation/` 안에 둔다. 평가 산출물(wav·노이즈 베드·결과)은 프로그램
    출력이라 `var/eval/` 이다.
  - 예외 규칙은 대용량 음원(`assets/audio/music|vocal`) 하나만 남긴다. 이건 크기 때문이지 성격 때문이 아니다.
- **`ray.env` 를 `~/.config/ray/` 에서 저장소 `config/` 로**: 레포 하나만 열면 애셋·상태·설정이 다 보이게 하려는 것이다.
  - tiktoken 캐시도 `var/cache/` 로 옮겨 홈 디렉터리 의존이 없어졌다([부팅 자동실행](boot-autostart.md)에서 홈으로 옮겼던 것).
  - 기기별 설정 파일(`device_settings.json`)은 프로그램이 쓰므로 `config/` 에 합치지 않고 `var/` 에 둔다.
- **모션 CSV 63개(12 MB)를 추적 대상으로**: 무시 규칙이 있는데 일부는 이미 추적돼 있던 모순 상태였다. C++ 가 읽는
  입력이라 assets 규칙대로 전부 추적한다. 크기가 부담되면 음원처럼 별도 배포로 빼는 결정을 따로 한다.
- **다음 단계로 남긴 것**: 기기별 모터 홈 값(`[robot.unitN]` + `RAY_UNIT`)도 같은 기준으로 로컬 `robot.toml` 로 분리.
  (→ 이후 [모터 기록](motor.md)의 robot.toml 분리로 진행)

## 대화 엔진을 `engines/gpt_live/` · `engines/cascade/` 로 분리 (2026-09-21)

목표는 트리에서 "어느 엔진 것인지" 보이게 하는 것이었다.

- **문제**:
  - `voice_pipeline/` 상위에 cascade 전용(session_loop·generator·prompt·turn_detector·text_session), live
    전용(live_session), 공통(history·trace…)이 나란히 있었다. 그래서 파일 이름만으로는 무엇이 실행되고 무엇이
    놀고 있는지 안 보였다. `__init__.py` 의 읽는 순서 독스트링이 그 부재를 메우고 있었다.
  - `live_session.py` 는 루프·지시문·툴 정의·기억 핸들러 700줄이 한 파일이라, 툴을 하나 더 넣을 때 "어디
    넣나"가 고민이 됐다.
- **기준**:
  - `engines/` 아래 두 폴더는 같은 자리의 두 구현이다. 한 엔진만 쓰는 건 그 폴더에, 둘이 같이 쓰는 건 상위에 둔다.
  - 폴더 이름: cascade 는 업계 용어 그대로, gpt_live 는 제품명 그대로(어댑터·문서와 대조하기 쉬움).
  - 두 루프 파일은 같은 역할이라 같은 이름 `loop.py` 다. 클래스 이름(`SessionLoop` / `LiveSessionLoop`)이 이미 다르다.
- **`prompt.py` 는 셋으로**:
  - 양 엔진이 실제로 공유하는 건 프로필 블록과 최근 세션 블록을 만드는 120줄뿐이라, `session_context.py`(상위)로 뺐다.
    `memory/` 에 두지 않은 이유는, 그쪽은 저장·검색이고 이건 LLM 에 보여 줄 문장 만들기라서다.
  - 턴마다 예산 안에 쌓는 `ContextBuilder` 는 cascade 만의 일이라 `context_builder.py` 로 갔다.
  - 롤링 요약은 `summarizer.py` 로 갔다.
- **`live_session.py` 는 셋으로** — 바뀌는 빈도로 나눴다:
  - `loop.py`: 500줄, 잘 안 바뀜.
  - `instructions.py`: 지시문 텍스트 50줄. 위임률이 문구 형식에 좌우돼 가장 자주 만지는 부분이다. GPT-Live API
    필드명 그대로 이름 붙였다.
  - `tools.py`: 툴 정의 + 핸들러. 앞으로 늘어난다.
  - 볼륨·밝기 툴도 `tools.py` 로 합쳐, `device_settings.py` 에는 상태·저장만 남겼다. `search_memory` 와
    `MemoryRetriever` 의 관계와 같아진다. 다른 입구(버튼·앱)가 생겨도 상태 모듈은 그대로다.
- **로거 이름은 유지**: `voice_pipeline.session_loop`·`voice_pipeline.live_session`·`voice_pipeline.prompt` 는
  파일명이 아니라 역할 이름으로 보고 그대로 둔다(`voice_pipeline.bridge`, `.tts` 처럼). 콘솔 서사 필터와
  로그 분석 스크립트가 이 이름을 본다.
- **범위**: 순수 이동과 파일 쪼개기만 했다. 로직·함수 이름 변경도, cascade 삭제도 하지 않았다.
