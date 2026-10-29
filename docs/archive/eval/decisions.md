# 평가 시스템

> 사용하지 않는 평가 시스템(`evaluation/`)의 기록. 고정 — 이후 변경은 반영하지 않는다. 코드는 태그 `archive/cascade` 에 있다.

## 개요

cascade 엔진을 E2E 로 재던 평가 파이프라인. 질문 WAV 를 외부 스피커로 틀어 실제 마이크·ASR·턴 감지·응답 생성·TTS 를
거치게 하고, 결과를 자동 채점해 대시보드로 냈다. 2026-06 ~ 07 에 배선 공유, 음향 노이즈 베드, 실행 순서 정리를 했고,
장기기억만 따로 재는 벤치(LoCoMo, LongMemEval)를 붙였다.

- 시스템 설계·사용법 원문: [eval-system.md](eval-system.md)
- cascade 엔진 자체의 결정: [archive/cascade/decisions.md](../cascade/decisions.md)


## 배선 공유 — evaluation 패키지 승격 + 배선 팩토리 추출 (2026-07)

- **배선 3중 사본 → `wiring.py` 단일화**: `__main__.py`와 eval `run.py`가 프로세스·세션 조립 ~150줄을 verbatim 복제하고
  있었음 (vad_fn 클로저는 주석까지 동일, LLM 설정 상수도 두 벌). 프로덕션에 컴포넌트가 추가될 때 eval이 조용히 어긋나는
  드리프트가 구조화된 상태.
  - `build_components(db_path, led_enabled)` + `ProcessComponents.create_session(**session_loop_kwargs)`로 추출.
  - eval 델타를 전수 조사한 결과 전부 **이미 존재하던 중립 파라미터**(스토어 DB 경로, LED off, SessionLoop
    콜백/`skip_generation`/`record_path`)여서 프로덕션에 eval 전용 기능이 새로 들어갈 필요가 없었음 — "프로덕션엔
    주입점만, eval 동작 금지" 규율을 CLAUDE.md에 명문화.
- **eval 전용 기본값은 eval 쪽 `partial`로**: `disable_exit_keywords=True`는 eval 정책이므로 wiring 기본값이 아니라
  run.py에서 `partial(components.create_session, disable_exit_keywords=True)`로 고정 — 주입점(파라미터)은 프로덕션에,
  정책(값)은 소비자에.
- **sandbox.py는 팩토리 미적용**: 의도적으로 컴포넌트를 대체(Observable/Stub/NoOp)하는 재현 하네스라, 팩토리에 태우면
  컴포넌트마다 오버라이드 파라미터가 필요해져 규율이 무너짐. 실 그래프 재현이 목적이 아니므로 자체 조립 유지.
- **`create_session`의 세션 인자는 `**kwargs` 통과**: SessionLoop 시그니처를 wiring에 복제하면 두 곳이 어긋날 수
  있어(이번 작업의 원인 그 자체), 타입 명시 대신 통과를 선택 — 검증은 SessionLoop 생성자가 담당.


## 턴 단위 계측과 대시보드

- **Prepare 유사도 게이트 평가 — judge 판정은 이진(meaning_changed / response_appropriate)**: 목적이 "threshold가
  적절한가"라는 예/아니오 질문이라 5점 척도보다 harmful skip *비율*이 바로 해석됨. 응답까지 judge에 보여주는 이유는
  의미가 달라져도 응답이 우연히 여전히 적절한 경우와 실제 피해(harmful)를 구분하기 위함 — threshold 문제의 심각도를
  과대평가하지 않게 함.
- **similarity 값을 CallRecord로 기록**: judge 판정만으로는 "현 threshold에서 bad skip N건"까지만 알고 *얼마로 올려야
  하는지*를 모름. 게이트의 4종 결정(skip/keep/regenerate/cancel)마다 similarity를 기록하면 harmful skip의 유사도 구간이
  보여 조정 폭이 정량화되고, regenerate 기록으로 "threshold를 낮추면 skip으로 바뀌었을 건수"도 역산 가능. 단,
  counterfactual 시뮬레이션이 아니라 현재 threshold에서 실제 발생한 결정만 판정.
- **턴 매칭 = `call_records.turn_index` 전용 컬럼** (초기엔 metadata JSON): 질문 단위 e2e 드릴다운이
  vap/turngpt/tts 등 *모든* 모듈 호출을 턴별로 귀속해야 해서 게이트에만 있는 metadata로는 부족.
  - `ICallStore`에 공유 턴 카운터(`set_turn_index`/`current_turn_index`)를 두고 TurnDetector(턴 권위)가 전환마다 갱신 —
    백그라운드 스레드의 vap/turngpt까지 같은 인덱스를 스탬프. (→ 이후 트레이싱 모듈 API의 `trace.set_turn()`으로 대체 —
    [history/repo-structure.md](../../history/repo-structure.md))
  - 생성(LLM/TTS)은 `commit()`으로 끝난 *직전* 교환에 귀속, 카운터는 `commit()`에서만 증가 — cancel rewind는 같은 턴의
    연속이라 reset/cancel에서 올리면 안 됨.
  - 타임스탬프는 µs 정밀도(`utc_now_str`)로 올려 cross-stage 기록 정렬 가능. report.py는 구 DB의 컬럼 부재를 감지해
    metadata 폴백.
- **asr_text ≠ system_text가 곧 게이트 skip의 흔적**: `system_text`(messages의 user 메시지)는 `_begin_streaming`이 기록한
  generator의 prepare 입력이고, `asr_text`는 turn shift 시점의 최종 ASR 텍스트. 두 값이 (정규화 후) 다른 경로는 유사도
  게이트의 skip/keep뿐이라, 별도 계측 없이 이 차이만으로 judge 후보를 선별 가능.
- **질문 단위 e2e 드릴다운 = 기존 대시보드 인라인 펼침**: 카테고리 통계만으론 한 질문의 전 단계(ASR→턴감지→게이트→
  생성→TTS)가 어떻게 맞물렸는지 볼 수 없음. 질문은 이미 각 탭에 나열되므로 그 자리에서 펼치는 게 탭 이동 0회로 맥락
  유지에 최적 — 전용 탭(질문 목록 중복)·CLI 승격(비시각·단건)은 기각.
- **드릴다운 데이터 경로 = report.py 사전 집계, dashboard는 scored.json만**: dashboard는 `f(scored.json)` 순수 함수
  유지(자족적·공유 용이). report.py가 call_records를 turn_index로 묶어 턴별 `stage_calls`를 첨부, eval.db 직접 조회는
  계약을 깨고 두 소스를 요구해 기각. vap 원시 프레임은 요약만(count/avg/max) — scored.json 폭증 방지. `call_issues`는
  turn_index 도입으로 세션→턴 단위로 정확화.


## 배경 잡음 — 음향 노이즈 베드 (2026-06)

- **디지털 사전 믹싱 → 음향 베드로 선회**: 처음엔 깨끗한 질문 WAV에 MUSAN noise를 목표 SNR로 미리 섞어 굽는 디지털
  방식을 구현까지 했다.
  - 그러나 디지털 믹스는 *ASR만* 정확한 SNR로 스트레스할 뿐 — 질문이 끝나는 순간 잡음도 끊겨, 마이크·VAD·침묵 기반 턴
    판정 등 나머지 파이프라인은 무잡음 입력을 봐서 e2e 잡음 강건성을 못 본다.
  - 둘째 스피커로 잡음을 연속 재생하는 음향 베드는 룸/마이크 경로를 그대로 통과해 전 단계가 현실 잡음 조건을 겪는다.
    그래서 디지털 경로를 전부 제거하고 베드만 남김.
  - SNR 정밀 통제를 포기하는 대신, 마이크 실측 캘리브로 *실효* SNR을 잡는다(아래).
- **단일 마스터 + 조건별 gain 기록 (조건별 WAV 안 굽기)**: 베드 레벨은 `bed_master.wav` 하나에 곱하는 gain으로 표현하고,
  gain은 `calibration.json`에 조건별로 *기록*만 한다(NoiseBed가 재생 시점에 `master × gain`). 캘리브마다
  `bed_<cond>.wav`를 새로 굽고 덮어쓰는 것보다 — 마스터 불변 + gain 한 숫자 기록이 재현성도 낫고(아티팩트가 안 흔들림)
  레벨 조절도 한 줄 편집으로 끝난다. 런타임 스케일은 캐시 마스터에 1회 곱하는 비용이라 무시 가능.
- **dmix 필수 — plughw/hw 독점 장치 금지**: 베드와 질문이 *한 카드에서 섞여야* 하므로 dmix PCM이 필수.
  `plughw:`/`hw:`는 한 번에 한 프로세스만 열 수 있어, 베드가 켜진 medium/loud 블록에서 질문 재생 aplay가 device busy로
  줄줄이 실패한다(베드의 stderr는 DEVNULL이라 조용히 죽기도). 그래서 질문 재생 기본 장치를 dmix로, noise-bed 모드에서
  비-dmix 장치면 시작 시 차단.
- **SNR 재설정은 마이크 재측정 없이 가능**: 마이크 실측 상수(speech_rms S, bed_n_ref_rms N_ref, room_floor_rms)를
  `calibration.json`에 저장하므로, SNR 목표만 바꾸면 `solve_gain`으로 gain을 오프라인 재계산할 수 있다(리그가 안 바뀌는
  한 스피커·마이크 불필요). gain 1.0 = 마스터 원본(= 마이크에서 N_ref) 기준.
- **베드 마스터는 오버레이로 정상(stationary)**: 여러 MUSAN noise를 끝이어붙이지(concat) 않고 겹쳐(overlay) 합산 —
  세션이 루프 어디에 겹치든 실효 SNR이 목표 근처로 유지된다. 단일 음원 순차 재생은 순간 조용/트랜지언트 구간이 세션별
  SNR을 흔든다.
- **범위 = ASR 스위트 · MUSAN `noise`(앰비언트) · 영어**: 잡음이 VAP/턴 감지에 미치는 영향은 별개 연구라
  turn-taking/interruption 스위트는 무잡음 유지. ASR 언어가 영어라 babble(타화자)은 다음 단계로 미룸 — 1차는 앰비언트
  단일 카테고리.


## 실행 순서 — questions.json 단일 기준 (2026-07)

- **대시보드 표시 순서 = questions.json**: 탭마다 제각각이던 정렬(suite_name 알파벳순, 카테고리 인덱스, scenario_id 등)을
  questions.json 순회 순서에서 유도한 rank로 통일. `turns` 배열(재생 순서)을 기준으로 삼지 않은 이유: noise-bed
  스케줄러가 재생 순서를 조건 블록으로 재배치하기 때문.
- **노이즈 스케줄 = 전역 rotating block → category별 조건 블록**: 기존 전역 rotating block(8세션 단위)은 재생이 suite
  경계를 넘나들어 사람이 실행을 지켜볼 때 경과 파악이 어려웠다.
  - 검토해 보니 전역 방식의 실익은 전환 횟수 최소화 정도뿐 — 조건 풀이 원래 나열 순서를 보존해 거시 흐름은 어차피
    questions.json 순서라 suite-시간 상관도 끊어주지 않았다.
  - category별 블록은 블록이 커서(3~21세션) 전환이 오히려 더 적고(15회), 실행이 "asr: quiet→medium→loud → turn_taking:
    …"으로 읽힌다.
  - suite별 블록 안은 3문항짜리 lq/mem suite가 1문항 블록으로 무너지고 quick 모드(스위트당 1문항)에서 성립 불가라 기각.
- **배정은 불변 (전역 연속 round-robin)**: 조건 배정은 기존과 동일한 flat index `i % N` 유지 — suite×조건 셀 구성이 기존
  스케줄과 완전히 같음을 시뮬레이션으로 확인(나머지 분배도 suite마다 다른 조건으로 자연 분산). 바뀐 것은 재생 묶음뿐이라
  과거 실행과 통계적으로 비교 가능.
- **시작 조건은 category마다 회전**: 항상 quiet부터 돌면 "quiet은 늘 category 초반"이라는 조건-시간 상관이 생겨, category
  인덱스만큼 오프셋을 밀어 상쇄(asr: q→m→l, tt: m→l→q, …).
- **set_level 후 settle 1초**: 레벨 전환은 aplay 프로세스 재시작이라 스피커 출력이 안정될 때까지 짧은 창이 있고, 블록 첫
  세션이 라벨과 다른 실효 SNR에서 시작할 수 있다. 전환 후 1초 대기 → 마이크 큐 drain 순서로 방지.
- **기존 특성 (이 변경과 무관)**: interruption suite는 유닛 나열이 delay→audio→question 순이라 질문↔조건이 1:1로
  고정된다(int_q_001은 항상 quiet 등). 배정 로직이 동일하므로 이전 스케줄에서도 같았음 — 문제 시 질문 수나 나열 순서
  조정으로 해소.


## `--text` 모드 복원 — TextSession은 유실이 아니었음 (2026-07)

- **"유실" 판정 정정**: `--text` 모드 제거(ea3d831)의 근거였던 "text_session.py 저장소 부재"는 오판 — 파일이 로봇 작업
  기기의 워킹 트리에 untracked로 존재했다(285줄 완본). 현재 코드베이스 API와 시그니처 대조 + fake LLM 스모크로 **무수정
  동작**을 확인하고 그대로 커밋. "재구현 필요" 아님.
- **조립은 wiring으로**: 구 run.py의 `create_text_session` 팩토리(컴포넌트 수동 조립)를 되살리는 대신
  `ProcessComponents.create_text_session()`으로 wiring에 배치 — `create_session()`과 대칭이고 "조립은 wiring 한 곳" 규율
  유지. TextSession도 voice_pipeline 소속이라 의존 방향 문제없음.
- **text-only 실행도 전체 그래프를 빌드**: 구 코드는 `needs_audio` 분기로 텍스트 전용 실행 시 오디오
  컴포넌트(VAP·TTS·AudioInput 등) 초기화를 건너뛰었지만, 복원판은 `build_components()`를 그대로 사용 — 시작이 ~10초
  무거워지는 대신 wiring에 조건부 조립 분기를 넣지 않는다(중립 주입점 규율).


## 장기기억 벤치 — LoCoMo (2026-07)

프로덕션 메모리 파이프라인(추출·저장·검색)을 고치지 않고 그대로 재는 하네스.

- **듀얼 인제스트 (대화당 관점별 DB 2개)**: LoCoMo는 user–user 대화인데 에피소드/프로필 추출 프롬프트는 user
  중심("Extract from USER utterances only")이라, 한 화자만 user로 매핑하면 상대 화자 기억이 통째로 빠져 점수가 검색
  품질이 아닌 매핑 손실을 측정하게 된다. 벤치용 2인 추출 프롬프트를 주입하는 대안은 "배포된 시스템의 점수"라는 주장을
  약화시켜 기각 — 같은 대화를 A/B 각각 user 관점으로 두 번 인제스트하고 QA 때 양쪽 DB 검색을 병합. 인제스트 비용 2배는
  gpt-4o-mini 기준 무시 가능.
- **`now_fn` 주입 (프로덕션 유일 변경)**: retriever의 recency decay(반감기 30일)가 `datetime.now()` 기준이라 과거(2023년)
  대화의 에피소드는 salience가 사실상 0이 된다. 날짜를 현재로 평행이동하는 대안은 temporal 카테고리 정답이 절대
  날짜("8 May 2023")를 참조해서 불가 → "현재"를 마지막 세션 다음 날로 고정하는 clock 주입점을 추가 (wiring의 중립 주입점
  규율에 부합).
- **에피소드 날짜는 QA 컨텍스트에서 렌더링**: 추출 프롬프트가 에피소드 본문에 날짜 포함을 금지하므로 temporal 질문의
  유일한 단서는 `Episode.timestamp` — 벤치 QA 컨텍스트에서 `[YYYY-MM-DD]` prefix로 렌더링하고, 순서도 검색 랭킹 대신
  시간순으로 배치 (temporal 추론 보조).
- **질문당 새 MemoryRetriever**: retained buffer가 세션(대화) 스코프 상태라 재사용하면 직전 질문의 검색 결과가 다음 질문
  컨텍스트를 오염시킨다. `update_citations()` 미호출로 DB는 읽기 전용 유지.
- **3단계 분리 (ingest/answer/score)**: 비용이 인제스트(LLM 추출)에 집중되므로 DB 스냅샷·answers.jsonl·judgements.jsonl을
  남겨 — retriever 상수 실험은 answer부터, judge/집계 변경은 score만 재실행. 각 단계는 처리 완료 키를 확인해 자연 resume.
- **헤드라인 = adversarial 제외 judge 정확도**: Mem0 등 기존 발표 수치와 비교 가능하도록 관례를 따름. adversarial은
  abstention 지시("Not mentioned in the conversation")를 넣고 별도 리포트. 채점의 실질 가치는 점수보다 오답의 실패
  귀속(추출/검색/생성) — LoCoMo evidence 주석으로 세션 단위 recall을 계산해 튜닝 근거를 뽑는 것.

### 실험에서 드러난 병목

- **파일럿(conv-26) 결과: recency decay가 검색 병목**: 질문 근거는 19개 세션에 고르게 분포하는데 반감기 30일 감쇠로
  5개월 전 에피소드의 salience가 최근 대비 ~2%가 되어 검색이 후반 세션에 쏠렸다(초반 6세션 검색 비율 1.4%, 헤드라인
  22.4%). 반감기를 사실상 무감쇠로 오버라이드하고 answer만 재실행하자 검색 귀속 실패 52→11건, single-hop +10pp, 초반 세션
  검색 28.4%. 상수 실험이 반복될 것이라 `answer --half-life-days` 옵션으로 상설화 (인제스트 재사용, 재실험 ~2분).
- **다음 병목은 추출 디테일 손실**: decay를 꺼도 헤드라인이 26.3%에 그친 이유 — evidence 세션의 에피소드가 검색까지 돼도
  본문에 질문의 구체 사실이 없는 경우가 지배적 (에피소드 "self-care가 중요함을 깨달음" vs 정답 "me-time에 달리기·독서·
  바이올린").
  - 추출 프롬프트의 "Be selective" + 1–3문장 추상 내러티브 방침의 트레이드오프로, 친구 컨셉의 의도일 수 있으나 구체
    회상도 친구다움의 일부라는 반론 여지.
  - 귀속 로직의 generation 버킷은 세션 단위 프록시라 이 손실과 순수 생성 실패를 구분하지 못한다.
  - temporal(8.1%)은 무감쇠에서도 불변 — decay가 아니라 날짜 추론 자체가 병목.
- **추출 프롬프트 v2 실험: 손실의 주범은 "Be selective"와 자족성 부재**: 벤치용 변형(선택성 → 화제 전수 커버리지, 구체
  명사·원문 어휘 보존, 대명사 해소 자족성 — 감정 맥락·전언 규칙은 유지)으로 conv-26을 재인제스트.
  - no-decay 기준 헤드라인 26.3→37.5%, multi-hop 2배(18.8→37.5%), 검색 귀속 실패 9건, 에피소드 수 +48%(166→246).
  - 특히 자족성 규칙이 "their assistant" 같은 지시어를 이름("Caroline")으로 해소해 상대 화자 관련 질문의 검색을 살렸다.
  - 주입은 `MemoryWriter(episode_system_prompt=...)` 중립 주입점으로.
  - 남은 병목은 temporal 날짜 추론(13.5%)과 답변 생성 단계. open-domain 불변(15.4%)은 벤치 QA 프롬프트의 외부 지식 금지
    규칙과 카테고리 성격의 충돌로 추정 — 기억 시스템 문제 아님.
  - v2의 프로덕션 채택은 별도 결정 사항 (선택적 기억 컨셉과의 긴장, 에피소드 수 증가 트레이드오프).
- **답변 프롬프트의 외부 지식 금지 → "개인 사실 한정"으로 축소**: 전면 금지는 open-domain만 죽이는 게 아니라 전
  카테고리에서 과잉 abstention을 유발하고 있었다 (완화 후 single-hop +8.6pp, 헤드라인 37.5→42.8%). 플립 분석으로 오염
  신호 없음을 확인 — 새로 맞힌 답의 근거 토큰이 검색된 에피소드 안에 존재. 축소된 규칙을 유지하는 근거: ① 작화
  방지(그럴듯한 개인 사실 추측이 점수를 오염), ② adversarial abstention의 앵커, ③ 벤치 공개(2024) 이후 컷오프
  모델(gpt-5.4-mini 등)로 바꿀 때의 암기 방어.
- **Mem0 하네스 검토 → 이식 기각, 상대비교 원칙 확정**: Mem0 평가 repo의 judge(목록 중 1개만 맞아도 CORRECT, 날짜
  ±14일·기간 ±50% 허용, gold 항목 0개일 때만 WRONG)와 답변 프롬프트(7단계 CoT, 메모리 200개 주입, abstention 금지)는
  LoCoMo 오답 패턴에 정밀 대응한 벤치 전용 설계라 이식하면 그 오버피팅을 수입하게 된다. 따라서 발표 수치(Mem0 66.9% 등)와
  우리 수치의 절대비교는 성립하지 않음.
  - 원칙: 벤더 관행의 통이식은 지양하되 하네스 개선 자체는 열어둔다(단 변경 시 기존 런 재채점으로 비교 정합성 유지 —
    judgements 캐시 덕에 저렴).
  - 점수는 상대비교 전용으로 쓰고, 기준선(full-context 등)이 필요하면 우리 하네스로 직접 측정한다.
  - 벤치 점수는 진단 증거일 뿐 — 프로덕션 반영은 변경 단위로 쪼개 레이 기준으로 별도 논증 (v2의 자족성 규칙은 무충돌,
    커버리지 강제·무감쇠는 "선택적 친구 기억" 컨셉과 긴장).


## 장기기억 벤치 — LongMemEval (2026-07)

- **도입 이유와 LoCoMo와의 역할 분담**: LongMemEval은 user–assistant 형식이라 레이 사용 방식과 일치한다 — 듀얼 인제스트
  불필요, 1인칭 질문("How many appointments did I...")이 곧 제품 시나리오. 지식 갱신·abstention이 1급 측정 축이고, 공식
  평가 스크립트·judge가 고정돼 있어(학계 저자 소유) 벤더 오버피팅 우려가 낮아 공식 judge를 원문 이식해 기본 채점으로
  쓴다(gpt-4o, 유형별 yes/no). LoCoMo는 장기 회상 축의 싼 A/B 도구로 유지.
- **인제스트 비용 구조가 정반대**: LoCoMo는 대화 하나에 질문 ~200개가 붙어 비용이 상각되지만(질문당 ~0.8콜), LME는
  문항마다 전용 히스토리(S: ~40세션)가 붙어 비용이 문항 수에 비례한다(질문당 ~120콜). 대응: `--sample-per-type`
  결정론적 층화 샘플 + oracle(evidence-only) 변형으로 배관·추출 축을 거의 공짜로 검증.
- **구조는 "파이프라인 공유 + 데이터셋 소유 분리"**: 로더·답변 프롬프트·전용 judge는 데이터셋 모듈이 소유한다. 답변
  프롬프트를 공용 템플릿로 묶는 안은 기각 — 화자 구조가 달라(2인 대화 vs 단일 사용자) 일반화 문구("그룹 헤더의 인물")가
  양쪽 다 나쁘게 만든다. 파이프라인 역학(인제스트·검색·기록·resume·실패 귀속)만 공유.
- **답변 프롬프트 슬림화 (규칙 6→3)**: 벤치 규칙 더미가 실패를 만들고 있었다 — preference 유형(기억 기반 추천 생성)이
  "없으면 정확히 abstain"+"최단 답변" 규칙에 막혀 0점. oracle 12문항 4조건 비교(모델·judge 고정)에서 엄격 33%→슬림 42%,
  preference 0→2/2. 단 삭제한 "나중 노트가 최신 상태" 규칙이 knowledge-update에서 일했을 가능성(1.0→0.5, n=2) — 전체
  런에서 재판단 (→ 아래 oracle 전체 500문항에서 노이즈로 판단).
- **제품축 답변 스타일 추가**: `--answer-style production` — 실제 레이 프롬프트(DEFAULT_SYSTEM_PROMPT)와 프로덕션
  포매터(Block 1/2/4 조립, 인용 태그 파싱)를 그대로 재사용해 "같은 기억으로 실제 레이라면 몇 점"을 잰다. 규칙 0개인데 슬림
  벤치 프롬프트와 대등(42%) — 진단축(슬림)/제품축(production) 병행의 근거. LoCoMo는 레이가 대화 참여자가 아니라 비적용.
- **no-memory ablation = 오염 검사**: 검색·프로필 없이 답하게 하면 0/12, CORRECT 0건 — 점수 전부가 기억에서 나옴을 확인.
  벤치 공개 이후 컷오프 모델(gpt-5.4-mini 등)로 답변자를 바꿀 때마다 이 ablation을 다시 볼 것.
- **oracle 파일럿의 추출 축 발견**: temporal(두 사건 간 날짜 계산)·multi-session(횟수·금액 집계)·single-session-assistant는
  어떤 프롬프트로도 0점. oracle은 검색 난이도가 0이므로 추출 단계 문제로 확정 — 구체 수치 손실(LoCoMo와 동일 병목) +
  assistant 발화 미추출. 후자는 제품 관점의 발견: "레이가 말해준 내용"을 레이가 기억하지 못한다.
- **oracle 전체 500문항 (수정 전 베이스라인)**: 헤드라인 46.8%(n=470) — 검색 무결 조건이므로 "추출+답변 축의 상한".
  - 유형별: single-session-user 76.6 / knowledge-update 69.4 / temporal 44.1 / multi-session 38.8 / preference 30.0 /
    assistant 16.1, abstention 27/30(90%).
  - 파일럿(n=2)의 이상치들이 정리됨 — knowledge-update의 "최신값 규칙" 삭제 영향은 노이즈였던 것으로 판단, assistant가
    정확히 0이 아닌 건 user 발화에 섞여 들어간 정보 덕.
  - 실패 귀속은 generation 버킷이 지배(244/250)인데 oracle 특성상 실체는 추출 디테일 손실 + 답변 단계의 집계·날짜 추론
    한계 — 세션 단위 프록시로는 이 둘이 안 갈라진다.
- **벤치 사용 원칙 — 티어링 (진단 임무 종료)**: decay·추출 디테일·assistant 미추출·temporal 추론·abstention 건전성까지
  기억 시스템 수정에 필요한 진단 목록은 확보됐다.
  - 이후 변경마다 전체 런을 돌리지 않는다 — 이터레이션에는 스모크(oracle 유형별 5문항 또는 LoCoMo conv-26, ~$0.2/~5분;
    +10pp급 효과는 이 규모로 판별됨), 전체 런은 프로덕션 반영 결정 시점의 확정 수치용만.
  - 수정 전 베이스라인 스냅샷: LoCoMo 42.8%(conv-26, v2+no-decay+knowledge 규칙), LME oracle 46.8%/abstention
    90%(프로덕션 추출 프롬프트·슬림 답변).
  - 실행 비용 감각: 전체 런 ~6M토큰/~$1.5, 스모크 ~0.2M토큰.
