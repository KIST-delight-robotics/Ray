# Decision Log — Work in Progress

진행 중인 작업의 결정 기록. 작업 완료 후 정리하여 `decisions.md`에 통합.


## eval 개편 — evaluation 패키지 승격 + 배선 팩토리 추출

- **배선 3중 사본 → `wiring.py` 단일화**: `__main__.py`와 eval `run.py`가 프로세스·세션 조립 ~150줄을 verbatim 복제하고 있었음 (vad_fn 클로저는 주석까지 동일, LLM 설정 상수도 두 벌). 프로덕션에 컴포넌트가 추가될 때 eval이 조용히 어긋나는 드리프트가 구조화된 상태. `build_components(db_path, led_enabled)` + `ProcessComponents.create_session(**session_loop_kwargs)`로 추출. eval 델타를 전수 조사한 결과 전부 **이미 존재하던 중립 파라미터**(스토어 DB 경로, LED off, SessionLoop 콜백/`skip_generation`/`record_path`)여서 프로덕션에 eval 전용 기능이 새로 들어갈 필요가 없었음 — "프로덕션엔 주입점만, eval 동작 금지" 규율을 CLAUDE.md에 명문화.
- **eval 전용 기본값은 eval 쪽 `partial`로**: `disable_exit_keywords=True`는 eval 정책이므로 wiring 기본값이 아니라 run.py에서 `partial(components.create_session, disable_exit_keywords=True)`로 고정 — 주입점(파라미터)은 프로덕션에, 정책(값)은 소비자에.
- **sandbox.py는 팩토리 미적용**: 의도적으로 컴포넌트를 대체(Observable/Stub/NoOp)하는 재현 하네스라, 팩토리에 태우면 컴포넌트마다 오버라이드 파라미터가 필요해져 규율이 무너짐. 실 그래프 재현이 목적이 아니므로 자체 조립 유지.
- **`create_session`의 세션 인자는 `**kwargs` 통과**: SessionLoop 시그니처를 wiring에 복제하면 두 곳이 어긋날 수 있어(이번 작업의 원인 그 자체), 타입 명시 대신 통과를 선택 — 검증은 SessionLoop 생성자가 담당.


## ASR 배경 잡음 — 음향 노이즈 베드 (스피커 동시 재생)

- **디지털 사전 믹싱 → 음향 베드로 선회 (디지털 경로 제거)**: 처음엔 깨끗한 질문 WAV에 MUSAN noise를 목표 SNR로 미리 섞어 굽는 디지털 방식을 썼다. 그러나 디지털 믹스는 *ASR만* 정확한 SNR로 스트레스할 뿐, 마이크·VAD·턴 감지 등 나머지 파이프라인은 무잡음 입력을 봐서 e2e 잡음 강건성을 못 본다. 둘째 스피커로 잡음을 연속 재생하는 음향 베드는 룸/마이크 경로를 그대로 통과해 전 단계가 현실 잡음 조건을 겪는다. 그래서 디지털 경로(`noise_mixer`, `prepare_audio --musan-dir`, run의 `snr_levels` 스윕, score `by_snr`)를 전부 제거하고 베드만 남김. SNR 정밀 통제를 포기하는 대신, 마이크 실측 캘리브로 *실효* SNR을 잡는다(아래).
- **단일 마스터 + 조건별 gain 기록 (조건별 WAV 안 굽기)**: 베드 레벨은 `bed_master.wav` 하나에 곱하는 gain으로 표현하고, gain은 `calibration.json`에 조건별로 *기록*만 한다(NoiseBed가 재생 시점에 `master × gain`). 캘리브마다 `bed_<cond>.wav`를 새로 굽고 덮어쓰는 것보다 — 마스터 불변 + gain 한 숫자 기록이 재현성도 낫고(아티팩트가 안 흔들림) 레벨 조절도 한 줄 편집으로 끝난다. 런타임 스케일은 캐시 마스터에 1회 곱하는 비용이라 무시 가능.
- **dmix 필수 — plughw/hw 독점 장치 금지**: 베드와 질문이 *한 카드에서 섞여야* 하므로 dmix PCM이 필수. `plughw:`/`hw:`는 한 번에 한 프로세스만 열 수 있어, 베드가 켜진 medium/loud 블록에서 질문 재생 aplay가 device busy로 줄줄이 실패한다(베드의 stderr는 DEVNULL이라 조용히 죽기도). 그래서 질문 재생 기본 장치를 dmix로, noise-bed 모드에서 비-dmix 장치면 시작 시 차단. `--device` 미지정 시 noise-bed는 calibration의 측정 장치를 자동 채택.
- **SNR 재설정은 마이크 재측정 없이 가능**: 마이크 실측 상수(speech_rms S, bed_n_ref_rms N_ref, room_floor_rms)를 `calibration.json`에 저장하므로, SNR 목표만 바꾸면 `solve_gain`으로 gain을 오프라인 재계산할 수 있다(리그가 안 바뀌는 한 스피커·마이크 불필요). gain 1.0 = 마스터 원본(= 마이크에서 N_ref) 기준.
- **베드 마스터는 오버레이로 정상(stationary)**: prepare_noise_bed가 여러 MUSAN noise를 끝이어붙이지(concat) 않고 겹쳐(overlay) 합산 — 세션이 루프 어디에 겹치든 실효 SNR이 목표 근처로 유지된다. 단일 음원 순차 재생은 순간 조용/트랜지언트 구간이 세션별 SNR을 흔든다.
- **범위 = ASR 스위트 · MUSAN `noise`(앰비언트) · 영어**: 잡음이 VAP/턴 감지에 미치는 영향은 별개 연구라 turn-taking/interruption 스위트는 무잡음 유지. ASR 언어가 영어라 babble(타화자)은 다음 단계로 미룸 — 1차는 앰비언트 단일 카테고리.


## Eval 순서 정합 — questions.json 단일 기준

- **대시보드 표시 순서 = questions.json**: 탭마다 제각각이던 정렬(suite_name 알파벳순, 카테고리 인덱스, scenario_id 등)을 config의 `suite_descriptions`/`question_texts` 키 순서(= questions.json 순회 순서)에서 유도한 rank로 통일. `turns` 배열(재생 순서)을 기준으로 삼지 않은 이유: noise-bed 스케줄러가 재생 순서를 조건 블록으로 재배치하기 때문.
- **노이즈 스케줄 = 전역 rotating block → category별 조건 블록**: 기존 전역 rotating block(8세션 단위)은 재생이 suite 경계를 넘나들어 사람이 실행을 지켜볼 때 경과 파악이 어려웠다. 검토해 보니 전역 방식의 실익은 전환 횟수 최소화 정도뿐 — 조건 풀이 원래 나열 순서를 보존해 거시 흐름은 어차피 questions.json 순서라 suite-시간 상관도 끊어주지 않았다. category별 블록은 블록이 커서(3~21세션) 전환이 오히려 더 적고(15회), 실행이 "asr: quiet→medium→loud → turn_taking: …"으로 읽힌다. suite별 블록 안은 3문항짜리 lq/mem suite가 1문항 블록으로 무너지고 quick 모드(스위트당 1문항)에서 성립 불가라 기각.
- **배정은 불변 (전역 연속 round-robin)**: 조건 배정은 기존과 동일한 flat index `i % N` 유지 — suite×조건 셀 구성이 기존 스케줄과 완전히 같음을 시뮬레이션으로 확인(나머지 분배도 suite마다 다른 조건으로 자연 분산). 바뀐 것은 재생 묶음뿐이라 과거 실행과 통계적으로 비교 가능.
- **시작 조건은 category마다 회전**: 항상 quiet부터 돌면 "quiet은 늘 category 초반"이라는 조건-시간 상관이 생겨, category 인덱스만큼 오프셋을 밀어 상쇄(asr: q→m→l, tt: m→l→q, …).
- **set_level 후 settle 1초**: 레벨 전환은 aplay 프로세스 재시작이라 스피커 출력이 안정될 때까지 짧은 창이 있고, 블록 첫 세션이 라벨과 다른 실효 SNR에서 시작할 수 있다. 전환 후 1초 대기 → 마이크 큐 drain 순서로 방지.
- **기존 특성 (이번 변경과 무관)**: interruption suite는 유닛 나열이 delay→audio→question 순이라 질문↔조건이 1:1로 고정된다(int_q_001은 항상 quiet 등). 배정 로직이 동일하므로 이전 스케줄에서도 같았음 — 문제 시 질문 수나 나열 순서 조정으로 해소.


## PENDING cancel — ASR 유사도 경로의 user_is_speaking 전제 제거

- **실측 계기**: "goodbye" 발화를 ASR이 'you'로 중간 인식 → TurnGPT가 그 시점에 turn_shift → 326ms 뒤 최종 텍스트 'goodbye' 도착. 커밋까지 1.6초 이상 남아 취소 창이 열려 있었지만, `_process_pending` 진입부의 `user_is_speaking=False` 조기 반환이 유사도 검사 실행 자체를 차단 → 'you'에 대한 응답 재생 + exit 키워드('goodbye') 미감지. (기존 차후 고려 "VAD 단일 실패가 안전장치 전체를 무력화" 항목의 해소.)
- **두 cancel 경로는 신호의 시제가 다름 — 가드를 경로별로 분리**: VAP user-favor는 "지금 발화권을 잡고 있다"는 현재형 신호라 `user_is_speaking` 전제가 정합적 (VAD 침묵 + VAP 예측만으로 취소는 과격) → 가드 유지. ASR 텍스트 변경은 이미 일어난 발화가 늦게 처리된 과거형 증거 → 전제 제거. turn_shift 자체가 VAD 침묵을 조건으로 발화되므로, 수백 ms 늦는 ASR final은 이 게이트에 구조적으로 항상 걸렸다 (증거 도착 시각과 사건 시각의 혼동).
- **유예시간 없이 PENDING 전체에서 평가**: 자연 경계는 commit(begin_streaming). VAD 침묵 중 ASR 텍스트가 새로 생기는 경우는 사실상 늦은 final뿐이고, 주변 발화로 인한 오취소는 기존에도 VAD-speaking 경로로 가능했던 리스크라 새로 생기는 게 아님. 오취소 비용도 되감기 후 turn_shift 재발화(~1초 지연)로 낮음. 소음 환경에서 오취소가 관찰되면 그때 유예창 도입 검토.


## eval --text 모드 복원 — TextSession은 유실이 아니었음

- **"유실" 판정 정정**: `--text` 모드 제거(ea3d831)의 근거였던 "text_session.py 저장소 부재"는 오판 — 파일이 로봇 작업 기기의 워킹 트리에 untracked로 존재했다(285줄 완본). 현재 코드베이스 API와 시그니처 대조 + fake LLM 스모크로 **무수정 동작**을 확인하고 그대로 커밋. "재구현 필요" 아님.
- **조립은 wiring으로**: 구 run.py의 `create_text_session` 팩토리(컴포넌트 수동 조립)를 되살리는 대신 `ProcessComponents.create_text_session()`으로 wiring에 배치 — `create_session()`과 대칭이고 "조립은 wiring 한 곳" 규율 유지. TextSession도 voice_pipeline 소속이라 의존 방향 문제없음.
- **text-only 실행도 전체 그래프를 빌드**: 구 코드는 `needs_audio` 분기로 텍스트 전용 실행 시 오디오 컴포넌트(VAP·TTS·AudioInput 등) 초기화를 건너뛰었지만, 복원판은 `build_components()`를 그대로 사용 — 시작이 ~10초 무거워지는 대신 wiring에 조건부 조립 분기를 넣지 않는다(중립 주입점 규율). 컴포넌트 정리(stop_threaded/vap.stop/executor/asr/led)는 오디오 블록 finally에서 공용 정리로 이동해 text-only 경로에서도 실행.
- **대시보드 (Text) 마커 조건 단순화**: 구 코드의 `... if text_mode else s_name.startswith("mem_")`는 --text 아닐 때도 mem을 (Text)로 표기하는 낡은 분기(memory가 텍스트 전용이던 시절 잔재)라, `--text && (lq_|mem_)`일 때만 표기하도록 정리.


## 세션 내 히스토리 롤링 요약 + 고정 블록 예산

- **전역 4096 예산 폐기 근거는 실측**: ray.db 55세션 실측 결과 대화 자체는 극소(교환당 평균 ~21토큰, 역대 최장 세션 전체가 507토큰)인 반면, 실제 input_tokens(평균 4.4k)의 지배 요인은 web_search 호스티드 툴이 서버측에서 주입하는 히든 프롬프트(~4.2k, 예산으로 제어 불가·대부분 캐시됨)였다. 전역 예산은 통제할 수 없는 것을 통제하는 척했고, remainder 방식은 고정 블록이 커지면 히스토리가 조용히 전멸하는 구조라 블록별 독립 예산으로 전환. `tools_token_cost` 회계도 같은 이유로 제거 (`get_tools_token_cost`는 현재 미사용으로 잔존).
- **히스토리 예산 8192는 "요약이 거의 안 발동"하도록 넉넉하게**: 테스트 대화가 짧아 실측치는 하한 — 발화가 실측의 5배여도 트리거(0.75)까지 ~40교환. 요약은 정상 경로가 아니라 아주 긴 세션의 안전장치. 히든 프롬프트가 이미 4.2k인 환경이라 캐시되는 히스토리를 키우는 비용은 미미.
- **요약은 컨텍스트 뷰 레벨 대체 — storage 원본 불변**: 세션 종료 후 MemoryWriter가 storage에서 전체 트랜스크립트를 읽어 에피소드를 추출하므로, 요약이 원본 메시지를 대체하면 메모리 추출 입력이 손실된다. 요약 상태는 summarizer 메모리에만 존재(세션 스코프, 파생 데이터라 영속화 불필요). 메모리 검색 쿼리도 history를 직접 읽어 무영향.
- **트리거는 ContextBuilder.build() 내부**: 토큰 회계가 이미 있는 유일한 곳이라 SessionLoop/인터페이스 변경 없이 후킹. 백그라운드 스왑이라 이번 호출엔 미반영, 다음 build부터 반영 — 트리거 비율 0.75가 in-flight 구간의 여유. 스왑 전 초과분은 기존 drop-oldest가 hard cap으로 잔존(요약 실패 시 fallback 겸용).
- **하드캡 도달 요약은 폐기**: 요약 LLM `max_tokens=512`(오작동 가드)에 걸린 결과는 문장 중간 절단이라, 손상된 요약으로 교체하는 것보다 이전 상태 유지 + 다음 트리거 재시도가 낫다. 판정은 `output_tokens >= _HARD_CAP_TOKENS` 휴리스틱(LLMResult가 finish_reason을 노출하지 않아). 평소 크기는 프롬프트 soft limit(250 words)으로 제어하고, 회계는 목표치가 아닌 생성물 실측 토큰으로 한다.
- **커트라인은 교환 경계 정렬**: ConversationHistory가 user/assistant 메시지에 각각 turn_id를 부여하므로(1턴 = 1메시지) 최근 n턴 유지가 교환 중간을 자를 수 있다 — 질문만 요약에 들어가고 답변이 밖에 남는 것을 막기 위해 첫 유지 턴이 user로 시작할 때까지 커트라인을 뒤로 민다. barge-in의 사후 `update_message`는 항상 최신 턴 대상이라 n(20턴=10교환) 유지가 요약된 턴의 사후 수정도 방어.
- **요약 스레드는 daemon 단발 Thread**: single-flight(동시 1개)라 executor가 불필요하고, non-daemon executor 스레드는 프로세스 종료를 in-flight LLM 호출(최대 30초)만큼 블로킹한다. 세션 정리는 ThreadedTurnGPT와 같은 패턴(wiring `stop_threaded`에서 다음 세션 생성 시 close).


## LoCoMo 벤치 하네스 — 프로덕션 메모리 파이프라인 무수정 측정

- **듀얼 인제스트 (대화당 관점별 DB 2개)**: LoCoMo는 user–user 대화인데 에피소드/프로필 추출 프롬프트는 user 중심("Extract from USER utterances only")이라, 한 화자만 user로 매핑하면 상대 화자 기억이 통째로 빠져 점수가 검색 품질이 아닌 매핑 손실을 측정하게 된다. 벤치용 2인 추출 프롬프트를 주입하는 대안은 "배포된 시스템의 점수"라는 주장을 약화시켜 기각 — 같은 대화를 A/B 각각 user 관점으로 두 번 인제스트하고 QA 때 양쪽 DB 검색을 병합. 인제스트 비용 2배는 gpt-4o-mini 기준 무시 가능.
- **`now_fn` 주입 (프로덕션 유일 변경)**: retriever의 recency decay(반감기 30일)가 `datetime.now()` 기준이라 과거(2023년) 대화의 에피소드는 salience가 사실상 0이 된다. 날짜를 현재로 평행이동하는 대안은 temporal 카테고리 정답이 절대 날짜("8 May 2023")를 참조해서 불가 → "현재"를 마지막 세션 다음 날로 고정하는 clock 주입점을 추가 (wiring의 중립 주입점 규율에 부합).
- **에피소드 날짜는 QA 컨텍스트에서 렌더링**: 추출 프롬프트가 에피소드 본문에 날짜 포함을 금지하므로 temporal 질문의 유일한 단서는 `Episode.timestamp` — 벤치 QA 컨텍스트에서 `[YYYY-MM-DD]` prefix로 렌더링하고, 순서도 검색 랭킹 대신 시간순으로 배치 (temporal 추론 보조).
- **질문당 새 MemoryRetriever**: retained buffer가 세션(대화) 스코프 상태라 재사용하면 직전 질문의 검색 결과가 다음 질문 컨텍스트를 오염시킨다. `update_citations()` 미호출로 DB는 읽기 전용 유지.
- **3단계 분리 (ingest/answer/score)**: 비용이 인제스트(LLM 추출)에 집중되므로 DB 스냅샷·answers.jsonl·judgements.jsonl을 남겨 — retriever 상수 실험은 answer부터, judge/집계 변경은 score만 재실행. 각 단계는 처리 완료 키를 확인해 자연 resume.
- **헤드라인 = adversarial 제외 judge 정확도**: Mem0 등 기존 발표 수치와 비교 가능하도록 관례를 따름. adversarial은 abstention 지시("Not mentioned in the conversation")를 넣고 별도 리포트. 채점의 실질 가치는 점수보다 오답의 실패 귀속(추출/검색/생성) — LoCoMo evidence 주석으로 세션 단위 recall을 계산해 튜닝 근거를 뽑는 것.
- **파일럿(conv-26) 결과: recency decay가 검색 병목**: 질문 근거는 19개 세션에 고르게 분포하는데 반감기 30일 감쇠로 5개월 전 에피소드의 salience가 최근 대비 ~2%가 되어 검색이 후반 세션에 쏠렸다(초반 6세션 검색 비율 1.4%, 헤드라인 22.4%). 반감기를 사실상 무감쇠로 오버라이드하고 answer만 재실행하자 검색 귀속 실패 52→11건, single-hop +10pp, 초반 세션 검색 28.4%. 상수 실험이 반복될 것이라 `answer --half-life-days` 옵션으로 상설화 (인제스트 재사용, 재실험 ~2분).
- **다음 병목은 추출 디테일 손실**: decay를 꺼도 헤드라인이 26.3%에 그친 이유 — evidence 세션의 에피소드가 검색까지 돼도 본문에 질문의 구체 사실이 없는 경우가 지배적 (에피소드 "self-care가 중요함을 깨달음" vs 정답 "me-time에 달리기·독서·바이올린"). 추출 프롬프트의 "Be selective" + 1–3문장 추상 내러티브 방침의 트레이드오프로, 친구 컨셉의 의도일 수 있으나 구체 회상도 친구다움의 일부라는 반론 여지. 현재 귀속 로직의 generation 버킷은 세션 단위 프록시라 이 손실과 순수 생성 실패를 구분하지 못한다. temporal(8.1%)은 무감쇠에서도 불변 — decay가 아니라 날짜 추론 자체가 병목.
- **추출 프롬프트 v2 실험: 손실의 주범은 "Be selective"와 자족성 부재**: 벤치용 변형(선택성 → 화제 전수 커버리지, 구체 명사·원문 어휘 보존, 대명사 해소 자족성 — 감정 맥락·전언 규칙은 유지)으로 conv-26을 재인제스트하자 no-decay 기준 헤드라인 26.3→37.5%, multi-hop 2배(18.8→37.5%), 검색 귀속 실패 9건, 에피소드 수 +48%(166→246). 특히 자족성 규칙이 "their assistant" 같은 지시어를 이름("Caroline")으로 해소해 상대 화자 관련 질문의 검색을 살렸다. 주입 경로는 `MemoryWriter(episode_system_prompt=...)` 중립 주입점 + `ingest --episode-prompt-file` (적용 전문은 run config에 기록, 변형 파일은 `evaluation/memory_bench/variants/`). 남은 병목은 temporal 날짜 추론(13.5%)과 답변 생성 단계. open-domain 불변(15.4%)은 벤치 QA 프롬프트의 외부 지식 금지 규칙과 카테고리 성격의 충돌로 추정 — 기억 시스템 문제 아님. v2의 프로덕션 채택은 별도 결정 사항 (선택적 기억 컨셉과의 긴장, 에피소드 수 증가 트레이드오프).
- **답변 프롬프트의 외부 지식 금지 → "개인 사실 한정"으로 축소**: 전면 금지는 open-domain만 죽이는 게 아니라 전 카테고리에서 과잉 abstention을 유발하고 있었다 (완화 후 single-hop +8.6pp, 헤드라인 37.5→42.8%). 플립 분석으로 오염 신호 없음을 확인 — 새로 맞힌 답의 근거 토큰이 검색된 에피소드 안에 존재. 축소된 규칙을 유지하는 근거: ① 작화 방지(그럴듯한 개인 사실 추측이 점수를 오염), ② adversarial abstention의 앵커, ③ 벤치 공개(2024) 이후 컷오프 모델(gpt-5.4-mini 등)로 바꿀 때의 암기 방어.
- **Mem0 하네스 검토 → 이식 기각, 상대비교 원칙 확정**: Mem0 평가 repo의 judge(목록 중 1개만 맞아도 CORRECT, 날짜 ±14일·기간 ±50% 허용, gold 항목 0개일 때만 WRONG)와 답변 프롬프트(7단계 CoT, 메모리 200개 주입, abstention 금지)는 LoCoMo 오답 패턴에 정밀 대응한 벤치 전용 설계라 이식하면 그 오버피팅을 수입하게 된다. 따라서 발표 수치(Mem0 66.9% 등)와 우리 수치의 절대비교는 성립하지 않음. 원칙: 하네스는 동결하고 상대비교 전용으로 쓰며, 기준선(full-context 등)이 필요하면 우리 하네스로 직접 측정한다. 벤치 점수는 진단 증거일 뿐 — 프로덕션 반영은 변경 단위로 쪼개 레이 기준으로 별도 논증 (v2의 자족성 규칙은 무충돌, 커버리지 강제·무감쇠는 "선택적 친구 기억" 컨셉과 긴장).

## 차후 고려

- **음악 댄스 메인 로봇 통합**: `music_dance/`는 현재 독립 실행(시리얼 포트 단독 점유). 메인 로봇의 한 모드로 통합 시 분석 코어(analyzer/timeline) 재사용 전제. 실시간(마이크) 입력이 필요해지면 HPSS(lookahead 필요) 재설계.
- **TurnDetector 유사도 임베딩 캐싱**: `_text_similarity` 호출 패턴에서 한쪽 텍스트(`_last_prepare_text`)가 반복됨. 한쪽 임베딩을 캐싱하면 추론 비용 절반 가능.
- **임베딩 실패 복구**: 배치 임베딩 실패 시 에피소드가 `embedding=None`으로 저장되고(warning만) 재생성/백필 메커니즘 없음 — 해당 에피소드는 벡터 검색에서 영구 제외 (`load_all_embeddings`가 `embedding IS NOT NULL`만 로드).
- **프로필 서브토픽 중복**: `(topic, sub_topic)` 정확 일치 매칭뿐이라 LLM이 유사 서브토픽을 다른 이름으로 생성 가능 (movie vs movies). merge 프롬프트가 기존 슬롯 전체를 보여주지 않아 유사 슬롯이 신규 APPEND로 빠질 가능성.
- **`write_executor.shutdown(wait=True)` 블로킹**: 종료 직전 마지막 세션의 `process_session`을 submit한 뒤 wait하므로, LLM 호출이 진행 중이면 프로세스 종료가 그 완료까지 블로킹. timeout/cancel_futures 미적용.
- **`started_at` 이중 생성**: `__main__.py`와 `ConversationHistory.new_session()`이 각자 `datetime.now()`를 호출 — 대개 같은 초로 절삭되지만 초 경계에서 1초 차 가능. 단일 생성·공유로 통합 검토.
