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


## 차후 고려

- **eval text 모드 유실**: `--text`(quality/memory 스위트를 음성 단계 없이 LLM 직접 평가)가 참조하던 `voice_pipeline/text_session.py`(TextSession)가 저장소에 존재하지 않아(커밋 이력에도 없음 — 미push 유실 추정) 모드 전체를 제거함. 재도입하려면 TextSession 재구현 필요.
- **음악 댄스 메인 로봇 통합**: `music_dance/`는 현재 독립 실행(시리얼 포트 단독 점유). 메인 로봇의 한 모드로 통합 시 분석 코어(analyzer/timeline) 재사용 전제. 실시간(마이크) 입력이 필요해지면 HPSS(lookahead 필요) 재설계.
- **TurnDetector 유사도 임베딩 캐싱**: `_text_similarity` 호출 패턴에서 한쪽 텍스트(`_last_prepare_text`)가 반복됨. 한쪽 임베딩을 캐싱하면 추론 비용 절반 가능.
- **임베딩 실패 복구**: 배치 임베딩 실패 시 에피소드가 `embedding=None`으로 저장되고(warning만) 재생성/백필 메커니즘 없음 — 해당 에피소드는 벡터 검색에서 영구 제외 (`load_all_embeddings`가 `embedding IS NOT NULL`만 로드).
- **프로필 서브토픽 중복**: `(topic, sub_topic)` 정확 일치 매칭뿐이라 LLM이 유사 서브토픽을 다른 이름으로 생성 가능 (movie vs movies). merge 프롬프트가 기존 슬롯 전체를 보여주지 않아 유사 슬롯이 신규 APPEND로 빠질 가능성.
- **`write_executor.shutdown(wait=True)` 블로킹**: 종료 직전 마지막 세션의 `process_session`을 submit한 뒤 wait하므로, LLM 호출이 진행 중이면 프로세스 종료가 그 완료까지 블로킹. timeout/cancel_futures 미적용.
- **`started_at` 이중 생성**: `__main__.py`와 `ConversationHistory.new_session()`이 각자 `datetime.now()`를 호출 — 대개 같은 초로 절삭되지만 초 경계에서 1초 차 가능. 단일 생성·공유로 통합 검토.
