# Decision Log — Work in Progress

진행 중인 작업의 결정 기록. 작업 완료 후 정리하여 `decisions.md`에 통합.


## 턴 종료(cancel/interrupt) 재설계 — Phase 상태기계

- **cancel/interrupt 경계 = begin_streaming**: "레이 음성이 잠깐이라도 났으면 무조건 interrupt(cancel 아님)"라는 사용자 관점 기준을 만족시키려면 재생이 *가능해지는* 마지막 Python 지점을 경계로 잡아야 함. `playback_started`는 C++가 `play()` 호출 시점(가청 시작보다 SFML/ALSA 버퍼만큼 앞)에 보내고 Python은 폴링 지연 후 처리 → 가청 시작과의 선후가 수십 ms 내에서 모호. `begin_streaming`(=`send_stream_start`)은 그 이전엔 C++에 재생 명령이 없어 *물리적 무음 보장*이 되는 유일 지점. 그래서 cancel은 항상 begin_streaming 이전(브리지 무접촉)이라 `send_stop` 불필요·STOPPING 미경유, interrupt는 항상 이후 — STOPPING이 항상 interrupt 의미가 되어 reason 태그 불필요.

- **STREAMING은 종료-감지 공백 구간(단일채널 interrupt 안 함)**: `robot_audio`(VAP 참조 채널)는 `playback_started`가 클럭을 세팅해야 생김. 그 전 STREAMING에선 VAP가 interrupt vs backchannel을 구분할 robot 채널이 없음. 단일채널 fallback은 (a) backchannel 오인터럽트, (b) stop_pos≈0 빈 기록 문제가 있어 채택하지 않고 interrupt 감지를 PLAYING으로 미룸. 비용: STREAMING 내에서 시작·종료하는 짧은 barge-in은 그 턴엔 무시됨(begin_streaming에서 ASR이 reset되고 이후 누적되므로 다음 턴 입력으로 이연 — 유실은 아님). bridge_ms 실측 중앙값 ~97ms라 공백이 보통 짧고, 긴 STREAMING(느린 TTS)이 최악. 근본 대응은 C++ prebuffer(INTERVAL_MS=360ms 분량)·TTS throughput이지 단일채널 감지가 아님.

- **cancel 신호 = user_is_speaking 전제 + p_now/p_fut(즉시), grace는 similarity**: cancel은 interrupt와 동일 구조 — `user_is_speaking`을 전제로 하고(실제로 말해야 함) 그 위에서 turn-taking 확률 `p_now/p_fut`로 플로어 회수를 확인. user_is_speaking *단독*은 backchannel·노이즈에 약하고, p *단독*은 무음 중 확률 변동에 오발화하므로 둘 다 필요. VAP가 네이티브 10Hz(각 결과가 이미 ~100ms 적분)라 100ms 미만 프레임 sustain은 같은 캐시 추론 재독이라 무의미 → 즉시 발화. ASR finalization noise는 시간 grace 대신 **마지막 prepare 텍스트(=응답이 생성된 기준)** 와의 유사도로 거름 — prepare-skip 게이트(`sim≥0.8`→재생성 생략)와 **같은 비교의 양면**이라 별도 기준선(T0) 추가 없이 `_last_prepare_text` 재사용. 이 user_is_speaking 전제가 "침묵 timeout(turn_shift Path2)으로 shift했는데 p만 높은" 모순 입력에서의 turn_shift↔cancel thrash도 막음.

- **detector 상태 wipe를 turn_shift→commit으로 지연(PENDING 도입)**: turn_shift는 로봇이 실제 커밋(begin_streaming)하기 전까진 잠정적. 기존엔 turn_shift 직후 per-frame 상태를 전부 지워 "cancel=같은 턴 연속"이 불가능했음. PENDING은 상태를 보존해 cancel 시 매끄럽게 rewind하고, commit에서 비로소 wipe+dialog append. 부수 효과로 detector가 interrupt 모드(ROBOT_TURN)에 진입하는 시점이 robot_audio가 생기는 시점과 일치 → 기존의 "ROBOT_TURN인데 robot_audio 없음" 사각이 소멸.

- **stale 응답 방지 = turn_shift의 prepare 선점 (detector 내부)**: turn_shift 조건이 충족돼도, 마지막 prepare 이후 ASR이 *유효하게(비유사)* 바뀐 게 남아 있으면(=`_check_prepare`가 발화하면) turn_shift 대신 **prepare를 먼저** 내보내 새 텍스트로 재생성하고 다음 프레임에 shift. "늦은 finalization으로 준비된 응답이 stale" 케이스를 detector의 **기존 유사도 게이트**(`_last_prepare_text` 대비)로 그대로 처리 — SessionLoop에 임베더/`similarity_fn` 주입 불필요(유사도 검사 중복 회피). prepare 선점은 `_check_prepare`의 `_asr_has_changed` 게이트 덕에 *미처리 변화가 있을 때만* 일어나, 텍스트가 안정된 흔한 경우엔 turn_shift가 바로 fire(speculation 이득 유지). (SessionLoop에 별도 `similarity_fn` staleness 가드를 두는 안은 always-on detector 검사와 불일치·중복이라 기각.)
- **Python↔C++ 순수 전송은 무시 가능(~0.04ms 편도, Pi 루프백 IXWebSocket 실측)**: turn-taking 타이밍에서 전송은 0으로 취급. bridge_ms(~57-97ms)는 통신이 아니라 C++ prebuffer + Python 프레임 폴링(~30ms)이 지배.


## 차후 고려

- **SimilarityConfig/MemoryConfig 임베딩 필드 중복**: 양쪽 config에 model, use_onnx 등이 중복 존재. 공유 EmbeddingConfig 추출 여부는 실제 사용 패턴 보고 판단.
- **similarity.compare() 임베딩 캐싱**: TurnDetector 호출 패턴에서 `a`(이전 텍스트)가 반복됨. 한쪽 임베딩을 캐싱하면 추론 비용 절반 가능. 기존 코드도 동일 패턴이라 regression은 아님.
- **similarity 유닛 테스트 부재**: EmbeddingSimilarity, DiffLibSimilarity, create_similarity 팩토리에 대한 유닛 테스트가 없음. 현재는 TurnDetector 테스트에서 ISimilarity를 mock하여 간접 검증.
