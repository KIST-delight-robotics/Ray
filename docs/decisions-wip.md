# Decision Log — Work in Progress

진행 중인 작업의 결정 기록. 작업 완료 후 정리하여 `decisions.md`에 통합.


## [미구현] ASR 배경 잡음 SNR 스윕 (MUSAN 디지털 사전 믹싱)

> 상태: 설계 확정, 구현 착수 전. 현재 `scripts/eval/prepare_audio.py`에는 잡음 믹싱 코드가 없음
> (TTS 생성·꼬리 트림·RMS 정규화만 존재).

- **디지털 사전 믹싱 채택, 별도 스피커 음향 동시 재생 기각**: eval의 목적이 WER-vs-SNR 곡선이라 *재현 가능하고 정확한* SNR이 핵심. 두 번째 스피커로 잡음을 동시 재생하면 룸 리버브·마이크 AGC까지 반영돼 더 현실적이지만 SNR을 정밀 통제할 수 없음. 변형 WAV를 사전 생성해 파일로 검수·재사용하는 쪽이 eval 성격에 맞음.
- **디지털 SNR은 상한선 — 음향 경로가 방 잡음을 더함**: 스피커→마이크 재생이라 마이크에서의 실효 SNR = 디지털 SNR − 룸/마이크 floor. 높은 SNR(20dB)에서는 방 자체 잡음이 지배해 곡선이 평탄해질 수 있음. 조용한 방에서 돌리고, 디지털 SNR을 통제 변수로 삼아 **절대 WER이 아닌 레벨 간 상대 순위**로 해석.
- **음성 레벨 고정, 잡음 floor만 가변 (SNR이 유일 독립 변수)**: 음성을 clean과 동일한 정규화 레벨(-20 dBFS)로 두고 잡음만 올림 — 음향 리그에서 음성-대-룸noise 비율이 모든 SNR에서 동일해야 주입 잡음의 효과만 분리됨. 합성 후 클리핑은 **피크 가드(>0.95면 합성 전체 감쇠)**로 처리하고 RMS 재정규화는 하지 않음: 재정규화하면 음성 재생 레벨이 SNR마다 달라져 통제가 깨짐. 피크 가드는 음성·잡음을 같은 배율로 줄여 SNR을 보존하며, 실측상 0dB·-20dBFS에서도 피크 0.48이라 발동조차 안 함.
- **SNR 기준 RMS는 전체 클립이 아닌 유성 구간**: 선행/후행 무음을 포함하면 speech RMS가 희석돼 실효 SNR이 목표보다 높아짐. trim 패스와 동일한 peak 대비 임계로 유성 구간만 잡아 RMS 계산.
- **변형은 `wav/noise/` 하위 폴더에 — normalize 패스 재진입 방지**: prepare_audio의 정규화/트림 패스가 `output_dir.glob("*.wav")`(비재귀)로 전체를 다시 정규화함. 변형을 같은 폴더에 두면 재실행 시 SNR이 파괴되므로, glob에 안 잡히는 하위 폴더에 격리. 변형 생성은 트림·정규화가 끝난 *최종* clean WAV를 입력으로 받도록 패스 순서 뒤에 배치.
- **잡음 클립 선택은 (id+SNR) 시드로 결정론적**: 재실행 시 바이트 동일 출력 → eval 재현성. clip 선택과 crop offset을 같은 `random.Random(seed)`에서 뽑음(`Random(str)`은 sha512 기반이라 PYTHONHASHSEED 무관하게 결정론적).
- **범위 = ASR 스위트 · MUSAN `noise`(앰비언트) · 영어**: 잡음이 VAP/턴 감지에 미치는 영향은 별개 연구라 turn-taking/interruption 스위트는 무잡음 유지. ASR 언어가 영어라 babble(타화자)은 다음 단계로 미룸 — 1차는 앰비언트 단일 카테고리로 SNR을 유일 축으로 둠. SNR 레벨 `[clean, 20, 15, 10, 5, 0]`은 clean~10dB가 정상~약한 잡음, 5/0dB가 스트레스.


## 차후 고려

- **음악 댄스 메인 로봇 통합**: `music_dance/`는 현재 독립 실행(시리얼 포트 단독 점유). 메인 로봇의 한 모드로 통합 시 분석 코어(analyzer/timeline) 재사용 전제. 실시간(마이크) 입력이 필요해지면 HPSS(lookahead 필요) 재설계.
- **VAD 단일 실패가 안전장치 전체를 무력화**: `user_is_speaking=False`가 PENDING의 cancel 두 경로(VAP user-favor, ASR 비유사)를 모두 전제 단계에서 차단 — VAP p_now=0.89·ASR 대폭 변화에도 cancel 불가 사례 실측. ASR 텍스트 갱신을 발화 증거로 쓰는 보강은 별도 검토 과제.
- **TurnDetector 유사도 임베딩 캐싱**: `_text_similarity` 호출 패턴에서 한쪽 텍스트(`_last_prepare_text`)가 반복됨. 한쪽 임베딩을 캐싱하면 추론 비용 절반 가능.
- **임베딩 실패 복구**: 배치 임베딩 실패 시 에피소드가 `embedding=None`으로 저장되고(warning만) 재생성/백필 메커니즘 없음 — 해당 에피소드는 벡터 검색에서 영구 제외 (`load_all_embeddings`가 `embedding IS NOT NULL`만 로드).
- **프로필 서브토픽 중복**: `(topic, sub_topic)` 정확 일치 매칭뿐이라 LLM이 유사 서브토픽을 다른 이름으로 생성 가능 (movie vs movies). merge 프롬프트가 기존 슬롯 전체를 보여주지 않아 유사 슬롯이 신규 APPEND로 빠질 가능성.
- **`write_executor.shutdown(wait=True)` 블로킹**: 종료 직전 마지막 세션의 `process_session`을 submit한 뒤 wait하므로, LLM 호출이 진행 중이면 프로세스 종료가 그 완료까지 블로킹. timeout/cancel_futures 미적용.
- **`started_at` 이중 생성**: `__main__.py`와 `ConversationHistory.new_session()`이 각자 `datetime.now()`를 호출 — 대개 같은 초로 절삭되지만 초 경계에서 1초 차 가능. 단일 생성·공유로 통합 검토.
