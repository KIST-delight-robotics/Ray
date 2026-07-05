# Decision Log — Work in Progress

진행 중인 작업의 결정 기록. 작업 완료 후 정리하여 `decisions.md`에 통합.


## 차후 고려

- **eval text 모드 유실**: `--text`(quality/memory 스위트를 음성 단계 없이 LLM 직접 평가)가 참조하던 `voice_pipeline/text_session.py`(TextSession)가 저장소에 존재하지 않아(커밋 이력에도 없음 — 미push 유실 추정) 모드 전체를 제거함. 재도입하려면 TextSession 재구현 필요.
- **음악 댄스 메인 로봇 통합**: `music_dance/`는 현재 독립 실행(시리얼 포트 단독 점유). 메인 로봇의 한 모드로 통합 시 분석 코어(analyzer/timeline) 재사용 전제. 실시간(마이크) 입력이 필요해지면 HPSS(lookahead 필요) 재설계.
- **VAD 단일 실패가 안전장치 전체를 무력화**: `user_is_speaking=False`가 PENDING의 cancel 두 경로(VAP user-favor, ASR 비유사)를 모두 전제 단계에서 차단 — VAP p_now=0.89·ASR 대폭 변화에도 cancel 불가 사례 실측. ASR 텍스트 갱신을 발화 증거로 쓰는 보강은 별도 검토 과제.
- **TurnDetector 유사도 임베딩 캐싱**: `_text_similarity` 호출 패턴에서 한쪽 텍스트(`_last_prepare_text`)가 반복됨. 한쪽 임베딩을 캐싱하면 추론 비용 절반 가능.
- **임베딩 실패 복구**: 배치 임베딩 실패 시 에피소드가 `embedding=None`으로 저장되고(warning만) 재생성/백필 메커니즘 없음 — 해당 에피소드는 벡터 검색에서 영구 제외 (`load_all_embeddings`가 `embedding IS NOT NULL`만 로드).
- **프로필 서브토픽 중복**: `(topic, sub_topic)` 정확 일치 매칭뿐이라 LLM이 유사 서브토픽을 다른 이름으로 생성 가능 (movie vs movies). merge 프롬프트가 기존 슬롯 전체를 보여주지 않아 유사 슬롯이 신규 APPEND로 빠질 가능성.
- **`write_executor.shutdown(wait=True)` 블로킹**: 종료 직전 마지막 세션의 `process_session`을 submit한 뒤 wait하므로, LLM 호출이 진행 중이면 프로세스 종료가 그 완료까지 블로킹. timeout/cancel_futures 미적용.
- **`started_at` 이중 생성**: `__main__.py`와 `ConversationHistory.new_session()`이 각자 `datetime.now()`를 호출 — 대개 같은 초로 절삭되지만 초 경계에서 1초 차 가능. 단일 생성·공유로 통합 검토.
