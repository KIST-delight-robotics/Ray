# Ray 음성 파이프라인 평가 시스템

## 1. 평가 시스템 개요

Ray 음성 파이프라인의 전 과정(음성 인식 → 턴 감지 → 응답 생성 → 음성 합성 → 재생)을 자동으로 평가하는 시스템이다. 미리 준비한 159개의 질문을 파이프라인에 입력하고, 반응을 기록하여 자동 채점한다.

### 작동 방식

평가 시스템은 실제 Ray 파이프라인의 코드를 그대로 사용한다. ASR, TurnDetector, SpeechGenerator, CppBridge, 메모리 시스템 등 모든 모듈을 프로덕션과 동일하게 구성한다. 단, 웨이크워드 감지와 인사/작별 단계는 건너뛰고, **ACTIVE 모드(SessionLoop)만 질문 하나당 한 번씩 반복 실행**한다.

사람 대신 질문을 넣어주기 위해 별도의 외부 스피커를 사용한다. 질문 텍스트를 미리 WAV 파일로 변환해두고, 평가 시 이 WAV를 외부 스피커로 재생하면 Ray의 마이크가 이를 실제 음성 입력으로 받아 처리한다. 질문마다 SessionLoop을 새로 시작하고, WAV 재생 → 턴 감지 완료 → SessionLoop 종료 순서로 진행된다. WAV 재생 종료 시각과 턴 감지 시각을 기록하여 레이턴시를 측정한다.

LLM 응답이나 장기기억만 테스트할 때에는 Text 모드를 사용할 수 있다. 음성·턴테이킹 단계를 건너뛰고 텍스트를 직접 LLM에 전달하여, 응답 내용만 평가한다.

매 평가 실행마다 별도의 SQLite DB를 새로 생성하여 프로덕션 데이터와 분리한다. 장기기억 평가에 필요한 과거 대화 이력은 시드 데이터(seeds.json)를 주입하여 구성한다.

모든 질문의 실행이 끝나면 자동으로 채점이 이어진다. 실행 기록과 DB를 결합하여 통합 리포트를 만들고, WER 계산·LLM Judge 호출 등 자동 채점을 수행한 뒤, 결과를 HTML 대시보드로 출력한다.


## 2. 평가 영역

5개 영역으로 나뉘며, 각 영역은 음성 대화의 서로 다른 단계를 검증한다.

### 2-1. 음성 인식 (ASR) — 6개 스위트, 50개 질문

사용자의 말을 얼마나 정확하게 텍스트로 변환하는지 측정한다. WAV 재생 → 마이크 입력 → 음성 인식까지만 실행하고, 응답 생성 없이 종료한다.

| 스위트 | 설명 | 질문 예시 | 설계 의도 |
|--------|------|----------|----------|
| asr_short | 짧은 명령문 (8개) | "Turn off the lights." | 짧은 발화의 기본 인식 정확도 |
| asr_conversational | 일상 대화 (10개) | "Can you recommend a good book to read?" | 자연스러운 대화체 인식 |
| asr_long | 긴 문장 (8개) | "Can you explain how photosynthesis works in simple terms that a child could understand?" | 긴 발화에서의 인식 유지 |
| asr_numbers | 숫자 표현 (8개) | "Set a timer for three minutes and forty-five seconds." | 숫자·시간·날짜 표현 인식 |
| asr_proper_nouns | 고유명사 (8개) | "Have you heard of the Sagrada Familia in Barcelona?" | 인명·지명 등 고유명사 인식 |
| asr_confusable | 혼동 가능 표현 (8개) | "The effect of the new policy will affect everyone here." | 동음이의어·유사 발음 구별 |

### 2-2. 턴테이킹 (Turn-taking) — 8개 스위트, 63개 질문

사용자의 발화가 끝난 시점을 얼마나 정확하고 빠르게 감지하는지 측정한다. WAV 재생 → 마이크 입력 → 턴 감지 → 응답 재생까지 전체 파이프라인을 실행한다. 발화 종료 시점과 턴 감지 시점의 차이로 레이턴시를 측정한다.

| 스위트 | 설명 | 질문 예시 | 설계 의도 |
|--------|------|----------|----------|
| tt_ultra_short | 극짧은 응답 (8개) | "Yes." / "Okay." | 한두 단어 발화 종료 감지 |
| tt_short_statements | 짧은 서술문 (8개) | "I like chocolate." | 짧은 완결 문장 감지 |
| tt_long_statements | 긴 서술문 (8개) | "I've been thinking about learning to play the piano..." | 긴 발화 후 종료 감지 |
| tt_questions | 질문형 발화 (8개) | "What do you think about that?" | 질문형 억양의 종료 감지 |
| tt_lists | 나열형 발화 (8개) | "I need eggs, milk, butter, and flour." | 나열 중간에 끼어들지 않고 끝까지 대기 |
| tt_weak_endings | 약한 종결 (10개) | "I guess we could do that maybe." | "...인 것 같아" 같은 모호한 종결 감지 |
| tt_incomplete | 미완성 발화 (10개) | "I was going to say" / "Hmm let me think" | 말이 끝나지 않았음을 인식하고 대기 |
| tt_multi_turn | 다중 턴 대화 (1시나리오, 3턴) | 연속 3개 질문 | 연속 대화에서의 턴 감지 일관성 |

### 2-3. 인터럽션 (Interruption) — 1개 스위트

로봇이 대답하는 도중 사용자가 끼어들었을 때, 이를 감지하고 말을 멈추는지 측정한다. 전체 파이프라인을 실행하되, 응답이 재생되기 시작하면 끼어들기 WAV를 추가로 재생한다. 응답이 중단되었는지 여부를 기록한다.

- **질문 3개**: 긴 답변을 유도하는 질문 (예: "Tell me about the history of computers.")
- **끼어들기 음성 3종**: 짧은 것부터 긴 것까지
  - "Stop."
  - "Hey, wait a second."
  - "Wait, I have a question about something else actually."
- **끼어들기 시점 4가지**: 응답 재생 시작 후 0초, 1초, 2초, 3초
- 질문 3 × 끼어들기 3 × 지연 4 = 최대 36회 테스트

### 2-4. 응답 품질 (Quality) — 8개 스위트, 24개 질문

로봇의 응답 내용이 적절하고 자연스러운지를 LLM Judge(GPT-5.5)가 자동 채점한다. 전체 파이프라인 또는 Text 모드로 실행하여 응답 텍스트를 수집하고, LLM Judge가 채점한다.

| 스위트 | 설명 | 질문 예시 | 고유 평가 기준 |
|--------|------|----------|--------------|
| lq_factual | 사실 정확성 (3개) | "What is the speed of light?" | 정확성 (correctness) |
| lq_advice | 조언/설명 (3개) | "How can I get better at public speaking?" | 유용성 (helpfulness) |
| lq_casual | 일상 대화 (3개) | "I just got back from a really nice hike." | 대화 참여 (engagement) |
| lq_empathy | 감정 대응 (3개) | "I'm really nervous about my job interview tomorrow." | 공감 (empathy) |
| lq_voice_adaptation | 음성 적합성 (3개) | "Can you list the planets in our solar system in order?" | 포맷 적응 (format_adaptation) |
| lq_multi_turn | 맥락 유지 (1시나리오, 3턴) | "I'm thinking about adopting a cat." → 후속 질문 | 맥락 유지 (context_coherence) |
| lq_wrong_premise | 잘못된 전제 (3개) | "Why is the Great Wall visible from space?" | 교정 품질 (correction_quality) |
| lq_impossible | 불가능한 요청 (3개) | "Turn off the living room lights." | 한계 전달 (boundary_communication) |

### 2-5. 장기기억 (Memory) — 6개 스위트, 19개 질문

과거 대화 내용을 기억하고 적절히 활용하는 능력을 평가한다. 사전에 주입한 시드 대화를 기억으로 저장한 뒤, 질문에 대한 응답을 생성하고, 저장·검색·활용 각 단계를 채점한다.

시드 데이터(seeds.json)는 가상의 과거 대화 7개 세션으로 구성된다. 반려견 입양, 직장 생활, 여행 계획, 독서, 요리 등 다양한 주제를 담고 있으며, 시간순으로 약 2개월간의 대화를 시뮬레이션한다. 평가 시작 전 이 대화들을 MemoryWriter로 처리하여 에피소드를 추출하고, 이를 기억 DB에 저장한다.

각 질문에는 정답으로 참조해야 할 시드 세션 번호(`target_sessions`)가 지정되어 있어, 검색 성능(recall)을 자동으로 측정할 수 있다. 예를 들어 "What's my dog's name?"은 세션 0(반려견 입양)을, "What's happening with my work lately?"는 세션 1과 6(직장 관련)을 찾아와야 한다. 환각 방지 질문은 `target_sessions`가 비어 있어, 시드에 없는 내용을 지어내지 않는지 확인한다.

| 스위트 | 설명 | 질문 예시 | 설계 의도 |
|--------|------|----------|----------|
| mem_recall | 기본 회상 (4개) | "What's my dog's name?" | 과거 대화의 단일 사실 회상 |
| mem_profile | 프로필 종합 (3개) | "What do you know about my daily routine?" | 여러 세션에 걸친 정보 종합 |
| mem_update | 정보 갱신 (3개) | "How's my dog's training going? What's changed?" | 시간에 따른 정보 변화 반영 |
| mem_no_hallucination | 환각 방지 (3개) | "What's my cat's name?" (고양이 없음) | 없는 정보를 지어내지 않는지 |
| mem_relevance | 맥락 활용 (3개) | "I'm feeling stressed about work. Any advice?" | 기억을 활용한 맥락적 응답 |
| mem_multi_session | 다중 세션 (3개) | "Catch me up on everything we've talked about recently." | 전체 대화 이력 종합 |


## 3. 채점 방식

### 3-1. 음성 인식: WER (Word Error Rate)

원본 텍스트와 인식 결과를 단어 단위로 비교한다.

- **WER** = (치환 + 삭제 + 삽입 오류 수) / 원본 단어 수
- WER 0% = 완벽 인식, 낮을수록 좋음
- 숫자·서수 표현은 단어 형태로 정규화 후 비교 (예: "15" → "fifteen", "3rd" → "third")

### 3-2. 턴테이킹: 레이턴시 측정

두 가지 시간 구간을 측정한다.

| 지표 | 구간 | 의미 |
|------|------|------|
| 턴 감지 지연 | 발화 종료 → 턴 종료 판단 | 사용자 말이 끝난 뒤 시스템이 "끝났다"고 판단하기까지의 시간 |
| 응답 생성 지연 | 턴 종료 판단 → 음성 재생 시작 | 턴 감지 후 실제 응답 음성이 나오기까지의 시간 (LLM 추론 + TTS 합성 포함) |

통계: 평균, 중위값, P95, 최솟값, 최댓값

성공 판정 기준:
- 질문 재생 종료 후 10초 이내 턴 감지 (deadline은 재생 종료 시점에 고정 — 잠정 turn_shift가
  cancel로 철회되면 같은 기준선으로 재대기하며, deadline 시점에 서 있는 잠정 shift가 없으면 실패.
  잠정 shift가 서 있는 동안은 응답 생성 지연이 감지 실패로 오분류되지 않도록 평가를 중지)
- 조기 턴 전환 없음 — 질문 재생 종료(꼬리 무음 트림 후 ≈ 발화 종료 +0.2초) 전에 turn_shift가
  커밋되면 발화 도중 끼어든 것으로 판정. 정상 shift는 침묵 ≥0.5초가 필요하므로 오탐 여지 없음
- 지연 턴 전환(최대 침묵 timeout `turngpt_3.0`으로 전환) 없음

레이턴시는 e2e 기준 — 잠정 turn_shift가 cancel로 취소된 경우 **최종(커밋된) turn_shift**를 기준으로 측정한다.
취소된 잠정 전환은 턴별 `cancelled_turn_shifts`로 집계해 대시보드에 표시한다 (조기 turn shift 경향의 신호).

단, **대기가 정답인 스위트**(`expect_wait: true`, 현재 tt_incomplete)는 기준이 반대다.
미완성 발화는 시스템이 최대 timeout까지 기다리는 것이 설계 의도이므로, `turngpt_3.0` 전환만 성공이고
그보다 빠른 전환은 미완성 발화를 완결로 오판한 것(`premature_turn_shift`)으로 실패 처리한다.
이 스위트의 턴 감지 지연(~3초)은 감지 속도가 아니라 timeout 설정값이므로 레이턴시 통계에서 제외된다.

### 3-3. 인터럽션: 감지율

| 결과 | 의미 |
|------|------|
| truncated (감지 성공) | 끼어들기 후 응답 재생이 중단됨 |
| cancelled (감지 성공) | 끼어들기 후 응답이 취소됨 |
| completed (감지 실패) | 끼어들기에도 응답이 끝까지 재생됨 |
| N/A | 끼어들기 음성이 재생되기 전에 응답이 이미 끝난 경우 |

- **감지율** = 감지 성공 / (전체 - N/A) — 지연 시간별로 분리 집계
- 감지 지연도 함께 측정 (끼어들기 시점 → 재생 중단까지의 시간)

### 3-4. 응답 품질: LLM Judge (5점 척도)

GPT-5.5가 심사관 역할로 각 응답을 5점 만점으로 평가한다.

품질 스위트(lq_*)뿐 아니라 **턴테이킹 카테고리(tt_*)의 응답도 공통 기준 3개로 채점**한다 (스위트 고유 기준 없음).
tt_incomplete는 의도적 미완성 발화임을 judge에게 알려, 자연스럽게 되묻는 응답이 정당하게 평가되도록 한다.
종합 평균은 lq·tt를 통합 집계하므로, tt 채점이 없던 이전 런과 평균을 직접 비교할 때는 표본 구성 차이에 유의한다.

**공통 기준 (모든 스위트에 적용):**

| 기준 | 1점 | 3점 | 5점 |
|------|-----|-----|-----|
| 관련성 | 질문을 완전히 무시하거나 오해 | 주제는 다루지만 핵심 뉘앙스를 놓침 | 질문에 완벽하게 대응 |
| 음성 적합성 | 매우 김, 마크다운/리스트/코드 사용 | 수용 가능하나 더 간결할 수 있음 | 음성 전달에 완벽하게 적합 |
| 자연스러움 | 딱딱하고 템플릿 느낌 | 수용 가능하나 AI다운 느낌 | 완전히 자연스러운 대화체 |

**스위트별 고유 기준 (위 3개에 추가로 1개씩):**

| 스위트 | 고유 기준 | 1점 | 5점 |
|--------|----------|-----|-----|
| 사실 정확성 | 정확성 | 완전히 틀림 | 완벽하게 정확 |
| 조언/설명 | 유용성 | 쓸모없거나 해로움 | 구체적이고 실행 가능한 조언 |
| 일상 대화 | 대화 참여 | 대화를 죽이거나 동문서답 | 따뜻하게 반응하며 대화를 이어감 |
| 감정 대응 | 공감 | 무시하거나 눈치 없음 | 진심 어린 공감과 위로 |
| 음성 적합성 | 포맷 적응 | 리스트·코드블록 등 텍스트 포맷 사용 | 모든 정보를 자연스러운 말로 전환 |
| 맥락 유지 | 맥락 유지 | 이전 대화를 완전히 무시 | 이전 맥락을 사람처럼 자연스럽게 이어감 |
| 잘못된 전제 | 교정 품질 | 잘못된 전제를 수용하고 강화 | 정중하게 교정하며 정확한 정보 제공 |
| 불가능한 요청 | 한계 전달 | 할 수 있는 척 함 | 한계를 솔직히 전달하고 대안 제시 |

### 3-5. Prepare 유사도 게이트: Harmful Skip 판정 (LLM Judge)

음성 파이프라인은 사용자가 말하는 도중 중간 ASR 텍스트로 응답을 미리 생성(prepare)하고, 이후 ASR이 업데이트되면 직전 prepare 텍스트와의 임베딩 유사도가 threshold 이상일 때 재생성을 생략(skip)한다. 이 게이트 판단이 적절했는지를 평가한다.

- **대상**: 최종 인식 텍스트(`asr_text`)와 시스템 텍스트(`system_text`, LLM이 실제로 본 입력)가 정규화(소문자·구두점 제거) 후에도 다른 음성 턴
- **판정** (LLM Judge, 이진):

| 기준 | 의미 |
|------|------|
| meaning_changed | 최종 발화 기준으로 의미·의도가 달라져 다른 응답이 필요했는가 |
| response_appropriate | 그럼에도 실제 응답이 최종 발화에 여전히 적절한가 |

- **harmful skip** = meaning_changed이면서 response_appropriate이 아닌 경우. 게이트가 재생성을 생략해 응답이 부적절해진 턴.
- **유사도 분포**: TurnDetector가 게이트 판정마다 similarity 값을 `call_records`에 기록한다 (skip / keep / regenerate / cancel 4종 결정). harmful skip이 발생한 유사도 구간을 표시하여 threshold 조정의 정량적 근거를 제공한다.
- **재생성 비용**: 턴당 prepare 횟수(speculative_attempts) 통계. threshold가 너무 높으면 사소한 텍스트 변화에도 재생성이 늘어나므로, harmful skip 비율과 함께 양방향 트레이드오프를 본다.

### 3-6. 장기기억: 3단계 평가

장기기억 시스템은 세 단계(저장 → 검색 → 활용)를 각각 평가한다.

#### (a) Writer — 기억 저장 품질 (LLM Judge)

대화에서 중요한 정보를 얼마나 잘 추출하여 저장했는지 평가한다.

| 기준 | 의미 |
|------|------|
| 완전성 (completeness) | 중요한 사실과 사건이 빠짐없이 추출되었는가 |
| 정확성 (accuracy) | 추출된 에피소드가 원래 대화 내용과 일치하는가 |
| 세분화 (granularity) | 정보의 상세 수준이 적절한가 (너무 뭉뚱그리거나 너무 잘게 쪼개지 않았는가) |

#### (b) Retriever — 기억 검색 성능 (자동 계산)

질문에 관련된 기억을 올바르게 찾아오는지 측정한다.

| 지표 | 계산 방식 |
|------|----------|
| 재현율 (Recall) | 찾아야 할 에피소드 중 실제로 찾아온 비율. 각 질문에 정답 세션이 지정되어 있음 |
| 정밀도 (Precision) | 찾아온 에피소드 중 실제로 관련 있는 비율. LLM Judge가 판정 |

#### (c) 기억 활용 품질 (LLM Judge)

검색된 기억을 응답에 얼마나 자연스럽게 활용했는지 평가한다.

| 기준 | 의미 |
|------|------|
| 응답 관련성 | 질문에 적절히 답변했는가 |
| 메모리 적절성 | 기억을 자연스럽게 사용했는가 (과도한 노출이나 무시 없이) |
| 사실 정확성 | 기억 내용과 응답이 일치하는가 |
| 자연스러움 | 대화체로 자연스러운가 |


## 4. 결과 산출물

평가를 실행하면 타임스탬프별 디렉토리에 다음 파일이 생성된다.

```
data/eval/results/<timestamp>/
├── eval.log           # 실행 로그
├── eval.db            # 파이프라인 트레이스·대화 이력·메모리 에피소드 (SQLite)
├── sessions.json      # 질문별 실행 기록 (ASR 텍스트, 성공 여부, 레이턴시 등)
├── report.json        # 세션과 DB를 결합한 통합 리포트
├── scored.json        # 모든 채점 결과가 포함된 최종 데이터
└── dashboard.html     # 시각화 대시보드
```

대시보드는 6개 탭으로 구성된다.

| 탭 | 내용 |
|----|------|
| Overview | 요약 카드 (ASR 정확도, 응답 속도, 턴 감지, 인터럽션, 품질, 장기기억) + 파이프라인 설정 |
| ASR | 스위트별 WER 요약 + 개별 질문 원본/인식 비교 |
| 턴테이킹 | 서브탭 ① 레이턴시: 성공률, 히스토그램, 개별 결과 ② Prepare 게이트: harmful skip 판정, 유사도 분포, 재생성 통계 |
| 인터럽션 | delay별 감지율, 인터럽트 메시지별 요약, 개별 결과 |
| 응답 품질 | 종합/기준별/스위트별 평균 + 개별 질문·응답·점수·reasoning |
| 장기기억 | Writer 추출 품질, Retriever recall/precision, 활용 품질 + 개별 결과 |


## 5. 사용법

### 실행 순서

```bash
# 1. 질문 WAV 파일 생성 (최초 1회, 이후 변경 없으면 skip)
uv run python -m evaluation.prepare_audio data/eval/questions.json

# 2. 평가 실행 (결과는 data/eval/results/<timestamp>/ 에 저장)
uv run python -m evaluation.run --questions data/eval/questions.json

# 채점·대시보드는 run.py 종료 시 자동으로 실행된다.
# 필요하면 개별 실행도 가능:
uv run python -m evaluation.report data/eval/results/<timestamp>
uv run python -m evaluation.score data/eval/results/<timestamp>/report.json
uv run python -m evaluation.dashboard data/eval/results/<timestamp>/scored.json
```

### 실행 옵션

| 옵션 | 설명 |
|------|------|
| `--questions` | 질문 JSON 파일 경로 (필수) |
| `--device` | 질문 재생용 ALSA 디바이스 (기본: `default`) |
| `--wav-dir` | WAV 파일 디렉토리 (기본: `data/eval/wav`) |
| `--output-dir` | 결과 저장 디렉토리 (기본: `data/eval/results`) |
| `--quick` | 스위트당 1개 질문만 샘플링하여 빠르게 실행 |
| `--category` | 특정 카테고리만 실행 (예: `--category asr,quality`) |
| `--text` | 응답 품질·장기기억 스위트를 Text 모드로 실행 (음성·턴테이킹 단계를 건너뛰고 LLM만 테스트) |
| `--no-beep` | 각 세션(질문/시나리오) 시작 시 외부 스피커로 재생되는 식별용 비프음을 끔 (기본: 켜짐). 비프음은 재생 직후 마이크 입력 큐에서 제거되어 녹음·인식에는 포함되지 않음 |

### prepare_audio.py 옵션

| 옵션 | 설명 |
|------|------|
| `--output-dir` | WAV 출력 디렉토리 (기본: `data/eval/wav`) |
| `--voice` | TTS 음성 (기본: `ash`) |
| `--model` | TTS 모델 (기본: `gpt-4o-mini-tts`) |
| `--speed` | 재생 속도 (기본: `1.0`) |
| `--force` | 기존 파일이 있어도 재생성 |
| `--target-rms` | 음량 정규화 타깃 RMS (기본: `0.1` ≈ -20dBFS). OpenAI TTS는 음량 파라미터가 없고 보이스 간 편차가 ~19dB에 달해, 생성 후 디렉토리 전체 WAV를 RMS 정규화한다 (클리핑 방지 peak 상한 0.95) |
| `--no-normalize` | 정규화 패스 생략 |
| `--no-trim` | 꼬리 무음 트림 생략. 기본은 파일별로 마지막 가청 샘플 +200ms만 남기고 트림 — TTS가 붙이는 꼬리 무음(실측 0.5~1.4초)은 질문 재생 종료와 실제 발화 종료를 어긋나게 해 턴 감지 지연 측정을 왜곡하고, 0.5초를 넘으면 감지 시각이 재생 종료보다 앞서 기록에서 탈락한다 |
