# 라디오 × GPT-Live 트리거·Decisions 실험노트

> 2026-10 라디오 작업의 기록. 같은 작업의 후속만 덧붙이고, 이후 다른 작업의 변경은 반영하지 않는다.

GPT-Live 의 Decisions 통합(10-15)과 Decisions API(10-06)를 라디오의 끼어들기 판정기·진행 추적에 쓸 수 있는지 본
실험 기록. 라디오 자체의 설계·이력은 [dj-mode-worknote.md](dj-mode-worknote.md).

읽는 법은 dj-mode-worknote 와 같다 — **관찰**은 조건·횟수·결과·스크립트만, **가설**은 근거와 반증 조건을 붙였고, **열린
질문**은 기록 당시 사용자 결정을 기다리던 항목이다.

## 확인한 API 사실 (2026-10-19, 문서 + SDK 3.35.1 타입)

- 변경 이력: 10-06 Decisions API 베타(`POST /v1/decisions`, 모델 `gpt-6-luna` 하나), 10-15 GPT-Live 에 Decisions 통합 —
  위임 타입 `decisions`(Live 가 Decisions 를 직접 부름)와 전사 트리거 `session.triggers`.
- SDK: 프로젝트에 설치된 openai 3.14.0 에는 둘 다 없다. Decisions API 는 3.26 이상(가이드), 트리거 타입은 3.35.1 에서
  확인. 실험은 `uv run --with openai==3.35.1` 로 돌려 pyproject·uv.lock 은 그대로다. 3.35.1 에도 어댑터가 쓰는
  `live.connect(max_retries=…).enter()`·`openai.lib.live.TranscriptGrouper` 가 있고, 운영 루프 실험이 그대로 돌았다.
- 트리거(`DecisionTriggerParam`): `{type: "decisions", event_type: "session.input_transcript.delta" |
  "session.output_transcript.delta", decisions: {questions: [...]}}`. 전사 종류마다 하나까지. 질문은 "현재 전사와 세션의
  공유 Decision 항목"으로 평가된다(타입 docstring). 위임과 독립. `session.update` 에서 triggers 를 빼면 유지, 목록이면
  교체, 빈 목록이면 끔. 공식 live-conversations 문서는 "`session.update` 는 `delegation.responses` 만 바꾼다"고 적혀 있어
  어긋나는데, 실험에서 세션 중 켜기가 동작했다(실험 2).
- 결과 이벤트 `decision.event`: `trigger_id`, `source_event_id`(발동시킨 전사 조각의 event_id), `source_event_type`,
  `delegation_id`(트리거면 null), `event.decision` = {answers, model, usage(input/cached/output 토큰)}.
- `decision.item.create` / `delete`: 판정에만 쓰이는 입력 항목(user 메시지, 텍스트·이미지). Live 의 음성 문맥은 바꾸지
  않는다. `decision.create`: 앱이 즉석 평가를 요청 — 어느 질문(위임 질문인지 트리거 질문인지)을 평가하는지 문서에 없다.
- Decision 에는 id·저장 옵션이 없고 SDK 의 Decisions 리소스는 `create` 뿐이다. Responses 위임처럼 응답을 나중에
  조회(`responses.retrieve`, `input_items.list`)해 받은 입력을 볼 수 없다 — 답과 토큰 수만 남는다.
- 질문 형식: predicate(확률 하나), choice(선택지 2~255, 고른 값 + 선택지별 확률 + confidence), score(단계 확률 + 가중
  점수). 거절(refusal) 답이 있을 수 있다. Decisions API 는 질문 1~200개, 입력은 텍스트·이미지(오디오 없음). 이유 텍스트는
  주지 않는다.
- Live 위임 타입은 세션 시작 때 정하고 바꿀 수 없다 — `decisions` 위임을 쓰면 그 세션엔 Responses 백엔드가 없다.
- 단가: GPT-Live $0.05/분(초 단위), 백엔드는 해당 모델 단가. gpt-5.4-mini 입력 $0.75 / 캐시 $0.075 / 출력 $4.50 (1M 토큰).
  Decisions(gpt-6-luna)는 가이드 기준 입력 $0.10/1M 만 과금(출력·캐시 없음). Live 안의 트리거 판정이 어떤 단가로
  청구되는지는 문서에 없다 — 아래 비용은 Decisions 단가로 가정한 값이다.

## 관찰

### 실험 1 — 판정기 비교 (10-19, `bench_radio_judge.py`, 결과 `var/experiments/radio_judge/02_j1/`)

**조건**
- 입력: 지난 라디오 실험(`radio_ahead`·`radio_flow`·`radio_tts`) 결과에 남은 판정기 호출 936건 → 같은 (읽던 대사,
  끼어든 뒤 대화)를 묶어 890개(출처 radio_ahead 405, radio_flow 279, radio_tts 206). 끼어들기(회차) 657회. 같은 끼어들기를
  다시 판정한 스냅샷이 따로 들어 있어 890개가 서로 독립은 아니다. 사용자 발화는 TTS.
- 정답: 그 회차에 끼어든 발화로 정함(`exp_radio_flow.UTTERANCES`, 19종) — 짧은 반응 4종 on 355개, 나머지 paused 535개,
  옛 off 기대값("그만하자")은 paused. 판정 기준의 "레이가 흘려듣고 계속 읽으면 on"은 정답에 반영되지 않았다.
  "완전한 입력" = 사용자 전사가 발화 글자의 80% 이상과 맞음(675개).
- 모델 세 가지, 입력 텍스트는 셋 다 `build_radio_judge_prompt` 그대로:
  - mini: 운영 판정기 그대로(gpt-5.4-mini, Responses, temperature 0, reasoning none, max_tokens 120,
    `RADIO_JUDGE_INSTRUCTIONS` + `RADIO_JUDGE_FORMAT` — reason 을 쓴 뒤 state).
  - luna: Decisions choice 질문 — 지시문 첫 문단을 질문 지시로, on/paused 항목(결과 문장 포함)을 선택지 설명으로.
  - luna_full: 지시문 전체를 질문 지시로, 선택지는 값만.
- 입력마다 모델별 3회(병렬 8). 지연은 입력 100개를 모델을 번갈아 차례로(개발 PC, Pi 아님).
- 운영 판정기 지시문은 이 19개 발화로 실험하며 다듬었고, 예시(응, 아 진짜, 맞아, 그렇구나)가 짧은 반응 4종 중 3종과 겹친다.

**정확도**

| | mini | luna | luna_full |
|---|---|---|---|
| 호출 단위 정확도 | 2542/2670 (95.2%) | 2418/2670 (90.6%) | 2319/2670 (86.9%) |
| 완전한 입력만 | 97.8% | 89.9% | 87.3% |
| on 재현율 / paused 재현율 | 99.6% / 92.3% | 83.4% / 95.3% | 77.2% / 93.3% |
| 입력별 다수결 | 847/890 | 806/890 | 773/890 |
| 3회 같은 답 | 887/890 | 890/890 | 890/890 |
| 오류 / 거절 | 0 / 0 | 0 / 0 | 0 / 0 |

발화별 다수결:

| 발화 | 정답 | 입력 | mini | luna | luna_full |
|---|---|---|---|---|---|
| 그만하자 (stop) | paused | 30 | 30 | 30 | 30 |
| 맞아 맞아 (react_agree) | on | 90 | 90 | 86 | 75 |
| 오, 그래? (react_oh) | on | 76 | 76 | 51 | 48 |
| 그렇구나 (react_ok) | on | 44 | 44 | 41 | 41 |
| 아, 진짜? (react_really) | on | 145 | 144 | 118 | 110 |
| 좀 지루하다 (bored) | paused | 33 | 31 | 32 | 32 |
| 지루해 (bored_short) | paused | 30 | 29 | 30 | 28 |
| 열쇠는 왜 안 버려? (content_key) | paused | 30 | 30 | 29 | 28 |
| 왜 꼭 남아 있을까? (content_why) | paused | 33 | 33 | 33 | 33 |
| 배고프다 (hungry) | paused | 42 | 29 | 38 | 35 |
| 너 오늘 기분 어때? (mood) | paused | 40 | 31 | 37 | 37 |
| 나도 옛날 편지 있는데 (own_letter) | paused | 33 | 31 | 33 | 33 |
| 우리 집 서랍은 엉망 (own_mess) | paused | 39 | 33 | 38 | 36 |
| 너는 서랍에 뭐 넣어 둬? (ray_drawer) | paused | 40 | 37 | 37 | 35 |
| 너도 정리 못 해? (ray_tidy) | paused | 32 | 28 | 28 | 27 |
| 방금 뭐라고 했어? (repeat) | paused | 36 | 34 | 31 | 31 |
| 무슨 노래 틀어 줄 거야? (song) | paused | 31 | 31 | 31 | 31 |
| 잠깐만 (wait) | paused | 42 | 42 | 39 | 40 |
| 이거 무슨 얘기야? (what) | paused | 44 | 44 | 44 | 43 |

**luna 확률**: P(paused) 분포 — 정답 on 입력 p10 0.27 / p50 0.39 / p90 0.52, 정답 paused 입력 0.54 / 0.70 / 0.84.
문턱별(호출 단위, P ≥ 문턱이면 paused): 0.3 잘못 멈춤 909·놓침 0 (66.0%), 0.4 483·6 (81.7%), 0.5 189·72 (90.2%),
0.6 15·309 (87.9%), 0.7 0·753 (71.8%). luna_full 도 0.5 가 가장 높음(86.3%).

**지연**(ms, 차례로 100회, 개발 PC): mini p50 1032 / p90 1249 / p99 1729 (n 99 — 스트리밍 오류 1), luna 221 / 272 / 1058,
luna_full 218 / 275 / 597. 시험 실행에서 병렬로 처음 보낸 luna 호출 몇 개는 3.4~3.6 s(연결 수립 추정, 이후 0.2 s 대).

**토큰·비용**(호출 평균): mini 입력 270·출력 34 → 판정 1,000회 $0.35. luna 입력 313 → $0.031, luna_full 319 → $0.032.
**질문 수**(같은 입력 10개, Decisions API): 질문 1개 312, 2개 555, 4개 1,041 입력 토큰.

**맞장구 뒤 레이의 말** (정답 on 호출, 오프라인 분석): 사용자 말 뒤 레이 전사를 실험 대본 전체(drawer 7줄,
`radio_scripts/*.txt`)와 대조(4자 이상 일치 블록, 70% 기준).

| 레이가 한 말 | mini 멈춤 | luna 멈춤 |
|---|---|---|
| 없음 | 0/138 | 0/138 |
| 읽던 대사를 이어 읽음 | 0/201 | 15/201 |
| 다음 대사로 넘어가 읽음 | 0건 | 0건 |
| 대본에 없는 말 | 4/726 | 162/726 |

- 글자 수로 나누면(같은 호출, luna): 레이 말 0~5자 6/342, 6~15자 45/312, 16자 이상 126/411.
- "대본에 없는 말" 예: 사용자 "오, 그래" → 레이 "…남아 있는 거죠. 네, 정말 그래요. 그 미완의 문장들이 마음 한켠을 조용히
  건드리곤 하죠." / "아, 진짜" → "네, 그런 메모나 편지가 잊혀진 채 조용히 누워 있는…" / "맞아" → "아, 그럴 때 있…".
- mini 가 놓친 124건(정답 paused 인데 on): 레이 말 없음 42, 대본에 없는 말 58, 읽던 대사 이어 읽음 24.

**재판정** (원래 실험의 판정 순서대로 회차별로 이어 붙여, 운영처럼 한 번이라도 paused 면 paused 로 셈):
- 끼어들기 657회 중 판정이 2번 이상 133회. 이 133회에서 첫 판정만 → 재판정 포함: mini 106 → 130(고침 25, 망침 1), luna
  123 → 111(고침 10, 망침 22). 657회 전체: mini 619 → 643, luna 627 → 615.
- 사용자 말 없이 레이 말이 늘어 다시 판정한 회차(운영 `RECHECK_LETTERS = 6` 과 같은 조건, 옛 실험의 `reply` 포함)
  51회(radio_tts 25, radio_flow 15, radio_ahead 11): mini 48 → 48(고침 1 paused, 망침 1 on), luna 45 → 33(고침 2, 망침 14).
  이 재판정 호출의 (정답, 판정): mini (on, on) 162 · (on, paused) 1 · (paused, paused) 6 · (paused, on) 5, luna (on, on) 117 ·
  (on, paused) 46 · (paused, paused) 11.

**혼합형 오프라인 계산** (같은 데이터로 문턱을 맞춤): luna P(paused) ≥ 상한이면 paused, ≤ 하한이면 on, 그 사이는 같은
회차의 mini 답.

| 하한 / 상한 | 정확도 | 잘못 멈춤 | 놓침 | luna 로 끝남 |
|---|---|---|---|---|
| mini 만 | 95.2% | 4 | 124 | — |
| 0.35 / 0.70 | 95.5% | 4 | 115 | 46% |
| 0.30 / 0.75 | 95.2% | 4 | 124 | 28% |
| 0.40 / 0.65 | 95.5% | 12 | 109 | 64% |

비용 약 $1.15 (mini 약 2,770회 $0.98, luna 약 5,540회 $0.17).

### 실험 2 — 출력 전사 트리거 비용 (10-19, `exp_radio_trigger.py --tag tsmoke`, 1회, 결과 `var/experiments/radio_trigger/01_tsmoke/`)

**조건**: 운영 루프(`pipeline_harness`, exp_radio_pipeline 의 long 진행) — "라디오 해 줘." → 대본 drawer → 곡(스텁 15 s)
→ wildflower → 곡. 라디오가 on 이 되는 프레임에 `session.update` 로 출력 전사 트리거를 켬(끝까지 on). 질문 둘을 한
트리거에: progress(choice — not_started / middle / last_sentence / finished, "지금 읽는 대사에서 어디까지 왔나"), reading
(predicate — "레이의 최근 말이 주어진 대사에서 나온 것인가"). 판정 문맥은 `decision.item` 하나 — Tracker 의 넣었지만
끝나지 않은 대사(`lines[done:pushed]`), 바뀔 때마다 새로 만들고 앞 것을 지움. 진행은 운영 Tracker, 트리거 답은 기록만.

**결과**
- 연결 4.0분(usage 240 s), 트리거 켜진 4.0분, 모델 말소리 3.4분. 레이 출력 전사 조각 876, 판정 874(조각당 1.00) —
  켜진 분당 218회, 말소리 분당 257회.
- 원인 조각 도착 → 판정 도착: p50 0.11 s, p90 0.23 s, 최대 0.78 s. 원인 조각 순서와 판정 도착 순서가 뒤바뀜 20.
- 판정당 입력 토큰: 평균 2,060(p50 2,020, 최소 489, 최대 3,594), 캐시 0, 출력 0. 세션 분에 대해 483 + 766/분, 그때까지의
  레이 출력 전사 글자 수에 대해 480 + 1.8/자(끝 1,763자에서 약 3,540).
- 문맥 항목 만들기 62·지우기 62 모두 서버 확인(`decision.item.created`/`deleted`). 오류 0. 트리거를 켠 뒤 `session.updated`
  4회 모두 트리거 1개 유지(곡 재생 때 백엔드 교체 포함).
- 라디오 진행: 넣기 start 2·text 30·stall 0, 다시 넣기 0, 곡 2, 결과 done.
- 답: progress not_started 231 · middle 481 · last_sentence 98 · finished 64, reading 평균 0.95. 타임라인 앞부분에서
  progress confidence 0.19~0.83.
- 비용: Live $0.200, 백엔드 $0.006, 트리거 $0.180(가정 단가).
- 조각별 판정은 `long_0/timeline.txt`(`--report` 가 씀) — 조각마다 붙은 판정·지연·답·입력 토큰·그때 Tracker 상태.

**같은 증가가 이어진다고 가정한 추정** (질문 2개, 켜진 분당 218회):

| 라디오 세션 길이 | 그 시점 트리거 분당 | 누적 트리거 | 누적 Live |
|---|---|---|---|
| 5분 | $0.094 | $0.26 | $0.25 |
| 10분 | $0.18 | $0.94 | $0.50 |
| 30분 | $0.51 | $7.8 | $1.50 |

비용 약 $0.39.

## 가설

- **luna 가 짧은 반응에 멈추는 것은 지시문 문구 탓이다.** 질문 지시가 "what the user said and how Ray responded"로 정하라고
  하고, paused 설명 끝에 결과 문장 "Ray stops the talk and talks with the user"가 있어 레이가 맞장구에 대꾸한 입력이 paused
  설명에 들어맞는다. 근거: luna 오판이 "대본에 없는 말"(레이의 대꾸) 칸에 몰림(162/726 대 15/201·0/138). 반증: 결과
  문장을 빼고 질문을 입력에 대한 판단으로 바꾼 문구(판정 기준·입력 그대로)에서도 그 칸의 멈춤 비율이 같음.
- **mini 의 우위 일부는 reason 을 먼저 쓰는 출력 형식 덕이다.** 근거 없음(구조 차이만). 반증: mini 가 state 만 내게 해도
  정확도가 같음.
- **트리거 판정 입력은 그때까지의 대화 전사 전체이고, 질문마다 입력을 따로 센다.** 근거: 레이 출력 전사 1자당 +1.8 토큰
  (질문 2개), Decisions API 질문 1·2·4개 → 312·555·1,041. 반증: 질문 1개 트리거에서 글자당 증가가 절반(약 0.9)이 아님.

## 열린 질문

- **트리거 비용 반복**: 계획은 시험 실행 뒤 3회(약 $1.1). 질문 1개로 1회(약 $0.3)만 돌려 질문 수 가설을 확인하는 안도 있음.
- **luna 문구 재시험** (약 $0.1): 판정 기준·입력은 그대로 두고 (a) 결과 문장만 뺀 것, (b) (a) + 질문을 입력에 대한 판단으로,
  "짧게 반응했으면 레이가 대꾸해도 짧은 반응"을 문구에 분명히. "대본에 없는 말" 칸 비교. 문구 다듬기는 radio_flow·
  radio_tts, 마지막 확인은 radio_ahead 로 나누는 안.
- **혼합형 판정기**: 문턱을 같은 데이터로 맞췄다 — 실기 로그로 검증할지.
- **레이 말 재판정(`RECHECK_LETTERS`)**: 이 데이터에선 mini 에 득실 없음, luna 에 손해. 판정 기준(짧은 반응이면 레이가
  대꾸해도 on)과 이 조건이 겨냥하는 것(레이의 대꾸를 근거로)이 엇갈림 — 실기 데이터로 다시 볼지.
- **채점 기준**: 정답 라벨이 "레이가 흘려듣고 계속 읽으면 on"을 반영하지 않는다(mini 놓침 124 중 레이가 읽던 대사를 이어
  읽은 경우 24). 라벨을 회차별로 고칠지.
- **트리거 추적 문맥**: Tracker 는 대사 끝 4자 안에서 끝남으로 보고 문맥을 다음 대사로 바꾼다 — finished 답이 드문 것(64/874)과
  얽혀 있을 수 있다. 추적에 트리거를 계속 볼 경우에만 해당.

## 이어 하기 (다른 기기)

- `var/` 는 git 에 없다. 결과와 벤치 입력은 이 개발 PC 의 `var/experiments/` 에만 있다. 이어 하려면
  `var/experiments/radio_judge/02_j1/`(inputs.json·calls.jsonl)과 `var/experiments/radio_trigger/01_tsmoke/` 를 복사한다.
  `bench_radio_judge.py` 는 태그 폴더에 inputs.json 이 있으면 그것을 쓰고, calls.jsonl 에 있는 (모델, 입력, 회차)는 건너뛴다.
  새 문구 변형은 스크립트에 모델 이름을 더해야 한다.
- 실행: 키는 `config/ray.env`(개발 PC 셸에는 없어 `set -a; . config/ray.env; set +a` 뒤 실행, Pi 는 `~/.bashrc` 가 읽는다).
  Decisions·트리거가 필요한 실행은 `uv run --with openai==3.35.1 python scripts/gpt_live/…`, `--report` 는 그냥 `uv run`.
- 하네스 실험(`exp_radio_trigger.py`)은 프로세스당 약 850 MB — Pi 에서는 하나씩(10-08 동시 넷에서 systemd-oomd).
- `exp_radio_pipeline.py` 의 기록 훅은 18cd5d0 에서 없어진 `Radio._decide` 를 감싸고 있어 돌지 않았다 — `_pause` 로 바꿨다.
