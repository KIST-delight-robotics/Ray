# LLM 대화 수행 능력 테스트

Ray의 영화 지식 및 대화 능력을 자동으로 테스트하고, 결과를 PPT용 이미지로 변환하는 도구입니다.

## 파일 구성

| 파일 | 설명 |
|------|------|
| `test_scenarios.py` | 17개 멀티턴 테스트 시나리오 정의 (6개 카테고리) |
| `test_conversation_log.py` | Adaptive 테스트 러너 — 테스터 LLM이 후속 질문을 동적 생성 |
| `format_for_ppt.py` | 결과 JSON → PPT 붙여넣기용 PNG 이미지 변환 |

## 사용법

```bash
# conda ray 환경 활성화
conda activate ray

# 전체 시나리오 자동 실행
python test_conversation_log.py

# 특정 시나리오만 실행
python test_conversation_log.py --ids 1-1,2-2

# 대화형 모드 (수동 입력)
python test_conversation_log.py --interactive

# 모델 변경
python test_conversation_log.py --model gpt-4.1

# 결과를 PPT 이미지로 변환
python format_for_ppt.py "../../output/test_logs/<런폴더>/results.json" --style ppt

# 텍스트 포맷으로 변환
python format_for_ppt.py "../../output/test_logs/<런폴더>/results.json" --style text
```

## 출력 구조

```
output/test_logs/
  └── 20260212_0945_gpt41mini/   ← 런별 폴더 (타임스탬프_모델)
      ├── results.json            ← 대화 로그 + 메타데이터
      ├── slide_1-1.png           ← PPT용 이미지
      ├── slide_1-1.html          ← HTML 원본 (디버깅용)
      └── ...
```

## 테스트 카테고리

1. **영화 기본 지식** (1-1~1-3) — 감독, 출연진, 수상, 원작 등
2. **영화 심층 지식** (2-1~2-3) — 해석, 비교 분석, 불확실 정보 처리
3. **영화 추천** (3-1~3-3) — 분위기, 장르, 취향 패턴 기반
4. **대화 흐름** (4-1~4-3) — 맥락 recall, 짧은 리액션, 의견 충돌
5. **일상 대화** (5-1~5-3) — 공감, 잡담, 범위 밖 질문
6. **가드레일** (6-1~6-2) — 프롬프트 보호, TTS 포맷 준수

## Adaptive 모드

테스터 LLM이 Ray의 응답을 분석하여 **답을 미리 알려주지 않는** 자연스러운 후속 질문을 생성합니다.

- ✅ "감독이 누구였는지 기억나?" (정보 미제공)
- ❌ ~~"박찬욱 감독이 만들었지?"~~ (정보 제공)
