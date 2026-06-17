# [Pi 작업 지시] essentia V/A 타임라인 → 기계판독용 CSV 출력 모드 추가

> 이 문서는 Raspberry Pi(`ray_mk3@161.122.114.128`)에서 작업하는 LLM/담당자용 지시서다.
> 사전 맥락은 같은 번들의 `HANDOFF_PROMPT.md`에 있다. 그걸 먼저 읽고 이 문서를 수행하라.
> 환경: Raspberry Pi 5 (aarch64), Ubuntu, Python 3.12. 작업 루트 `~/Downloads/mood_analysis_bundle/`.

---

## 0. 한 줄 요약
`essentia_va_timeline.py` 에 **`--csv` 출력 모드**를 추가하라. 이 모드는 stdout으로
**오직 깨끗한 CSV만** 뱉어야 한다(ASCII 곡선·로그·진행상황 출력 금지). Windows 쪽이 SSH로
이 스크립트를 호출하고 **stdout을 그대로 파일로 캡처**해서 궤적생성 입력으로 쓴다.

## 1. 왜 이 작업이 필요한가 (전체 그림)
시스템은 두 머신으로 **역할 분리**되어 있다. 머신을 합치지 않는다.
- **Pi (여기)** = essentia V/A 모델이 유일하게 도는 곳. 곡당 1회 V/A 시계열만 계산해서 내보낸다.
- **Windows** = vibrato 궤적, 퍼커시브 온셋, 곡 구조(후렴) 분석, LED 합성. 빠른 개발 루프.

Windows 래퍼가 SSH로 Pi를 호출 → Pi가 CSV를 stdout으로 반환 → Windows가 캐시 후 궤적생성단에 입력.
이 흐름의 **유일한 인터페이스가 아래 CSV 계약**이다. 그래서 형식을 정확히 지켜야 한다.

Windows가 실제로 호출할 명령 (이걸로 직접 테스트하라):
```bash
cd ~/Downloads/mood_analysis_bundle && export LD_LIBRARY_PATH=$PWD/.venv/lib && \
./.venv/bin/python essentia_va_timeline.py <wav경로> deam --csv
```

## 2. CSV 출력 계약 (반드시 이대로)
`--csv` 가 인자에 있으면, stdout은 다음 형식의 **CSV 한 덩어리만** 출력한다:

```
time_sec,valence,arousal
0.00,5.12,4.83
1.49,5.30,5.61
2.97,4.95,6.10
...
```

규칙:
1. **첫 줄은 정확히** `time_sec,valence,arousal` 헤더.
2. 한 줄 = 패치 1개. `time_sec` = 해당 패치의 **시작 시각(초)**, 소수 2자리.
   (스크립트가 이미 패치별 시각을 계산하고 있다 — 그 값을 쓰면 된다. 시작/중심 중 무엇을 쓰든
    한 가지로 통일하고, 이 문서의 "시작 시각" 정의를 그대로 따른다.)
3. `valence`, `arousal` = essentia 1~9 스케일 원본 값, 소수 2자리. **정규화/반올림 금지.**
4. 기본 헤드는 `deam` (HANDOFF에서 시계열 분석용으로 채택). 인자로 `emomusic|deam|muse` 받으면 그대로 사용.
5. **stdout에는 CSV 외 아무것도 출력하지 마라.** 로그·경고·"loading model..."·tensorflow 메시지 등은
   **전부 stderr로** 보내라(`print(..., file=sys.stderr)`). stdout에 한 줄이라도 섞이면 Windows 파싱이 깨진다.
6. tensorflow/essentia가 import 시 stdout으로 뱉는 배너가 있으면, 그것도 stderr로 리다이렉트하거나
   `--csv` 모드에서 억제하라(예: `TF_CPP_MIN_LOG_LEVEL=3`, 또는 import 전후 stdout 임시 차단).

## 3. 기존 동작 보존
- `--csv` **없이** 실행하면 지금까지의 ASCII 곡선·고-arousal 구간 출력은 **그대로 유지**한다.
- 즉 `--csv`는 순수 추가 플래그다. 기존 사용법을 깨지 마라.

## 4. 인자 파싱
현재: `essentia_va_timeline.py <wav> [head]`
변경 후: `essentia_va_timeline.py <wav> [head] [--csv]`
- `head` 생략 시 `deam`.
- `--csv` 위치 무관(어디 와도 인식). argparse 권장.

## 5. 완료 기준 (이걸로 자가 검증)
1. `./.venv/bin/python essentia_va_timeline.py <테스트wav> deam --csv > /tmp/out.csv` 실행 시
   `/tmp/out.csv` 가 **헤더 1줄 + 패치 수만큼의 데이터 줄**로만 구성된다.
2. `head -3 /tmp/out.csv` → 헤더와 숫자 줄만 보이고, 로그/배너 한 줄도 없다.
3. `python3 -c "import csv; list(csv.DictReader(open('/tmp/out.csv')))"` 가 에러 없이 통과.
4. `--csv` 없이 실행하면 예전처럼 ASCII 곡선이 나온다(회귀 없음).
5. 패치 수·시각이 ASCII 모드의 시계열과 일치한다(같은 데이터, 형식만 다름).

## 6. 작업 후 회신할 것
- 수정한 파일과 변경 요지 1~2줄.
- 위 완료기준 1~3의 실제 출력 처음 5줄(복붙).
- 한 곡 처리 시간(warm 기준 대략).

## 7. 하지 말 것
- 모델·헤드·mel 계산 로직 변경 금지(형식 출력만 추가).
- valence/arousal 값 가공(정규화·클리핑·스무딩) 금지 — 원본 그대로. 스무딩은 Windows에서 한다.
- 곡 구조/후렴 라벨을 여기서 만들려 하지 마라 — 이 모델은 V/A 숫자만 낸다(HANDOFF 27번 줄). 구조분석은 Windows 담당.
