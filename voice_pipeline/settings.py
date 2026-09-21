"""파이프라인 전체에서 가정하는 오디오 형식 상수.

모든 오디오 처리 모듈(AudioInput, ASR, VAP, Wakeword, Orchestrator 등)이
이 상수들을 직접 import해서 사용한다. 시스템 전체가 동일한 PCM 형식을
가정하므로 한 곳에서 관리한다.

값 변경 시: 모든 의존 모듈에 영향. 마이크/모델/플랫폼 호환성 모두 확인 필요.
"""

from __future__ import annotations

from typing import Literal

# 대화 엔진. cascade = ASR + 턴테이킹 + LLM + TTS 체인, live = GPT-Live 하나로 대체(live_session.py)
ENGINE: Literal["cascade", "live"] = "live"

SAMPLE_RATE = 16000  # 샘플레이트 (Hz). Google STT, VAP 모델 등이 가정하는 값
CHANNELS = 1  # 마이크 채널 수 (mono)
SAMPLE_WIDTH = 2  # 샘플당 바이트 수 (16-bit PCM = 2). LINEAR16 인코딩
FRAME_DURATION_MS = 30  # 한 프레임 길이 (ms). turn_detector/orchestrator 시간축 단위

# C++ 재생 프로세스가 기대하는 출력 오디오 형식 (cpp/main.cpp AUDIO_SAMPLE_RATE 와 일치). mono 16-bit PCM.
# TTS 어댑터와 GPT-Live 세션은 이 레이트로 오디오를 만들어 브리지로 보낸다.
BRIDGE_SAMPLE_RATE = 24000

# Derived
FRAME_SIZE_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480
FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * SAMPLE_WIDTH * CHANNELS  # 960

# ---------------------------------------------------------------------------
# 여러 모듈이 공유하는 값
# ---------------------------------------------------------------------------

VAR_DIR = "var"  # 프로그램이 실행 중에 쓰는 파일의 루트 (DB, 로그, 캐시, 생성 오디오). 폴더 전체 gitignore
DEFAULT_DB_PATH = f"{VAR_DIR}/ray.db"  # history / memory / trace / call 스토어가 공유하는 SQLite 파일
DEVICE_SETTINGS_PATH = f"{VAR_DIR}/device_settings.json"  # 사용자가 말로 바꾼 볼륨·밝기 단계 (device_settings.py)

# 프롬프트 토큰 예산 — prompt.py(ContextBuilder, HistorySummarizer)와 wiring.py(요약 LLM max_tokens)가 함께 사용
HISTORY_TOKEN_BUDGET = 8192  # 히스토리 뷰 예산 (이월 + 요약 블록 + 라이브 턴)
SUMMARY_MAX_TOKENS = 512  # 롤링 요약 LLM의 max_output_tokens
