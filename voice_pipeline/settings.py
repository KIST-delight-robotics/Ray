"""파이프라인 전체에서 가정하는 오디오 형식 상수.

모든 오디오 처리 모듈(AudioInput, ASR, VAP, Wakeword, Orchestrator 등)이
이 상수들을 직접 import해서 사용한다. 시스템 전체가 동일한 PCM 형식을
가정하므로 한 곳에서 관리한다.

값 변경 시: 모든 의존 모듈에 영향. 마이크/모델/플랫폼 호환성 모두 확인 필요.
"""

from __future__ import annotations

SAMPLE_RATE = 16000  # 샘플레이트 (Hz). Google STT, VAP 모델 등이 가정하는 값
CHANNELS = 1  # 마이크 채널 수 (mono)
SAMPLE_WIDTH = 2  # 샘플당 바이트 수 (16-bit PCM = 2). LINEAR16 인코딩
FRAME_DURATION_MS = 30  # 한 프레임 길이 (ms). turn_detector/orchestrator 시간축 단위

# Derived
FRAME_SIZE_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480
FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * SAMPLE_WIDTH * CHANNELS  # 960

# ---------------------------------------------------------------------------
# 여러 모듈이 공유하는 값
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = "data/ray.db"  # history / memory / trace / call 스토어가 공유하는 SQLite 파일

# 프롬프트 토큰 예산 — prompt.py(ContextBuilder, HistorySummarizer)와 wiring.py(요약 LLM max_tokens)가 함께 사용
HISTORY_TOKEN_BUDGET = 8192  # 히스토리 뷰 예산 (이월 + 요약 블록 + 라이브 턴)
SUMMARY_MAX_TOKENS = 512  # 롤링 요약 LLM의 max_output_tokens
