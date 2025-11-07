import os
import sys
from pathlib import Path

# --- 기본 설정 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

# --- 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_LOG_DIR = OUTPUT_DIR / "logs"
OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# 재생용 오디오 파일
AWAKE_FILE = ASSETS_DIR / "audio" / "awake.wav"
SLEEP_FILE = ASSETS_DIR / "audio" / "sleep.wav"

# --- 오디오 설정 ---
AUDIO_CONFIG = {
    'SAMPLE_RATE': 16000,
    'CHANNELS': 1,
    'AUDIO_DTYPE': "int16",
    'VAD_CHUNK_SIZE': 512,
    'PRE_BUFFER_DURATION': 0.3,     # STT에 전달될 사전 버퍼 길이 (초)
    'VAD_THRESHOLD': 0.5,           # VAD 민감도 (0.0 ~ 1.0, 높을수록 민감)
    'VAD_CONSECUTIVE_CHUNKS': 3,    # VAD가 음성으로 판단하기 위한 연속 청크 수.
    'VAD_RESET_INTERVAL': 20.0,     # 주기적으로 VAD 상태를 초기화하는 간격 (초)
}

# --- OpenAI 설정 ---
TTS_MODEL = "gpt-4o-mini-tts"
VOICE = "coral"
REALTIME_MODEL = "gpt-4o-mini-realtime-preview"
RESPONSES_MODEL = "gpt-4.1"

# --- 키워드 및 타임아웃 설정 ---
START_KEYWORD = "레이"
END_KEYWORDS = ["종료", "쉬어"]
ACTIVE_SESSION_TIMEOUT = 120.0 # 사용자 응답 없이 Active 모드가 유지되는 최대 시간 (초)

