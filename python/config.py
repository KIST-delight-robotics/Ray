import os
import sys
from pathlib import Path

from typing import Any, Dict

# --- 기본 설정 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

# --- OpenAI 설정 ---
TTS_MODEL = "gpt-4o-mini-tts"
VOICE = "coral"
REALTIME_MODEL = "gpt-4o-mini-realtime-preview"
RESPONSES_MODEL = "gpt-4.1-mini"
SUMMARY_MODEL = "gpt-4.1-mini"
RESPONSES_PRESETS = {
    "gpt-5.1": {
        "model": "gpt-5.1",
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
    },
    "gpt-5-mini": {
        "model": "gpt-5-mini",
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
    },
    "gpt-5-nano": {
        "model": "gpt-5-nano",
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
    },
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
    },
}


# --- 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_LOG_DIR = OUTPUT_DIR / "logs"
OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- RAG 설정 ---
RAG_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_db"
RAG_COLLECTION_NAME = "movie_archive"
RAG_EMBEDDING_MODEL = "text-embedding-3-small"
RAG_CHUNK_SIZE = 700        # 한국어 최적화
RAG_CHUNK_OVERLAP = 100
RAG_TOP_K = 3               # 검색 결과 수

# 재생용 오디오 파일
AWAKE_FILE = ASSETS_DIR / "audio" / f"awake_{VOICE}.wav"
SLEEP_FILE = ASSETS_DIR / "audio" / f"sleep_{VOICE}.wav"
AWAKE_FILE_SCRIPT = "네, 부르셨어요?"
SLEEP_FILE_SCRIPT = "다음에 또 불러주세요."

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

# --- 키워드 및 타임아웃 설정 ---
START_KEYWORD = "레이"
END_KEYWORDS = ["종료", "쉬어"]
ACTIVE_SESSION_TIMEOUT = 30.0 # 사용자 응답 없이 Active 모드가 유지되는 최대 시간 (초)

# --- Smart Turn 모델 설정 ---
SMART_TURN_MODEL_PATH = "smart-turn-v3.0.onnx"
TURN_END_SILENCE_CHUNKS = 15        # 무음으로 판단하기 위한 연속 청크 수. 15 chunks * 32ms/chunk ≈ 480ms
MAX_TURN_CHUNKS = 5625              # 사용자 입력 최대 길이. 5625 chunks * 32ms/chunk ≈ 3분 (google stt 최대 길이 한도는 약 5분)
SMART_TURN_GRACE_PERIOD = 0.3       # Smart Turn이 '진행중'으로 판단 시 유예 시간 (초)
SMART_TURN_MAX_RETRIES = 3          # '진행중'일 때 재추론 최대 횟수 (무한 반복 방지)


STT_WAIT_TIMEOUT_SECONDS = 5.0     # STT 결과 대기 최대 시간 (초)

# --- LED 색상 설정 ---
LED_COLOR_OFF = (0, 0, 0)
LED_COLOR_SLEEP = (100, 100, 30)       # Sleep 상태 Ring (따뜻한 흰색)
LED_COLOR_ACTIVE = (233, 233, 50)      # 활성 상태 (노란색)
LED_COLOR_LISTENING = (233, 50, 50)    # Listening 상태 Bar (빨간색)

# --- WebSocket 설정 ---
WS_HOST = "127.0.0.1"
WS_PORT = 5000

# --- OpenAI 클라이언트 팩토리 ---
def create_openai_client():
    """동기 OpenAI 클라이언트를 생성합니다."""
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

def create_async_openai_client():
    """비동기 OpenAI 클라이언트를 생성합니다."""
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=OPENAI_API_KEY)