"""노래나 TTS 를 ReSpeaker 로 틀면서 마이크 6채널을 녹음한다 — AEC·더블토크 확인용. Ctrl+C 로 언제 끝내도 저장된다.

    uv run python scripts/hardware/aec_play_record.py [곡키 | tts] [--seconds 60] [--lead 6]

- 곡키: assets/audio/music/<곡키>.wav 를 그대로 재생 (음원 자체가 정규화돼 있어 C++ 재생과 같은 레벨).
  결과는 var/probe/song/<시각>/.
- tts: 일기예보 TTS(GPT-Live 목소리, 대화 말소리 레벨로 정규화)를 재생. 결과는 var/probe/aec/<시각>/.
  TTS 파일이 없으면 처음 한 번 합성한다 (OPENAI_API_KEY 필요).
  재생 전 --lead 초(기본 6) 동안 혼자 한 문장 말하기 — 비교 기준.
들을 때용 사본: ch1_loud.wav(+25 dB)·ch2_loud.wav(+45 dB). 파이프라인이 떠 있으면 마이크를 못 여니 먼저 내릴 것.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

from voice_pipeline.adapters.audio_input import AudioInput
from voice_pipeline.adapters.tts_openai import OpenAITTS
from voice_pipeline.audio_level import ensure_normalized
from voice_pipeline.engines.gpt_live.instructions import LIVE_GREETING_TTS_MODEL, LIVE_VOICE
from voice_pipeline.greeting_audio import synthesize_to_wav

MUSIC_DIR = Path("assets/audio/music")
TTS_SRC = Path("var/probe/aec/tts_weather.wav")  # 합성 원본. 재생은 말소리 레벨로 맞춘 사본
TTS_TEXT = (
    "오늘 서울은 대체로 맑겠고, 낮 최고기온은 이십삼 도로 어제보다 조금 높겠습니다. "
    "오후부터 서쪽 지역에 구름이 많아지겠고, 밤에는 경기 북부에 약한 비가 오는 곳이 있겠습니다. "
    "바다의 물결은 서해와 남해 앞바다에서 영 점 오에서 일 미터로 일겠습니다. "
    "내일 아침 기온은 십이 도 안팎으로 쌀쌀하겠으니 건강 관리에 유의하시기 바랍니다."
)
LOUD_DB = {1: 25, 2: 45}


def tts_file() -> Path:
    if not TTS_SRC.exists():
        print(f"TTS 합성 중 ({LIVE_VOICE}, {LIVE_GREETING_TTS_MODEL}) …")
        synthesize_to_wav(OpenAITTS(voice=LIVE_VOICE, model=LIVE_GREETING_TTS_MODEL), TTS_TEXT, TTS_SRC)
    return ensure_normalized(TTS_SRC, speech=True)


def drain(q: queue.Queue, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            q.get(timeout=0.1)
        except queue.Empty:
            pass


def make_loud(rec: Path) -> None:
    for ch, gain_db in LOUD_DB.items():
        with wave.open(str(rec / f"ch{ch}.wav")) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
        y = np.clip(x * 10 ** (gain_db / 20), -32768, 32767).astype(np.int16)
        with wave.open(str(rec / f"ch{ch}_loud.wav"), "wb") as o:
            o.setnchannels(1)
            o.setsampwidth(2)
            o.setframerate(sr)
            o.writeframes(y.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="Dynamite_BTS", help="assets/audio/music/ 의 곡 이름 또는 tts")
    ap.add_argument("--seconds", type=float, default=None, help="이만큼만 재생 (기본: 끝까지 또는 Ctrl+C)")
    ap.add_argument("--lead", type=float, default=None, help="재생 전 대기(초). 기본: tts 6, 노래 1")
    args = ap.parse_args()

    if args.what == "tts":
        src, out_dir, lead = tts_file(), Path("var/probe/aec"), 6.0
    else:
        src, out_dir, lead = MUSIC_DIR / f"{args.what}.wav", Path("var/probe/song"), 1.0
    if args.lead is not None:
        lead = args.lead
    if not src.exists():
        names = ", ".join(sorted(p.stem for p in MUSIC_DIR.glob("*.wav")))
        raise SystemExit(f"없는 파일: {src}\n곡: {names}, 또는 tts")

    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.iterdir())
    q: queue.Queue = queue.Queue(maxsize=300)
    stop = threading.Event()
    threading.Thread(target=drain, args=(q, stop), daemon=True).start()
    mic = AudioInput(q, dump_dir=str(out_dir))
    mic.start()
    player = None
    try:
        time.sleep(0.5)
        if mic.error:
            raise SystemExit(f"마이크 열기 실패 (파이프라인이 떠 있나?): {mic.error}")
        if lead > 1.0:
            print(f"\n[녹음 중] {lead:.0f}초 동안 혼자 한 문장 말해 주세요 (기준).")
        time.sleep(max(0.0, lead - 0.5))
        print(f"[재생 중] {src.name} — 위로 말해 보세요. Ctrl+C 로 언제든 끝낼 수 있습니다.")
        player = subprocess.Popen(["pw-play", str(src)])
        player.wait(timeout=args.seconds)
        time.sleep(1.0)
    except subprocess.TimeoutExpired:
        pass
    except KeyboardInterrupt:
        print("\n중지")
    finally:
        if player is not None and player.poll() is None:
            player.terminate()
            player.wait()
        mic.stop()
        stop.set()

    (rec,) = set(out_dir.iterdir()) - before
    make_loud(rec)
    print(f"녹음: {rec}/ch0~5.wav  (들을 때: ch0.wav, ch1_loud.wav +25 dB, ch2_loud.wav +45 dB)")


if __name__ == "__main__":
    main()
