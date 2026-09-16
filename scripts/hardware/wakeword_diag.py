"""웨이크워드 진단: 마이크 레벨, VAD 확률, 상태 전이, STT 결과를 초 단위로 찍는다.

    uv run python scripts/hardware/wakeword_diag.py [--seconds 60]

"말했는데 안 잡힌다"가 (a) 마이크 레벨, (b) VAD 미검출, (c) STT 빈 결과 중 어디서 막히는지 가른다.
"""

from __future__ import annotations

import argparse
import array
import logging
import math
import queue
import sys
import time

from voice_pipeline.adapters.audio_input import AudioInput
from voice_pipeline.adapters.wakeword import WakewordDetector
from voice_pipeline.types import AudioFrame

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)
for n in ("google", "urllib3", "voice_pipeline.audio"):
    logging.getLogger(n).setLevel(logging.WARNING)
logging.getLogger("voice_pipeline.wakeword").setLevel(logging.DEBUG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    det = WakewordDetector()
    cfg = det._recognition_config
    print(f"STT: primary={cfg.language_code} alt={list(cfg.alternative_language_codes)} keywords={det._KEYWORDS}")

    # VAD 확률과 상태 전이를 가로채서 기록
    probs: list[float] = []
    orig = det._process_vad

    def spy(prob: float, chunk: bytes) -> None:
        probs.append(prob)
        before = det._state
        orig(prob, chunk)
        if det._state is not before:
            print(f"      VAD {before.name} -> {det._state.name} (p={prob:.2f})")

    det._process_vad = spy  # type: ignore[method-assign]

    q: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
    mic = AudioInput(q)
    mic.start()
    print(f"듣는 중 {args.seconds:.0f}초 — '레이', 'Ray', 'hello' 등을 말해 보세요\n")
    t0 = time.monotonic()
    win_frames: list[bytes] = []
    last_print = t0
    try:
        while time.monotonic() - t0 < args.seconds:
            try:
                frame = q.get(timeout=0.1)
            except queue.Empty:
                continue
            win_frames.append(frame)
            if det.feed_audio(frame):
                print(">>> WAKEWORD DETECTED <<<")
            now = time.monotonic()
            if now - last_print >= 1.0:
                pcm = array.array("h", b"".join(win_frames))
                rms = int(math.sqrt(sum(v * v for v in pcm) / max(len(pcm), 1))) if pcm else 0
                pmax = max(probs) if probs else 0.0
                peak = max((abs(v) for v in pcm), default=0)
                state = det._state.name
                print(f"[{now - t0:5.1f}s] mic rms={rms:5d} peak={peak:5d}  VAD max p={pmax:.2f}  state={state}")
                win_frames.clear()
                probs.clear()
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        det.close()


if __name__ == "__main__":
    main()
