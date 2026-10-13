"""AudioInput: microphone capture on a daemon thread.

reSpeaker 는 6채널로 열어 한 채널만 mono 로 뽑는다. 기본은 CH0(후처리 출력), 노래 재생 중에는
:meth:`AudioInput.set_raw_capture` 로 원본 채널(+게인)로 바꾼다. ``dump_dir`` 를 주면 뽑기 전 전 채널을 채널별
WAV 로 계속 남긴다 — 후처리 출력과 원본 마이크를 한 시간축에서 비교하는 AEC 진단용.
"""

from __future__ import annotations

import array
import logging
import queue
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from voice_pipeline.settings import (
    CHANNELS,
    FRAME_SIZE_SAMPLES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from voice_pipeline.types import AudioFrame

logger = logging.getLogger("voice_pipeline.audio")


class AudioInput:
    """Captures audio from a microphone via PyAudio on a daemon thread.

    Frames are pushed to the provided queue. If the queue is full,
    frames are dropped (never blocks the capture thread).
    """

    _THREAD_JOIN_TIMEOUT_SEC = 2.0  # 캡처 스레드 종료 대기 (초)
    _DEVICE_INDEX: int | None = None  # PyAudio 입력 디바이스 인덱스 명시 오버라이드. None이면 _DEVICE_NAME으로 탐색
    _DEVICE_NAME: str | None = "respeaker"  # 장치 이름 부분 문자열 매칭 (대소문자 무시). 인덱스와 달리 재열거에 안정적
    _CAPTURE_CHANNELS: int | None = 6  # 디바이스에서 캡처할 채널 수. None은 mono (ReSpeaker 6ch는 6)
    _EXTRACT_CHANNEL = 0  # 다중 채널 캡처 시 mono 추출에 사용할 채널 인덱스 (0-based). CH0 = 처리 출력 (Conference)
    _RAW_CHANNEL = 2  # set_raw_capture 가 쓰는 원본 마이크 채널 (CH2~5 가 게인 없는 원본, CH1 은 ASR 용 처리 출력)
    _RAW_GAIN_DB = 15.0  # 원본 채널 소프트웨어 게인 (CH0 와 레벨을 맞추는 값)

    def __init__(self, audio_queue: queue.Queue[AudioFrame], dump_dir: str | None = None) -> None:
        """Initialize capture state and ensure PyAudio is importable.

        Args:
            audio_queue: 캡처된 오디오 프레임을 push할 공유 큐.
            dump_dir: 있으면 장치에서 읽은 프레임을 게인 없이 채널별로 이 폴더의
                ``<스트림 연 시각>/ch<N>.wav`` 에 계속 쓴다. None 이면 쓰지 않는다.
        """
        self._queue = audio_queue
        self._dump_dir = dump_dir
        self._extract: tuple[int, float] = (self._EXTRACT_CHANNEL, 0.0)  # (채널, 게인 dB). 캡처 스레드가 매 프레임 읽음

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

        # Lazy import PyAudio
        try:
            import pyaudio  # noqa: F401

            self._pyaudio_module = pyaudio
        except ImportError as exc:
            raise RuntimeError("PyAudio is not installed. Install it with: pip install pyaudio") from exc

    def start(self) -> None:
        """Start capturing audio. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing audio and release resources."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._THREAD_JOIN_TIMEOUT_SEC)
            self._thread = None

    @property
    def error(self) -> Exception | None:
        """Return the captured error if the capture thread has died."""
        return self._error

    def set_raw_capture(self, enabled: bool) -> None:
        """Switch between the processed channel (default) and the raw microphone channel with gain.

        다음 프레임부터 적용된다. 단일 채널 캡처에서는 무시된다.

        Args:
            enabled: True 면 ``_RAW_CHANNEL`` 에 ``_RAW_GAIN_DB`` 를 적용해 내보낸다. False 면 기본 채널.
        """
        capture_ch = self._CAPTURE_CHANNELS or CHANNELS
        if capture_ch == CHANNELS:
            return
        self._extract = (self._RAW_CHANNEL, self._RAW_GAIN_DB) if enabled else (self._EXTRACT_CHANNEL, 0.0)
        logger.info("Capture channel → CH%d (gain %+.0f dB)", *self._extract)

    def _resolve_device_index(self, pa: Any) -> int | None:
        """입력 장치 인덱스 결정: 명시 인덱스 > 이름 매칭 > 시스템 기본(None).

        Args:
            pa: 열려 있는 PyAudio 인스턴스.

        Returns:
            ``pa.open()``에 전달할 장치 인덱스. None이면 시스템 기본 장치.

        Raises:
            RuntimeError: ``_DEVICE_NAME``과 일치하는 입력 장치가 없는 경우.
        """
        if self._DEVICE_INDEX is not None:
            return self._DEVICE_INDEX
        if self._DEVICE_NAME is None:
            return None

        needle = self._DEVICE_NAME.lower()
        candidates: list[str] = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) <= 0:
                continue
            name = str(info.get("name", ""))
            if needle in name.lower():
                logger.info("Input device matched '%s': [%d] %s", self._DEVICE_NAME, i, name)
                return i
            candidates.append(f"[{i}] {name}")
        raise RuntimeError(
            f"No input device matching '{self._DEVICE_NAME}'. Available input devices: "
            f"{', '.join(candidates) if candidates else '(none)'}"
        )

    def _capture_loop(self) -> None:
        """Thread target: open stream, read frames, push to queue."""
        pa = None
        stream = None
        dump: list[wave.Wave_write] | None = None
        try:
            pa = self._pyaudio_module.PyAudio()

            device_index = self._resolve_device_index(pa)
            capture_ch = self._CAPTURE_CHANNELS or CHANNELS
            need_extract = capture_ch != CHANNELS

            stream = pa.open(
                format=pa.get_format_from_width(SAMPLE_WIDTH),
                channels=capture_ch,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=FRAME_SIZE_SAMPLES,
            )

            if need_extract:
                for ch_idx in (self._EXTRACT_CHANNEL, self._RAW_CHANNEL):
                    if ch_idx >= capture_ch:
                        raise RuntimeError(f"extract_channel ({ch_idx}) >= capture_channels ({capture_ch})")
                logger.info(
                    "Capturing %dch, extracting CH%d as mono",
                    capture_ch,
                    self._extract[0],
                )
            if self._dump_dir is not None:
                dump = self._open_dump(self._dump_dir, capture_ch)

            while not self._stop_event.is_set():
                data = stream.read(FRAME_SIZE_SAMPLES, exception_on_overflow=False)
                if dump is not None:
                    try:
                        channels = np.frombuffer(data, dtype=np.int16).reshape(-1, capture_ch)
                        for ch, wf in enumerate(dump):
                            wf.writeframes(channels[:, ch].tobytes())  # 매번 헤더를 고쳐 써서 프로세스가 죽어도 유효
                    except Exception:
                        logger.warning("Mic dump write failed — dump stopped", exc_info=True)
                        self._close_dump(dump)
                        dump = None
                if need_extract:
                    ch_idx, gain_db = self._extract
                    samples = array.array("h", data)
                    mono = samples[ch_idx::capture_ch]
                    if gain_db:
                        scaled = np.frombuffer(mono.tobytes(), dtype=np.int16).astype(np.float32) * 10 ** (gain_db / 20)
                        data = np.clip(scaled, -32768, 32767).astype(np.int16).tobytes()
                    else:
                        data = mono.tobytes()
                try:
                    self._queue.put_nowait(data)
                except queue.Full:
                    logger.warning("Audio queue full — dropping frame")

        except Exception as exc:
            logger.error("AudioInput capture error: %s", exc, exc_info=True)
            self._error = exc
        finally:
            if dump is not None:
                self._close_dump(dump)
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    logger.debug("Error closing audio stream", exc_info=True)
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    logger.debug("Error terminating PyAudio", exc_info=True)

    @classmethod
    def _open_dump(cls, dump_dir: str, channels: int) -> list[wave.Wave_write] | None:
        """``dump_dir/<시각>/`` 에 채널별 mono WAV 를 연다. 실패하면 경고만 남기고 None — 캡처는 계속된다."""
        folder = Path(dump_dir) / f"{datetime.now():%Y%m%d_%H%M%S}"
        dump: list[wave.Wave_write] = []
        try:
            folder.mkdir(parents=True, exist_ok=True)
            for ch in range(channels):
                wf = wave.open(str(folder / f"ch{ch}.wav"), "wb")
                dump.append(wf)
                wf.setnchannels(1)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
        except Exception:
            logger.warning("Mic dump disabled: cannot open %s", folder, exc_info=True)
            cls._close_dump(dump)
            return None
        logger.info("Mic dump → %s/ch0..%d.wav (sample 0 ≈ this log time)", folder, channels - 1)
        return dump

    @staticmethod
    def _close_dump(dump: list[wave.Wave_write]) -> None:
        for wf in dump:
            try:
                wf.close()
            except Exception:
                logger.debug("Error closing mic dump", exc_info=True)
