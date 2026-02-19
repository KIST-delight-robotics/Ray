"""마이크 입력 스트림 관리."""

import queue
import logging

import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)


def find_input_device(device_name_substring: str = 'pipewire') -> int | None:
    """주어진 문자열이 포함된 오디오 입력 장치를 검색합니다.
    지정된 이름을 찾지 못하면 시스템 기본 입력 장치를 사용합니다.
    """
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device_name_substring.lower() in device['name'].lower() and device['max_input_channels'] > 0:
            logger.info(f"발견된 입력 장치: [{idx}] {device['name']}")
            return idx

    # 지정된 이름을 찾지 못한 경우: 시스템 기본 입력 장치 사용
    try:
        default_device = sd.default.device[0]  # (input, output) 튜플
        if default_device is not None and default_device >= 0:
            info = sd.query_devices(default_device)
            if info['max_input_channels'] > 0:
                logger.info(f"기본 입력 장치 사용: [{default_device}] {info['name']}")
                return default_device
    except Exception:
        pass

    # 기본 장치도 없으면 첫 번째 입력 가능 장치 탐색
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            logger.info(f"첫 번째 입력 장치 사용: [{idx}] {device['name']}")
            return idx

    logger.warning("사용 가능한 입력 장치를 찾지 못했습니다.")
    return None


class MicrophoneStream:
    """마이크로부터 오디오 데이터를 읽어 큐에 넣는 클래스."""

    def __init__(self, mic_audio_queue: queue.Queue, sample_rate: int, chunk_size: int, channels: int, dtype: str, device_idx: int | None = None):
        self.mic_audio_queue = mic_audio_queue
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = dtype
        self.device_idx = device_idx
        self.stream: sd.InputStream | None = None

    def start(self):
        if self.stream is not None and self.stream.active:
            logger.warning("MicrophoneStream이 이미 활성화되어 있습니다.")
            return

        logger.info("MicrophoneStream 시작.")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=self.channels,
            dtype=self.dtype,
            device=self.device_idx,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            logger.info("MicrophoneStream 중지.")
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning(f"[오디오 상태] {status}")
        self.mic_audio_queue.put(indata.copy())
