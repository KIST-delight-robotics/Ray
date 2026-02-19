"""Silero VAD 모델을 사용한 음성 활동 감지."""

import time
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VADProcessor:
    """Silero VAD 모델을 사용하여 음성 활동을 감지하는 클래스."""

    def __init__(self, sample_rate: int, chunk_size: int, threshold: float = 0.5, consecutive_chunks: int = 3, reset_interval: float = 20.0):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.consecutive_chunks_required = consecutive_chunks
        self.reset_interval = reset_interval

        try:
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
            self.vad_model = model
            logger.info("Silero VAD 초기화 완료")
        except Exception as e:
            logger.error(f"Silero VAD 로드 실패: {e}")
            self.vad_model = None

        self.vad_buffer = torch.tensor([])
        self.consecutive_speech_chunks = 0
        self.vad_detection_start_time = time.time()

    def process(self, audio_chunk_int16: np.ndarray) -> bool:
        """
        오디오 청크를 처리하고 음성 감지 여부를 반환합니다.

        Returns:
            bool: 음성 시작 조건(연속 청크 수)이 충족되었는지 여부.
        """
        if self.vad_model is None:
            return False

        audio_chunk_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_chunk_float32.flatten())

        self.vad_buffer = torch.cat([self.vad_buffer, audio_tensor])

        speech_detected = False

        while len(self.vad_buffer) >= self.chunk_size:
            vad_chunk = self.vad_buffer[:self.chunk_size]
            self.vad_buffer = self.vad_buffer[self.chunk_size:]

            speech_prob = self.vad_model(vad_chunk, self.sample_rate).item()

            if speech_prob > self.threshold:
                self.consecutive_speech_chunks += 1
                self.vad_detection_start_time = time.time()
            else:
                self.consecutive_speech_chunks = 0

            if self.consecutive_speech_chunks >= self.consecutive_chunks_required:
                speech_detected = True
                break

        return speech_detected

    def reset_if_inactive(self):
        """일정 시간 동안 음성 감지가 없으면 VAD 상태를 초기화합니다."""
        if time.time() - self.vad_detection_start_time > self.reset_interval:
            logger.info(f"{self.reset_interval}초 동안 음성 감지가 없어 VAD 상태를 초기화합니다.")
            self.reset()
            self.vad_detection_start_time = time.time()

    def reset(self):
        """VAD 상태를 초기화합니다."""
        if self.vad_model:
            self.vad_model.reset_states()
        self.vad_buffer = torch.tensor([])
        self.consecutive_speech_chunks = 0
        logger.info("VAD 상태 초기화 완료.")
