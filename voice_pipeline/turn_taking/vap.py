"""VAP (Voice Activity Projection) model wrapper.

Wraps the external VoiceActivityProjection model behind the IVAP interface.
Maintains a rolling stereo audio buffer (1, 2, n_samples) at 16kHz,
triggers inference every ``step_sec``, and returns ``VAPResult``.
"""

from __future__ import annotations

import logging
import struct
import time

try:
    import torch
    import torchaudio.functional
except ImportError:
    torch = None  # type: ignore[assignment]
    torchaudio = None  # type: ignore[assignment]

from voice_pipeline.audio.constants import SAMPLE_RATE
from voice_pipeline.core.interfaces import IVAP
from voice_pipeline.core.types import AudioFrame, VAPResult
from voice_pipeline.turn_taking.exceptions import VAPError

logger = logging.getLogger("voice_pipeline.turn_taking.vap")


class VAPWrapper(IVAP):
    """IVAP implementation wrapping the external VAP model.

    The wrapper keeps a rolling stereo buffer on CPU and copies it to the
    configured device only at inference time to avoid persistent GPU memory
    usage between frames.

    Args:
        tts_sample_rate: Robot(TTS) 출력 샘플레이트. 리샘플링 기준.
    """

    _DEFAULT_RESULT = VAPResult(0.0, 0.0, False)  # 추론 실패/초기 상태 반환값
    _MODEL_PATH = "external/VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
    _CONTEXT_SEC = 20.0  # 롤링 버퍼 길이 (초)
    _STEP_SEC = 0.1  # 추론 간격 (초)
    _TT_TIME = 0.5  # turn-taking 평균화 lookahead (초)
    _DEVICE = "cpu"  # PyTorch 디바이스 ("cpu" / "cuda")
    _VAD_THRESHOLD = 0.5  # user_is_speaking 임계값
    _VAP_FRAME_HZ = 50  # VAP 체크포인트 내부 프레임 레이트 (Hz). 현재 체크포인트 50Hz 고정

    def __init__(
        self,
        tts_sample_rate: int,
    ) -> None:
        if torch is None:
            raise VAPError(
                "torch and torchaudio are required for VAPWrapper. Install with: uv sync --extra models-pytorch"
            )
        self._robot_sample_rate = tts_sample_rate

        # Timing calculations (validated: zero values cause incorrect slicing)
        self._n_samples = round(self._CONTEXT_SEC * SAMPLE_RATE)
        self._step_samples = round(self._STEP_SEC * SAMPLE_RATE)
        self._tt_frames = round(self._TT_TIME * self._VAP_FRAME_HZ)
        if self._n_samples < 1 or self._step_samples < 1 or self._tt_frames < 1:
            raise VAPError(
                f"Invalid timing config: n_samples={self._n_samples}, "
                f"step_samples={self._step_samples}, tt_frames={self._tt_frames}. "
                "All must be >= 1."
            )

        # Load model
        try:
            from vap.model import VapConfig, VapGPT

            model = VapGPT(VapConfig())
            sd = torch.load(self._MODEL_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
            self._model = model.to(self._DEVICE).eval()
        except Exception as exc:
            raise VAPError(f"Failed to load VAP model: {exc}") from exc

        # Rolling stereo buffer: (1, 2, n_samples) on CPU
        self._buffer = torch.zeros((1, 2, self._n_samples))
        self._samples_since_inference = 0
        self._cached_result = self._DEFAULT_RESULT

    def feed_audio(self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None) -> VAPResult:
        """Feed one pipeline frame and return voice activity estimates."""
        try:
            user_tensor = self._pcm_to_tensor(user_audio)
            n = user_tensor.shape[0]
            if n == 0:
                return self._cached_result

            # Clamp to context buffer size (drop oldest if frame > buffer)
            if n > self._n_samples:
                user_tensor = user_tensor[-self._n_samples :]
                n = self._n_samples

            if robot_audio is not None:
                robot_tensor = self._decode_and_resample_robot(robot_audio, n)
            else:
                robot_tensor = torch.zeros(n)

            # Roll buffer left and write new samples at tail
            self._buffer = self._buffer.roll(-n, dims=-1)
            self._buffer[0, 0, -n:] = user_tensor
            self._buffer[0, 1, -n:] = robot_tensor

            self._samples_since_inference += n
            if self._samples_since_inference >= self._step_samples:
                self._cached_result = self._run_inference()
                self._samples_since_inference %= self._step_samples

            return self._cached_result
        except Exception:
            logger.warning("Error in feed_audio, returning default result", exc_info=True)
            return self._DEFAULT_RESULT

    def reset(self) -> None:
        """Clear the rolling buffer and internal state for a new turn."""
        self._buffer.zero_()
        self._samples_since_inference = 0
        self._cached_result = self._DEFAULT_RESULT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_inference(self) -> VAPResult:
        """Run VAP model inference on the current buffer."""
        try:
            t0 = time.monotonic()
            out = self._model.probs(self._buffer.to(self._DEVICE))
            p_now = out["p_now"][0, -self._tt_frames :, 0].mean().item()
            p_fut = out["p_future"][0, -self._tt_frames :, 0].mean().item()
            user_is_speaking = out["vad"][0, -1, 0].item() > self._VAD_THRESHOLD
            elapsed_ms = (time.monotonic() - t0) * 1000
            step_ms = self._STEP_SEC * 1000
            if elapsed_ms > step_ms:
                logger.warning("VAP inference slow: %.0fms (step %.0fms)", elapsed_ms, step_ms)
            # else:
            #     logger.debug("VAP inference: %.0fms", elapsed_ms)
            return VAPResult(p_now, p_fut, user_is_speaking)
        except Exception:
            logger.warning("VAP inference error, returning default result", exc_info=True)
            return self._DEFAULT_RESULT

    def _pcm_to_tensor(self, pcm: bytes) -> torch.Tensor:
        """Convert 16-bit PCM bytes to a float32 tensor normalized to [-1, 1]."""
        n_samples = len(pcm) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm)
        return torch.tensor(samples, dtype=torch.float32) / 32768.0

    def _decode_and_resample_robot(self, robot_audio: bytes, target_length: int) -> torch.Tensor:
        """Decode robot PCM and resample from TTS rate to pipeline rate."""
        robot_tensor = self._pcm_to_tensor(robot_audio)
        if self._robot_sample_rate != SAMPLE_RATE:
            robot_tensor = torchaudio.functional.resample(
                robot_tensor.unsqueeze(0),
                orig_freq=self._robot_sample_rate,
                new_freq=SAMPLE_RATE,
            ).squeeze(0)
        # Pad or trim to match user audio length
        if robot_tensor.shape[0] < target_length:
            robot_tensor = torch.nn.functional.pad(robot_tensor, (0, target_length - robot_tensor.shape[0]))
        elif robot_tensor.shape[0] > target_length:
            robot_tensor = robot_tensor[:target_length]
        return robot_tensor
