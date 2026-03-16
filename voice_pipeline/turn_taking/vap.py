"""VAP (Voice Activity Projection) model wrapper.

Wraps the external VoiceActivityProjection model behind the IVAP interface.
Maintains a rolling stereo audio buffer (1, 2, n_samples) at 16kHz,
triggers inference every ``step_sec``, and returns ``VAPResult``.
"""

from __future__ import annotations

import logging
import struct
import time

import torch
import torchaudio.functional

from voice_pipeline.core.config import AudioConfig, TTSConfig, VAPConfig
from voice_pipeline.core.interfaces import IVAP
from voice_pipeline.core.types import AudioFrame, VAPResult
from voice_pipeline.turn_taking.exceptions import VAPError

logger = logging.getLogger("voice_pipeline.turn_taking.vap")

_DEFAULT_RESULT = VAPResult(0.0, 0.0, False)

# VAP model internal frame rate (frames per second).
_VAP_FRAME_HZ = 50


class VAPWrapper(IVAP):
    """IVAP implementation wrapping the external VAP model.

    The wrapper keeps a rolling stereo buffer on CPU and copies it to the
    configured device only at inference time to avoid persistent GPU memory
    usage between frames.
    """

    def __init__(
        self,
        vap_config: VAPConfig,
        audio_config: AudioConfig,
        tts_config: TTSConfig,
    ) -> None:
        self._config = vap_config
        self._audio_config = audio_config
        self._robot_sample_rate = tts_config.output_sample_rate
        self._device = vap_config.device

        # Timing calculations (validated: zero values cause incorrect slicing)
        self._n_samples = round(vap_config.context_sec * audio_config.sample_rate)
        self._step_samples = round(vap_config.step_sec * audio_config.sample_rate)
        self._tt_frames = round(vap_config.tt_time * _VAP_FRAME_HZ)
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
            sd = torch.load(vap_config.model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
            self._model = model.to(self._device).eval()
        except Exception as exc:
            raise VAPError(f"Failed to load VAP model: {exc}") from exc

        # Rolling stereo buffer: (1, 2, n_samples) on CPU
        self._buffer = torch.zeros((1, 2, self._n_samples))
        self._samples_since_inference = 0
        self._cached_result = _DEFAULT_RESULT

    def feed_audio(
        self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None
    ) -> VAPResult:
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
            return _DEFAULT_RESULT

    def reset(self) -> None:
        """Clear the rolling buffer and internal state for a new turn."""
        self._buffer.zero_()
        self._samples_since_inference = 0
        self._cached_result = _DEFAULT_RESULT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_inference(self) -> VAPResult:
        """Run VAP model inference on the current buffer."""
        try:
            t0 = time.monotonic()
            out = self._model.probs(self._buffer.to(self._device))
            p_now = out["p_now"][0, -self._tt_frames :, 0].mean().item()
            p_fut = out["p_future"][0, -self._tt_frames :, 0].mean().item()
            user_is_speaking = out["vad"][0, -1, 0].item() > self._config.vad_threshold
            elapsed_ms = (time.monotonic() - t0) * 1000
            if elapsed_ms > self._config.step_sec * 1000:
                logger.warning("VAP inference slow: %.0fms (step %.0fms)", elapsed_ms, self._config.step_sec * 1000)
            else:
                logger.debug("VAP inference: %.0fms", elapsed_ms)
            return VAPResult(p_now, p_fut, user_is_speaking)
        except Exception:
            logger.warning("VAP inference error, returning default result", exc_info=True)
            return _DEFAULT_RESULT

    def _pcm_to_tensor(self, pcm: bytes) -> torch.Tensor:
        """Convert 16-bit PCM bytes to a float32 tensor normalized to [-1, 1]."""
        n_samples = len(pcm) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm)
        return torch.tensor(samples, dtype=torch.float32) / 32768.0

    def _decode_and_resample_robot(self, robot_audio: bytes, target_length: int) -> torch.Tensor:
        """Decode robot PCM and resample from TTS rate to pipeline rate."""
        robot_tensor = self._pcm_to_tensor(robot_audio)
        if self._robot_sample_rate != self._audio_config.sample_rate:
            robot_tensor = torchaudio.functional.resample(
                robot_tensor.unsqueeze(0),
                orig_freq=self._robot_sample_rate,
                new_freq=self._audio_config.sample_rate,
            ).squeeze(0)
        # Pad or trim to match user audio length
        if robot_tensor.shape[0] < target_length:
            robot_tensor = torch.nn.functional.pad(
                robot_tensor, (0, target_length - robot_tensor.shape[0])
            )
        elif robot_tensor.shape[0] > target_length:
            robot_tensor = robot_tensor[:target_length]
        return robot_tensor
