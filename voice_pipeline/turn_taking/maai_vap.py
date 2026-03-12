"""MaAI VAP wrapper with ONNX encoder + PyTorch transformer.

Uses ONNX Runtime for the CPC encoder (2 channels) and PyTorch for the
GPT cross-attention transformer.  Optionally applies ``torch.compile``
for ~22% transformer speedup on RPi 5.

Optimal RPi 5 config: pt_threads=1, ort_threads=1 (single-threaded).

External dependency: ``maai`` package (cloned at ``external/MaAI/``).
"""

from __future__ import annotations

import logging
import os
import struct
import numpy as np
import onnxruntime as ort
import torch

from voice_pipeline.core.config import AudioConfig, MaAIVAPConfig, TTSConfig
from voice_pipeline.core.interfaces import IVAP
from voice_pipeline.core.types import AudioFrame, VAPResult
from voice_pipeline.turn_taking.exceptions import VAPError
from voice_pipeline.turn_taking.onnx_export import export_encoder_onnx

logger = logging.getLogger("voice_pipeline.turn_taking.maai_vap")

_DEFAULT_RESULT = VAPResult(0.0, 0.0, False)


# ---------------------------------------------------------------------------
# MaAI VAP wrapper (IVAP implementation)
# ---------------------------------------------------------------------------


class MaAIVAPWrapper(IVAP):
    """IVAP implementation using MaAI VAP with ONNX encoder.

    Internally stateful: maintains LSTM hidden states for the encoder
    and KV-cache for the transformer across ``feed_audio`` calls.
    """

    def __init__(
        self,
        config: MaAIVAPConfig,
        audio_config: AudioConfig,
        tts_config: TTSConfig,
    ) -> None:
        self._config = config
        self._audio_config = audio_config
        self._robot_sample_rate = tts_config.output_sample_rate

        # Set thread counts before loading anything
        torch.set_num_threads(config.pt_threads)

        # Load MaAI (creates the full model, we extract parts from it)
        try:
            from maai import Maai, MaaiInput

            ch1 = MaaiInput.Chunk()
            ch2 = MaaiInput.Chunk()
            self._maai = Maai(
                mode="vap",
                lang=config.lang,
                frame_rate=config.frame_rate,
                context_len_sec=config.context_len_sec,
                audio_ch1=ch1,
                audio_ch2=ch2,
                device="cpu",
                use_kv_cache=True,
            )
        except Exception as exc:
            raise VAPError(f"Failed to load MaAI: {exc}") from exc

        self._vap = self._maai.vap

        # Export and load ONNX encoder sessions
        try:
            onnx_path = export_encoder_onnx(self._maai, config.frame_rate)

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = config.ort_threads

            self._sess1 = ort.InferenceSession(onnx_path, sess_opts)
            self._sess2 = ort.InferenceSession(onnx_path, sess_opts)

            os.unlink(onnx_path)
        except Exception as exc:
            raise VAPError(f"Failed to create ONNX encoder sessions: {exc}") from exc

        # Optional torch.compile
        if config.use_torch_compile:
            try:
                self._vap_forward = torch.compile(
                    self._vap.forward, mode="reduce-overhead"
                )
                logger.info("torch.compile enabled for transformer")
            except Exception:
                logger.warning("torch.compile failed, falling back to eager mode", exc_info=True)
                self._vap_forward = self._vap.forward
        else:
            self._vap_forward = self._vap.forward

        # Audio buffering constants
        self._frame_rate = config.frame_rate
        self._frame_contxt_padding = 320
        self._audio_frame_size = 16000 // config.frame_rate + self._frame_contxt_padding
        self._audio_context_len = int(config.context_len_sec * config.frame_rate)

        # Mutable state
        self._h1 = np.zeros((1, 1, 256), dtype=np.float32)
        self._c1 = np.zeros((1, 1, 256), dtype=np.float32)
        self._h2 = np.zeros((1, 1, 256), dtype=np.float32)
        self._c2 = np.zeros((1, 1, 256), dtype=np.float32)
        self._buf_x1 = np.zeros(self._frame_contxt_padding, dtype=np.float32)
        self._buf_x2 = np.zeros(self._frame_contxt_padding, dtype=np.float32)
        self._vap_cache: dict | None = None
        self._cached_result = _DEFAULT_RESULT

        logger.info(
            "MaAIVAPWrapper initialized: frame_rate=%d, context=%.1fs, "
            "pt_threads=%d, ort_threads=%d, torch_compile=%s",
            config.frame_rate, config.context_len_sec,
            config.pt_threads, config.ort_threads, config.use_torch_compile,
        )

    def feed_audio(
        self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None
    ) -> VAPResult:
        """Feed one pipeline frame and return voice activity estimates."""
        try:
            x1 = self._pcm_to_numpy(user_audio)
            if robot_audio is not None:
                x2 = self._pcm_to_numpy(robot_audio)
                x2 = self._resample_robot(x2, len(x1))
            else:
                x2 = np.zeros(len(x1), dtype=np.float32)

            result = self._process_frame(x1, x2)
            if result is not None:
                self._cached_result = result

            return self._cached_result
        except Exception:
            logger.warning("Error in feed_audio, returning cached result", exc_info=True)
            return self._cached_result

    def reset(self) -> None:
        """Clear encoder state, KV cache, and audio buffers."""
        self._h1[:] = 0
        self._c1[:] = 0
        self._h2[:] = 0
        self._c2[:] = 0
        self._buf_x1 = np.zeros(self._frame_contxt_padding, dtype=np.float32)
        self._buf_x2 = np.zeros(self._frame_contxt_padding, dtype=np.float32)
        self._vap_cache = None
        self._cached_result = _DEFAULT_RESULT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_frame(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> VAPResult | None:
        """Run one frame through ONNX encoder + PyTorch transformer."""
        # Audio buffering
        self._buf_x1 = np.concatenate([self._buf_x1, x1])
        self._buf_x2 = np.concatenate([self._buf_x2, x2])

        if len(self._buf_x1) < self._audio_frame_size:
            return None

        # ONNX encoder (both channels)
        wav1 = self._buf_x1.reshape(1, 1, -1)
        wav2 = self._buf_x2.reshape(1, 1, -1)

        e1_np, self._h1, self._c1 = self._sess1.run(
            None, {"waveform": wav1, "h_in": self._h1, "c_in": self._c1}
        )
        e2_np, self._h2, self._c2 = self._sess2.run(
            None, {"waveform": wav2, "h_in": self._h2, "c_in": self._c2}
        )

        # Transformer forward
        e1 = torch.from_numpy(e1_np)
        e2 = torch.from_numpy(e2_np)

        with torch.inference_mode():
            out, self._vap_cache = self._vap_forward(e1, e2, cache=self._vap_cache)

        # Cache trimming
        if self._vap_cache is not None:
            limit = self._audio_context_len - 1
            new_cache: dict = {}
            for key, (k_list, v_list) in self._vap_cache.items():
                new_cache[key] = (
                    [
                        t[..., -limit:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t
                        for t in k_list
                    ],
                    [
                        t[..., -limit:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t
                        for t in v_list
                    ],
                )
            self._vap_cache = new_cache

        # Buffer trimming
        self._buf_x1 = self._buf_x1[-self._frame_contxt_padding:].copy()
        self._buf_x2 = self._buf_x2[-self._frame_contxt_padding:].copy()

        # Convert MaAI output to VAPResult
        p_now = float(out["p_now"])
        p_fut = float(out["p_future"])
        user_is_speaking = float(out["vad"][0]) > self._config.vad_threshold

        return VAPResult(p_now, p_fut, user_is_speaking)

    def _pcm_to_numpy(self, pcm: bytes) -> np.ndarray:
        """Convert 16-bit PCM bytes to float32 numpy array normalized to [-1, 1]."""
        n_samples = len(pcm) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm)
        return np.array(samples, dtype=np.float32) / 32768.0

    def _resample_robot(self, robot: np.ndarray, target_length: int) -> np.ndarray:
        """Resample robot audio from TTS rate to pipeline rate and match length."""
        if self._robot_sample_rate != self._audio_config.sample_rate:
            ratio = self._audio_config.sample_rate / self._robot_sample_rate
            n_out = int(len(robot) * ratio)
            indices = np.linspace(0, len(robot) - 1, n_out).astype(np.float64)
            lo = indices.astype(np.int64)
            hi = np.minimum(lo + 1, len(robot) - 1)
            frac = (indices - lo).astype(np.float32)
            robot = robot[lo] * (1 - frac) + robot[hi] * frac

        # Pad or trim
        if len(robot) < target_length:
            robot = np.pad(robot, (0, target_length - len(robot)))
        elif len(robot) > target_length:
            robot = robot[:target_length]
        return robot
