"""MaAI VAP wrapper with ONNX encoder and transformer.

Loads pre-exported ONNX files for both encoder and transformer by default.
If ``transformer_onnx_path`` is empty, falls back to PyTorch transformer
via MaAI (requires ``maai`` package).

Optimal RPi 5 config: ort_threads=1 (single-threaded).

External dependency: ``maai`` package (cloned at ``external/MaAI/``),
only required when using PyTorch transformer fallback.
"""

from __future__ import annotations

import logging
import os
import struct

import numpy as np
import onnxruntime as ort

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from voice_pipeline.core.config import AudioConfig, MaAIVAPConfig, TTSConfig
from voice_pipeline.core.interfaces import IVAP
from voice_pipeline.core.types import AudioFrame, VAPResult
from voice_pipeline.turn_taking.exceptions import VAPError

logger = logging.getLogger("voice_pipeline.turn_taking.maai_vap")

_DEFAULT_RESULT = VAPResult(0.0, 0.0, False)

# MaAI model architecture constants (fixed for all lang/frame_rate variants)
_MAAI_DIM = 256
_MAAI_NUM_HEADS = 4
_MAAI_HEAD_DIM = _MAAI_DIM // _MAAI_NUM_HEADS
_MAAI_CH_LAYERS = 1
_MAAI_CROSS_LAYERS = 3


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
        self._use_onnx_transformer = bool(config.transformer_onnx_path)
        self._use_torch_compile = (
            not self._use_onnx_transformer and config.use_torch_compile
        )

        # ORT session options (shared by encoder and transformer)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = config.ort_threads

        # Encoder (ONNX from file)
        if not config.encoder_onnx_path:
            raise VAPError("encoder_onnx_path is required")
        if not os.path.isfile(config.encoder_onnx_path):
            raise VAPError(f"Encoder ONNX file not found: {config.encoder_onnx_path}")
        try:
            self._sess1 = ort.InferenceSession(config.encoder_onnx_path, sess_opts)
            self._sess2 = ort.InferenceSession(config.encoder_onnx_path, sess_opts)
        except Exception as exc:
            raise VAPError(f"Failed to load ONNX encoder: {exc}") from exc

        # Transformer
        if self._use_onnx_transformer:
            if not os.path.isfile(config.transformer_onnx_path):
                raise VAPError(
                    f"Transformer ONNX file not found: {config.transformer_onnx_path}"
                )
            try:
                self._tfm_sess = ort.InferenceSession(config.transformer_onnx_path, sess_opts)
                logger.info("ONNX transformer loaded from %s", config.transformer_onnx_path)
            except Exception as exc:
                raise VAPError(f"Failed to load ONNX transformer: {exc}") from exc

            self._n_ch_layers = _MAAI_CH_LAYERS
            self._n_cross_layers = _MAAI_CROSS_LAYERS
            self._nh = _MAAI_NUM_HEADS
            self._hd = _MAAI_HEAD_DIM
        else:
            # PyTorch transformer requires torch
            if torch is None:
                raise VAPError(
                    "torch is required when transformer_onnx_path is not set"
                )
            torch.set_num_threads(config.pt_threads)

            # Load MaAI for PyTorch transformer
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
            if config.use_torch_compile:
                try:
                    self._vap_forward = torch.compile(self._vap.forward, mode="reduce-overhead")
                    logger.info("torch.compile enabled for transformer")
                except Exception:
                    logger.warning(
                        "torch.compile failed, falling back to eager mode",
                        exc_info=True,
                    )
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

        # Warmup: pre-allocate ORT buffers / trigger torch.compile
        self._warmup()
        self.reset()

        mode = "onnx" if self._use_onnx_transformer else "pytorch"
        logger.info(
            "MaAIVAPWrapper initialized: frame_rate=%d, context=%.1fs, "
            "ort_threads=%d, pt_threads=%d, transformer=%s",
            config.frame_rate,
            config.context_len_sec,
            config.ort_threads,
            config.pt_threads,
            mode,
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

        if self._use_torch_compile:
            # torch.compile: zero values but keep tensor shapes
            # to avoid recompilation on shape change
            self._zero_pytorch_cache()
        else:
            # ONNX and PyTorch eager: safe to clear entirely
            self._vap_cache = None

        self._cached_result = _DEFAULT_RESULT

    def _zero_pytorch_cache(self) -> None:
        """Zero PyTorch KV cache values, preserving tensor shapes.

        Replaces inference-mode tensors with normal zero tensors of the
        same shape, since inference tensors cannot be modified in-place.
        """
        if self._vap_cache is None:
            return
        for key, (k_list, v_list) in self._vap_cache.items():
            self._vap_cache[key] = (
                [torch.zeros_like(t) if isinstance(t, torch.Tensor) else t for t in k_list],
                [torch.zeros_like(t) if isinstance(t, torch.Tensor) else t for t in v_list],
            )

    def _warmup(self) -> None:
        """Run dummy inference to pre-allocate ORT buffers / trigger torch.compile."""
        if self._use_onnx_transformer:
            n_frames = 2
        else:
            n_frames = self._audio_context_len

        logger.info(
            "Warmup: %d frames (transformer=%s)...",
            n_frames,
            "onnx" if self._use_onnx_transformer else "pytorch",
        )

        # Warmup encoder ORT sessions
        dummy_wav = np.zeros((1, 1, self._audio_frame_size), dtype=np.float32)
        dummy_h = np.zeros((1, 1, 256), dtype=np.float32)
        dummy_c = np.zeros((1, 1, 256), dtype=np.float32)
        for sess in (self._sess1, self._sess2):
            sess.run(None, {"waveform": dummy_wav, "h_in": dummy_h, "c_in": dummy_c})

        # Warmup transformer
        dummy_e = np.zeros((1, 1, _MAAI_DIM), dtype=np.float32)
        for _ in range(n_frames):
            if self._use_onnx_transformer:
                self._process_transformer_onnx(dummy_e, dummy_e)
            else:
                self._process_transformer_pytorch(dummy_e, dummy_e)

        logger.info("Warmup complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_frame(self, x1: np.ndarray, x2: np.ndarray) -> VAPResult | None:
        """Run one frame through ONNX encoder + transformer."""
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
        if self._use_onnx_transformer:
            out = self._process_transformer_onnx(e1_np, e2_np)
        else:
            out = self._process_transformer_pytorch(e1_np, e2_np)

        # Buffer trimming
        self._buf_x1 = self._buf_x1[-self._frame_contxt_padding :].copy()
        self._buf_x2 = self._buf_x2[-self._frame_contxt_padding :].copy()

        # Convert to VAPResult
        p_now = float(out["p_now"])
        p_fut = float(out["p_future"])
        user_is_speaking = float(out["vad"][0]) > self._config.vad_threshold

        return VAPResult(p_now, p_fut, user_is_speaking)

    def _process_transformer_onnx(self, e1_np: np.ndarray, e2_np: np.ndarray) -> dict:
        """Run transformer via ONNX Runtime."""
        limit = self._audio_context_len - 1

        # Build cache inputs
        if self._vap_cache is None:
            empty_ch = np.zeros((self._n_ch_layers, 1, self._nh, 0, self._hd), dtype=np.float32)
            empty_cr = np.zeros((self._n_cross_layers, 1, self._nh, 0, self._hd), dtype=np.float32)
            cache = {
                "ar1_k": empty_ch,
                "ar1_v": empty_ch.copy(),
                "ar2_k": empty_ch.copy(),
                "ar2_v": empty_ch.copy(),
                "cross1_k": empty_cr,
                "cross1_v": empty_cr.copy(),
                "cross2_k": empty_cr.copy(),
                "cross2_v": empty_cr.copy(),
                "cross1_c_k": empty_cr.copy(),
                "cross1_c_v": empty_cr.copy(),
                "cross2_c_k": empty_cr.copy(),
                "cross2_c_v": empty_cr.copy(),
            }
        else:
            cache = self._vap_cache

        # Run
        inputs = {"x1": e1_np, "x2": e2_np, **cache}
        outputs = self._tfm_sess.run(None, inputs)
        out_names = [o.name for o in self._tfm_sess.get_outputs()]
        result = dict(zip(out_names, outputs, strict=False))

        # Update cache with trimming
        new_cache = {
            "ar1_k": result["out_ar1_k"],
            "ar1_v": result["out_ar1_v"],
            "ar2_k": result["out_ar2_k"],
            "ar2_v": result["out_ar2_v"],
            "cross1_k": result["out_cross1_k"],
            "cross1_v": result["out_cross1_v"],
            "cross2_k": result["out_cross2_k"],
            "cross2_v": result["out_cross2_v"],
            "cross1_c_k": result["out_cross1_c_k"],
            "cross1_c_v": result["out_cross1_c_v"],
            "cross2_c_k": result["out_cross2_c_k"],
            "cross2_c_v": result["out_cross2_c_v"],
        }
        for name, arr in new_cache.items():
            if arr.ndim >= 4 and arr.shape[3] > limit:
                new_cache[name] = arr[:, :, :, -limit:, :]
        self._vap_cache = new_cache

        return {
            "p_now": result["p_now"][0],
            "p_future": result["p_future"][0],
            "vad": [float(result["vad1"]), float(result["vad2"])],
        }

    def _process_transformer_pytorch(self, e1_np: np.ndarray, e2_np: np.ndarray) -> dict:
        """Run transformer via PyTorch (fallback path)."""
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

        # Normalize to same format as ONNX path:
        # VapGPT.forward() returns p_now/p_future as [speaker1, speaker2] lists.
        p_now = out["p_now"]
        p_fut = out["p_future"]
        return {
            "p_now": p_now[0] if isinstance(p_now, list) else p_now,
            "p_future": p_fut[0] if isinstance(p_fut, list) else p_fut,
            "vad": out["vad"],
        }

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
