"""MaAI VAP inference model (ONNX encoder + transformer).

Loads pre-exported ONNX files for both encoder and transformer by default.
Toggle ``_USE_ONNX_TRANSFORMER = False`` (class var) for PyTorch fallback
via MaAI (requires ``maai`` package).

Optimal RPi 5 config: ort_threads=1 (single-threaded).

Pure synchronous inference: ``infer(user, robot)`` returns a ``VAPResult``.
Background scheduling (buffering, the inference thread, the pipeline-facing
``IVAP`` runtime) lives in ``ThreadedVAP`` (``threaded_vap.py``), which holds
an instance of this model. This class is not thread-safe; ``ThreadedVAP``
serializes access.

External dependency: ``maai`` package (cloned at ``external/MaAI/``),
only required when using PyTorch transformer fallback.
"""

from __future__ import annotations

import logging
import os
import struct
import time

import numpy as np
import onnxruntime as ort

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from voice_pipeline.audio.constants import SAMPLE_RATE
from voice_pipeline.core.types import AudioFrame, VAPResult
from voice_pipeline.turn_taking.exceptions import VAPError

logger = logging.getLogger("voice_pipeline.turn_taking.maai_vap")


class MaAIVAPModel:
    """MaAI VAP inference (ONNX encoder + transformer), synchronous.

    ``infer(user, robot)`` runs one frame through the encoder + transformer
    and returns the voice-activity estimate. Internally stateful — LSTM
    hidden states for the encoder and KV-cache for the transformer carry
    across calls; ``reset()`` clears them for a new turn.

    Not thread-safe. Use ``ThreadedVAP`` for the pipeline (it owns the
    inference thread and serializes access). Dev tools (bench/trace) call
    ``infer`` directly on a single thread.

    Args:
        tts_sample_rate: Robot(TTS) 출력 샘플레이트. 리샘플링 기준.
    """

    ENCODER_ONNX_PATH = "models/maai/encoder_10hz_5s.onnx"
    TRANSFORMER_ONNX_PATH = "models/maai/transformer_en_5s.onnx"

    _USE_ONNX_TRANSFORMER = True  # True=ONNX transformer / False=PyTorch fallback
    _USE_TORCH_COMPILE = True  # torch.compile 활성화 (PyTorch fallback 전용)
    _FRAME_RATE = 10  # VAP 추론 프레임 레이트 (Hz)
    _CONTEXT_LEN_SEC = 5.0  # KV 캐시 컨텍스트 길이 (초)
    _ORT_THREADS = 1  # ONNX Runtime intra-op 스레드 수 (RPi 5 최적 1)
    _PT_THREADS = 1  # PyTorch 스레드 수 (PyTorch fallback 전용)

    _DEFAULT_RESULT = VAPResult(0.0, 0.0, False)  # 추론 실패/초기 상태 반환값
    _LANG = "en"  # MaAI 언어 코드 (PyTorch fallback 경로 전용)
    _VAD_THRESHOLD = 0.5  # user_is_speaking 임계값
    _TORCH_DEVICE = "cpu"  # MaAI PyTorch 디바이스
    _TORCH_COMPILE_MODE = "reduce-overhead"  # torch.compile 모드

    # MaAI 모델 아키텍처 (모든 lang/frame_rate 변종에 고정)
    _MAAI_DIM = 256
    _MAAI_NUM_HEADS = 4
    _MAAI_HEAD_DIM = _MAAI_DIM // _MAAI_NUM_HEADS
    _MAAI_CH_LAYERS = 1
    _MAAI_CROSS_LAYERS = 3
    _FRAME_CTX_PADDING = 320  # encoder 입력 padding (MaAI 아키텍처 고정)

    def __init__(
        self,
        tts_sample_rate: int,
    ) -> None:
        # Snapshot class vars to instance attrs (safe for concurrent fixtures
        # where another instance may mutate class vars afterwards).
        self._robot_sample_rate = tts_sample_rate
        self._use_onnx_transformer = self._USE_ONNX_TRANSFORMER
        self._use_torch_compile = not self._use_onnx_transformer and self._USE_TORCH_COMPILE
        self._frame_rate = self._FRAME_RATE
        self._audio_frame_size = SAMPLE_RATE // self._frame_rate + self._FRAME_CTX_PADDING
        self._audio_context_len = int(self._CONTEXT_LEN_SEC * self._frame_rate)

        # ORT session options (shared by encoder and transformer)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = self._ORT_THREADS

        # Encoder (ONNX from file)
        if not self.ENCODER_ONNX_PATH:
            raise VAPError("ENCODER_ONNX_PATH is required")
        if not os.path.isfile(self.ENCODER_ONNX_PATH):
            raise VAPError(f"Encoder ONNX file not found: {self.ENCODER_ONNX_PATH}")
        try:
            self._sess1 = ort.InferenceSession(self.ENCODER_ONNX_PATH, sess_opts)
            self._sess2 = ort.InferenceSession(self.ENCODER_ONNX_PATH, sess_opts)
        except Exception as exc:
            raise VAPError(f"Failed to load ONNX encoder: {exc}") from exc

        # Transformer
        if self._use_onnx_transformer:
            if not os.path.isfile(self.TRANSFORMER_ONNX_PATH):
                raise VAPError(f"Transformer ONNX file not found: {self.TRANSFORMER_ONNX_PATH}")
            try:
                self._tfm_sess = ort.InferenceSession(self.TRANSFORMER_ONNX_PATH, sess_opts)
                logger.info("ONNX transformer loaded from %s", self.TRANSFORMER_ONNX_PATH)
            except Exception as exc:
                raise VAPError(f"Failed to load ONNX transformer: {exc}") from exc
        else:
            # PyTorch transformer requires torch
            if torch is None:
                raise VAPError("torch is required when _USE_ONNX_TRANSFORMER is False")
            torch.set_num_threads(self._PT_THREADS)

            # Load MaAI for PyTorch transformer
            try:
                from maai import Maai, MaaiInput

                ch1 = MaaiInput.Chunk()
                ch2 = MaaiInput.Chunk()
                self._maai = Maai(
                    mode="vap",
                    lang=self._LANG,
                    frame_rate=self._frame_rate,
                    context_len_sec=self._CONTEXT_LEN_SEC,
                    audio_ch1=ch1,
                    audio_ch2=ch2,
                    device=self._TORCH_DEVICE,
                    use_kv_cache=True,
                )
            except Exception as exc:
                raise VAPError(f"Failed to load MaAI: {exc}") from exc

            self._vap = self._maai.vap
            if self._USE_TORCH_COMPILE:
                try:
                    self._vap_forward = torch.compile(self._vap.forward, mode=self._TORCH_COMPILE_MODE)
                    logger.info("torch.compile enabled for transformer")
                except Exception:
                    logger.warning(
                        "torch.compile failed, falling back to eager mode",
                        exc_info=True,
                    )
                    self._vap_forward = self._vap.forward
            else:
                self._vap_forward = self._vap.forward

        # Mutable state
        self._h1 = np.zeros((1, 1, self._MAAI_DIM), dtype=np.float32)
        self._c1 = np.zeros((1, 1, self._MAAI_DIM), dtype=np.float32)
        self._h2 = np.zeros((1, 1, self._MAAI_DIM), dtype=np.float32)
        self._c2 = np.zeros((1, 1, self._MAAI_DIM), dtype=np.float32)
        self._buf_x1 = np.zeros(self._FRAME_CTX_PADDING, dtype=np.float32)
        self._buf_x2 = np.zeros(self._FRAME_CTX_PADDING, dtype=np.float32)
        self._vap_cache: dict | None = None
        self._last_result = self._DEFAULT_RESULT

        # Warmup: pre-allocate ORT buffers / trigger torch.compile
        self._warmup()
        self.reset()

        mode = "onnx" if self._use_onnx_transformer else "pytorch"
        logger.info(
            "MaAIVAPModel initialized: frame_rate=%d, context=%.1fs, ort_threads=%d, pt_threads=%d, transformer=%s",
            self._frame_rate,
            self._CONTEXT_LEN_SEC,
            self._ORT_THREADS,
            self._PT_THREADS,
            mode,
        )

    def infer(self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None) -> VAPResult:
        """Run inference on one (possibly batch-drained) audio chunk.

        Returns the latest voice-activity estimate. When the internal frame
        buffer has not yet accumulated a full inference window, returns the
        previous result (so callers always get a valid ``VAPResult``).
        """
        try:
            x1 = self._pcm_to_numpy(user_audio)
            if robot_audio is not None:
                x2 = self._pcm_to_numpy(robot_audio)
                x2 = self._resample_robot(x2, len(x1))
            else:
                x2 = np.zeros(len(x1), dtype=np.float32)

            result = self._process_frame(x1, x2)
            if result is not None:
                self._last_result = result
        except Exception:
            logger.warning("Error in VAP inference, keeping cached result", exc_info=True)
        return self._last_result

    def reset(self) -> None:
        """Clear encoder LSTM state, transformer KV cache, and audio buffers."""
        self._h1[:] = 0
        self._c1[:] = 0
        self._h2[:] = 0
        self._c2[:] = 0
        self._buf_x1 = np.zeros(self._FRAME_CTX_PADDING, dtype=np.float32)
        self._buf_x2 = np.zeros(self._FRAME_CTX_PADDING, dtype=np.float32)

        if self._use_torch_compile:
            # torch.compile: zero values but keep tensor shapes
            # to avoid recompilation on shape change
            self._zero_pytorch_cache()
        else:
            # ONNX and PyTorch eager: safe to clear entirely
            self._vap_cache = None

        self._last_result = self._DEFAULT_RESULT

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
        n_frames = 2 if self._use_onnx_transformer else self._audio_context_len

        logger.info(
            "Warmup: %d frames (transformer=%s)...",
            n_frames,
            "onnx" if self._use_onnx_transformer else "pytorch",
        )

        # Warmup encoder ORT sessions
        dummy_wav = np.zeros((1, 1, self._audio_frame_size), dtype=np.float32)
        dummy_h = np.zeros((1, 1, self._MAAI_DIM), dtype=np.float32)
        dummy_c = np.zeros((1, 1, self._MAAI_DIM), dtype=np.float32)
        for sess in (self._sess1, self._sess2):
            sess.run(None, {"waveform": dummy_wav, "h_in": dummy_h, "c_in": dummy_c})

        # Warmup transformer
        dummy_e = np.zeros((1, 1, self._MAAI_DIM), dtype=np.float32)
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

        t0 = time.monotonic()

        # ONNX encoder (both channels)
        wav1 = self._buf_x1.reshape(1, 1, -1)
        wav2 = self._buf_x2.reshape(1, 1, -1)

        e1_np, self._h1, self._c1 = self._sess1.run(None, {"waveform": wav1, "h_in": self._h1, "c_in": self._c1})
        e2_np, self._h2, self._c2 = self._sess2.run(None, {"waveform": wav2, "h_in": self._h2, "c_in": self._c2})

        # Transformer forward
        if self._use_onnx_transformer:
            out = self._process_transformer_onnx(e1_np, e2_np)
        else:
            out = self._process_transformer_pytorch(e1_np, e2_np)

        elapsed_ms = (time.monotonic() - t0) * 1000
        budget_ms = 1000.0 / self._frame_rate
        if elapsed_ms > budget_ms:
            logger.warning("VAP inference slow: %.0fms (budget %.0fms)", elapsed_ms, budget_ms)
        # else:
        #     logger.debug("VAP inference: %.0fms", elapsed_ms)

        # Buffer trimming
        self._buf_x1 = self._buf_x1[-self._FRAME_CTX_PADDING :].copy()
        self._buf_x2 = self._buf_x2[-self._FRAME_CTX_PADDING :].copy()

        # Convert to VAPResult
        p_now = float(out["p_now"])
        p_fut = float(out["p_future"])
        user_is_speaking = float(out["vad"][0]) > self._VAD_THRESHOLD

        return VAPResult(p_now, p_fut, user_is_speaking)

    def _process_transformer_onnx(self, e1_np: np.ndarray, e2_np: np.ndarray) -> dict:
        """Run transformer via ONNX Runtime."""
        limit = self._audio_context_len - 1

        # Build cache inputs
        if self._vap_cache is None:
            empty_ch = np.zeros(
                (self._MAAI_CH_LAYERS, 1, self._MAAI_NUM_HEADS, 0, self._MAAI_HEAD_DIM), dtype=np.float32
            )
            empty_cr = np.zeros(
                (self._MAAI_CROSS_LAYERS, 1, self._MAAI_NUM_HEADS, 0, self._MAAI_HEAD_DIM), dtype=np.float32
            )
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
                    [t[..., -limit:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t for t in k_list],
                    [t[..., -limit:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t for t in v_list],
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
        if self._robot_sample_rate != SAMPLE_RATE:
            ratio = SAMPLE_RATE / self._robot_sample_rate
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
