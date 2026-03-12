"""VAP ONNX pipeline: shared utility for benchmark and stress test scripts.

ONNX encoder + PyTorch transformer pipeline that bypasses MaAI's process()
to eliminate torch<->numpy conversion overhead.

This module is imported by benchmark, stress test, and visualization scripts.
It is NOT used in production — see ``voice_pipeline.turn_taking.maai_vap``
for the production IVAP implementation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from voice_pipeline.turn_taking.onnx_export import export_encoder_onnx


class VapOnnxPipeline:
    """Custom VAP pipeline: ONNX encoder + PyTorch transformer.

    Exports the ONNX encoder from the MaAI instance itself so that
    downsample weights always match the loaded model variant.
    """

    def __init__(
        self,
        frame_rate: int,
        context_len_sec: float,
        lang: str = "en",
        device: str = "cpu",
        ort_threads: int = 4,
    ):
        from maai import Maai, MaaiInput

        ch1 = MaaiInput.Chunk()
        ch2 = MaaiInput.Chunk()
        self._maai = Maai(
            mode="vap",
            lang=lang,
            frame_rate=frame_rate,
            context_len_sec=context_len_sec,
            audio_ch1=ch1,
            audio_ch2=ch2,
            device=device,
            use_kv_cache=True,
        )
        self.vap = self._maai.vap

        # Export ONNX encoder from this MaAI instance (weight-matched)
        onnx_path = export_encoder_onnx(self._maai, frame_rate)

        # ONNX encoder sessions (one per channel)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = ort_threads

        self.sess1 = ort.InferenceSession(onnx_path, sess_opts)
        self.sess2 = ort.InferenceSession(onnx_path, sess_opts)

        os.unlink(onnx_path)  # cleanup temp file

        # Encoder LSTM hidden states
        self.h1 = np.zeros((1, 1, 256), dtype=np.float32)
        self.c1 = np.zeros((1, 1, 256), dtype=np.float32)
        self.h2 = np.zeros((1, 1, 256), dtype=np.float32)
        self.c2 = np.zeros((1, 1, 256), dtype=np.float32)

        # Audio buffering (same logic as MaAI.process)
        self.frame_rate = frame_rate
        self.sampling_rate = 16000
        self.frame_contxt_padding = 320
        self.audio_frame_size = (
            self.sampling_rate // self.frame_rate + self.frame_contxt_padding
        )
        self.audio_context_len = int(context_len_sec * frame_rate)

        self.current_x1 = np.zeros(self.frame_contxt_padding, dtype=np.float32)
        self.current_x2 = np.zeros(self.frame_contxt_padding, dtype=np.float32)
        self.vap_cache = None

    def reset(self):
        self.h1[:] = 0
        self.c1[:] = 0
        self.h2[:] = 0
        self.c2[:] = 0
        self.current_x1 = np.zeros(self.frame_contxt_padding, dtype=np.float32)
        self.current_x2 = np.zeros(self.frame_contxt_padding, dtype=np.float32)
        self.vap_cache = None

    def process(self, x1: np.ndarray, x2: np.ndarray) -> dict | None:
        # 1. Audio buffering (identical to MaAI)
        self.current_x1 = np.concatenate([self.current_x1, x1])
        self.current_x2 = np.concatenate([self.current_x2, x2])

        if len(self.current_x1) < self.audio_frame_size:
            return None

        # 2. ONNX encoder — numpy directly, no torch intermediate
        wav1 = self.current_x1.reshape(1, 1, -1)
        wav2 = self.current_x2.reshape(1, 1, -1)

        e1_np, self.h1, self.c1 = self.sess1.run(
            None, {"waveform": wav1, "h_in": self.h1, "c_in": self.c1}
        )
        e2_np, self.h2, self.c2 = self.sess2.run(
            None, {"waveform": wav2, "h_in": self.h2, "c_in": self.c2}
        )

        # 3. Single numpy→torch conversion
        e1 = torch.from_numpy(e1_np)
        e2 = torch.from_numpy(e2_np)

        # 4. Transformer forward (unchanged PyTorch)
        with torch.no_grad():
            out, self.vap_cache = self.vap.forward(e1, e2, cache=self.vap_cache)

        # 5. Cache trimming (identical to MaAI)
        if self.vap_cache is not None:
            limit = self.audio_context_len - 1
            new_cache = {}
            for key, (k_list, v_list) in self.vap_cache.items():
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
            self.vap_cache = new_cache

        # 6. Buffer trimming (identical to MaAI)
        self.current_x1 = self.current_x1[-self.frame_contxt_padding :].copy()
        self.current_x2 = self.current_x2[-self.frame_contxt_padding :].copy()

        return out


def create_maai(frame_rate: int, context_len_sec: float):
    """Create a standard MaAI instance for baseline comparison."""
    from maai import Maai, MaaiInput

    ch1 = MaaiInput.Chunk()
    ch2 = MaaiInput.Chunk()
    return Maai(
        mode="vap",
        lang="en",
        frame_rate=frame_rate,
        context_len_sec=context_len_sec,
        audio_ch1=ch1,
        audio_ch2=ch2,
        device="cpu",
        use_kv_cache=True,
    )
