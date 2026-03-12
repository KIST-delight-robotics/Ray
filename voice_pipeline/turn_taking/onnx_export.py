"""ONNX encoder export utilities for MaAI VAP.

Provides the EncoderONNXWrapper and export function used by both
the production MaAIVAPWrapper and benchmark scripts.
"""

from __future__ import annotations

import tempfile

import torch
import torch.nn as nn


class EncoderONNXWrapper(nn.Module):
    """Minimal wrapper around CPC encoder for clean ONNX export.

    Replaces einops Rearrange with torch.permute and exposes
    LSTM hidden state as explicit I/O for incremental inference.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.g_encoder = encoder.encoder.gEncoder
        self.g_ar = encoder.encoder.gAR.baseNet
        ds = encoder.downsample
        self.ds_conv = ds[1]
        self.ds_ln = ds[2].ln
        self.ds_act = ds[3]

    def forward(
        self, waveform: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.g_encoder(waveform)
        z = z.permute(0, 2, 1)
        z = z[:, 1:-1, :]
        z, (h_out, c_out) = self.g_ar(z, (h_in, c_in))
        z = z.permute(0, 2, 1)
        z = self.ds_conv(z)
        z = z.permute(0, 2, 1)
        z = self.ds_ln(z)
        z = self.ds_act(z)
        return z, h_out, c_out


def export_encoder_onnx(maai_instance: object, frame_rate: int) -> str:
    """Export ONNX encoder from a live MaAI instance (weight-matched).

    Returns the path to the temporary ONNX file. Caller is responsible
    for cleanup (``os.unlink``).
    """
    encoder = maai_instance.vap.encoder1
    encoder.eval()

    wrapper = EncoderONNXWrapper(encoder)
    wrapper.eval()

    samples_per_frame = 16000 // frame_rate
    input_size = 320 + samples_per_frame

    dummy_wav = torch.randn(1, 1, input_size)
    dummy_h = torch.zeros(1, 1, 256)
    dummy_c = torch.zeros(1, 1, 256)

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()

    torch.onnx.export(
        wrapper,
        (dummy_wav, dummy_h, dummy_c),
        tmp.name,
        input_names=["waveform", "h_in", "c_in"],
        output_names=["embedding", "h_out", "c_out"],
        dynamic_axes={"waveform": {2: "n_samples"}, "embedding": {1: "n_frames"}},
        opset_version=17,
        dynamo=False,
    )
    return tmp.name
