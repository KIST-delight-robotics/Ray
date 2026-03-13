"""ONNX export utilities for MaAI VAP.

Provides encoder and transformer ONNX wrappers and export functions
used by both the production MaAIVAPWrapper and benchmark scripts.
"""

from __future__ import annotations

import math
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Transformer ONNX wrapper
# ---------------------------------------------------------------------------

# Cache layout: 6 groups, each (k_stack, v_stack).
# ar1/ar2: shape (n_ch_layers, B, nh, T, hd)
# cross1/cross2/cross1_c/cross2_c: shape (n_cross_layers, B, nh, T, hd)
_CACHE_GROUPS = ("ar1", "ar2", "cross1", "cross2", "cross1_c", "cross2_c")


def _build_alibi_mask(num_heads: int, max_T: int) -> torch.Tensor:
    """Pre-compute ALiBi causal mask for up to *max_T* positions.

    Returns shape ``(1, num_heads, max_T, max_T)``.
    """

    def _get_slopes(n: int) -> list[float]:
        def _power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * start**i for i in range(n)]

        if math.log2(n).is_integer():
            return _power_of_2(n)
        closest = 2 ** math.floor(math.log2(n))
        return _power_of_2(closest) + _get_slopes(2 * closest)[0::2][: n - closest]

    m = torch.tensor(_get_slopes(num_heads))  # (nh,)

    # Relative bias: (1, nh, 1, max_T)
    rel = torch.arange(max_T).view(1, 1, -1).expand(1, num_heads, -1).float()
    alibi = rel * m.view(1, -1, 1)  # (1, nh, max_T)
    alibi = alibi.unsqueeze(-2)  # (1, nh, 1, max_T)

    # Causal mask
    causal = torch.tril(torch.ones(max_T, max_T)).view(1, 1, max_T, max_T)
    causal = causal.repeat(1, num_heads, 1, 1)
    causal.masked_fill_(causal == 0, float("-inf"))
    causal.masked_fill_(causal == 1, 0.0)

    mask = alibi + causal  # (1, nh, max_T, max_T)
    return mask


class _MHAForward(nn.Module):
    """ONNX-friendly multi-head attention forward (no einops, no state)."""

    def __init__(self, mha: nn.Module, alibi_mask: torch.Tensor) -> None:
        super().__init__()
        self.key = mha.key
        self.query = mha.query
        self.value = mha.value
        self.proj = mha.proj
        self.num_heads = mha.num_heads
        self.scale = 1.0 / math.sqrt(mha.dim)
        self.register_buffer("alibi_mask", alibi_mask)

    def _unstack(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.reshape(B, T, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

    def _stack(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, T, H * D)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._unstack(self.query(Q))
        k = self._unstack(self.key(K))
        v = self._unstack(self.value(V))

        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

        att = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        T_total = att.size(-1)
        T_query = att.size(-2)
        mask = self.alibi_mask[..., :T_total, :T_total]
        mask = mask[..., -T_query:, :]
        att = att + mask

        att = F.softmax(att, dim=-1)
        y = att @ v
        y = self._stack(y)
        y = self.proj(y)
        return y, k, v


class _TransformerLayerForward(nn.Module):
    """Single transformer layer: self-attn + optional cross-attn + FFN."""

    def __init__(
        self,
        layer: nn.Module,
        alibi_mask: torch.Tensor,
        has_cross: bool,
    ) -> None:
        super().__init__()
        self.ln_self = layer.ln_self_attn
        self.ln_ffn = layer.ln_ffnetwork
        self.ffn = layer.ffnetwork
        self.self_attn = _MHAForward(layer.mha, alibi_mask)
        self.has_cross = has_cross
        if has_cross:
            self.ln_cross = layer.ln_src_attn
            self.cross_attn = _MHAForward(layer.mha_cross, alibi_mask)

    def forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        past_k_c: torch.Tensor,
        past_v_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.ln_self(x)
        sa, k, v = self.self_attn(z, z, z, past_k, past_v)
        x = x + sa

        k_c = past_k_c
        v_c = past_v_c
        if self.has_cross:
            z = self.ln_cross(x)
            ca, k_c, v_c = self.cross_attn(z, src, src, past_k_c, past_v_c)
            x = x + ca

        x = x + self.ffn(self.ln_ffn(x))
        return x, k, v, k_c, v_c


class TransformerONNXWrapper(nn.Module):
    """ONNX-exportable wrapper for the full VapGPT transformer.

    Replaces dict-based KV cache with flat stacked tensors,
    removes einops, and pre-computes ALiBi mask.

    KV cache convention (per group):
      - ``k``: ``(n_layers, 1, num_heads, T_cached, head_dim)``
      - ``v``: same shape
      - First call: pass zero-length tensors ``(n_layers, 1, nh, 0, hd)``
    """

    def __init__(self, vap: nn.Module, max_context: int = 256) -> None:
        super().__init__()
        conf = vap.conf
        num_heads = conf.num_heads
        alibi_mask = _build_alibi_mask(num_heads, max_context)
        self.register_buffer("_alibi", alibi_mask)

        # Channel tower (shared weights, 1 layer)
        self.ch_layers = nn.ModuleList()
        for layer in vap.ar_channel.layers:
            self.ch_layers.append(_TransformerLayerForward(layer, alibi_mask, has_cross=False))
        self.n_ch_layers = len(self.ch_layers)

        # Cross tower (3 layers)
        self.cross_layers = nn.ModuleList()
        for layer in vap.ar.layers:
            self.cross_layers.append(_TransformerLayerForward(layer, alibi_mask, has_cross=True))
        self.n_cross_layers = len(self.cross_layers)

        # Combinator + heads
        self.combinator = vap.ar.combinator
        self.va_classifier = vap.va_classifier
        self.vap_head = vap.vap_head

        # Objective constants (pre-compute for ONNX)
        objective = vap.objective
        idx = torch.arange(objective.codebook.n_classes)
        states = objective.codebook.decode(idx)  # (n_classes, 2, n_bins)
        # p_now bins [0,1], p_future bins [2,3]
        abp_now = states[:, :, 0:2].sum(-1)  # (n_classes, 2)
        abp_fut = states[:, :, 2:4].sum(-1)  # (n_classes, 2)
        self.register_buffer("abp_now", abp_now)
        self.register_buffer("abp_fut", abp_fut)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        # ar1 cache
        ar1_k: torch.Tensor,
        ar1_v: torch.Tensor,
        # ar2 cache
        ar2_k: torch.Tensor,
        ar2_v: torch.Tensor,
        # cross1 cache (self-attn of speaker 1 in stereo)
        cross1_k: torch.Tensor,
        cross1_v: torch.Tensor,
        # cross2 cache (self-attn of speaker 2 in stereo)
        cross2_k: torch.Tensor,
        cross2_v: torch.Tensor,
        # cross1_c cache (cross-attn: speaker 1 <- speaker 2)
        cross1_c_k: torch.Tensor,
        cross1_c_v: torch.Tensor,
        # cross2_c cache (cross-attn: speaker 2 <- speaker 1)
        cross2_c_k: torch.Tensor,
        cross2_c_v: torch.Tensor,
    ) -> tuple[
        torch.Tensor,  # p_now
        torch.Tensor,  # p_future
        torch.Tensor,  # vad1
        torch.Tensor,  # vad2
        # 12 updated cache tensors
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # --- Channel tower (shared weights, applied to each speaker) ---
        o1 = x1
        new_ar1_k_list: list[torch.Tensor] = []
        new_ar1_v_list: list[torch.Tensor] = []
        dummy = torch.zeros(1, 1, 0, 1)  # unused cross-attn placeholder
        for i, layer in enumerate(self.ch_layers):
            o1, k, v, _, _ = layer(
                o1,
                o1,  # src=self for no cross-attn
                ar1_k[i],
                ar1_v[i],
                dummy,
                dummy,
            )
            new_ar1_k_list.append(k)
            new_ar1_v_list.append(v)

        o2 = x2
        new_ar2_k_list: list[torch.Tensor] = []
        new_ar2_v_list: list[torch.Tensor] = []
        for i, layer in enumerate(self.ch_layers):
            o2, k, v, _, _ = layer(
                o2,
                o2,
                ar2_k[i],
                ar2_v[i],
                dummy,
                dummy,
            )
            new_ar2_k_list.append(k)
            new_ar2_v_list.append(v)

        # --- Cross tower ---
        z1, z2 = o1, o2
        new_c1_k: list[torch.Tensor] = []
        new_c1_v: list[torch.Tensor] = []
        new_c2_k: list[torch.Tensor] = []
        new_c2_v: list[torch.Tensor] = []
        new_c1c_k: list[torch.Tensor] = []
        new_c1c_v: list[torch.Tensor] = []
        new_c2c_k: list[torch.Tensor] = []
        new_c2c_v: list[torch.Tensor] = []

        for i, layer in enumerate(self.cross_layers):
            # Save originals — TransformerStereoLayer uses original inputs
            # as cross-attention sources, not the updated ones.
            z1_in, z2_in = z1, z2
            # Speaker 1: self-attn on z1, cross-attn from original z2
            z1, k1, v1, k1c, v1c = layer(
                z1_in,
                z2_in,
                cross1_k[i],
                cross1_v[i],
                cross1_c_k[i],
                cross1_c_v[i],
            )
            # Speaker 2: self-attn on z2, cross-attn from original z1
            z2, k2, v2, k2c, v2c = layer(
                z2_in,
                z1_in,
                cross2_k[i],
                cross2_v[i],
                cross2_c_k[i],
                cross2_c_v[i],
            )
            new_c1_k.append(k1)
            new_c1_v.append(v1)
            new_c2_k.append(k2)
            new_c2_v.append(v2)
            new_c1c_k.append(k1c)
            new_c1c_v.append(v1c)
            new_c2c_k.append(k2c)
            new_c2c_v.append(v2c)

        # --- Combinator + heads ---
        x_combined = self.combinator(z1, z2)
        logits = self.vap_head(x_combined)
        probs = logits.softmax(dim=-1)

        # p_now — output both speaker probabilities: (2,)
        p_now_all = torch.einsum("bid,dc->bic", probs, self.abp_now)
        p_now_all = p_now_all / (p_now_all.sum(-1, keepdim=True) + 1e-5)
        p_now = p_now_all[0, -1]  # (2,): [speaker1, speaker2]

        # p_future — same
        p_fut_all = torch.einsum("bid,dc->bic", probs, self.abp_fut)
        p_fut_all = p_fut_all / (p_fut_all.sum(-1, keepdim=True) + 1e-5)
        p_future = p_fut_all[0, -1]  # (2,)

        # VAD
        vad1 = self.va_classifier(o1).sigmoid()[0, -1, 0]
        vad2 = self.va_classifier(o2).sigmoid()[0, -1, 0]

        # Stack cache outputs
        out_ar1_k = torch.stack(new_ar1_k_list)
        out_ar1_v = torch.stack(new_ar1_v_list)
        out_ar2_k = torch.stack(new_ar2_k_list)
        out_ar2_v = torch.stack(new_ar2_v_list)
        out_c1_k = torch.stack(new_c1_k)
        out_c1_v = torch.stack(new_c1_v)
        out_c2_k = torch.stack(new_c2_k)
        out_c2_v = torch.stack(new_c2_v)
        out_c1c_k = torch.stack(new_c1c_k)
        out_c1c_v = torch.stack(new_c1c_v)
        out_c2c_k = torch.stack(new_c2c_k)
        out_c2c_v = torch.stack(new_c2c_v)

        return (
            p_now,
            p_future,
            vad1,
            vad2,
            out_ar1_k,
            out_ar1_v,
            out_ar2_k,
            out_ar2_v,
            out_c1_k,
            out_c1_v,
            out_c2_k,
            out_c2_v,
            out_c1c_k,
            out_c1c_v,
            out_c2c_k,
            out_c2c_v,
        )


def export_transformer_onnx(
    vap: nn.Module,
    max_context: int = 256,
) -> str:
    """Export the VAP transformer to ONNX.

    Returns the path to the temporary ONNX file. Caller is responsible
    for cleanup (``os.unlink``).
    """
    wrapper = TransformerONNXWrapper(vap, max_context)
    wrapper.eval()

    conf = vap.conf
    num_heads = conf.num_heads
    head_dim = conf.dim // num_heads
    n_ch = len(list(vap.ar_channel.layers))
    n_cross = len(list(vap.ar.layers))

    # Dummy inputs: 1-frame embedding + empty cache
    dummy_x = torch.randn(1, 1, conf.dim)
    dummy_ar_k = torch.zeros(n_ch, 1, num_heads, 0, head_dim)
    dummy_ar_v = torch.zeros(n_ch, 1, num_heads, 0, head_dim)
    dummy_cross_k = torch.zeros(n_cross, 1, num_heads, 0, head_dim)
    dummy_cross_v = torch.zeros(n_cross, 1, num_heads, 0, head_dim)

    dummy_inputs = (
        dummy_x,
        dummy_x,
        dummy_ar_k,
        dummy_ar_v,  # ar1
        dummy_ar_k,
        dummy_ar_v,  # ar2
        dummy_cross_k,
        dummy_cross_v,  # cross1
        dummy_cross_k,
        dummy_cross_v,  # cross2
        dummy_cross_k,
        dummy_cross_v,  # cross1_c
        dummy_cross_k,
        dummy_cross_v,  # cross2_c
    )

    input_names = [
        "x1",
        "x2",
        "ar1_k",
        "ar1_v",
        "ar2_k",
        "ar2_v",
        "cross1_k",
        "cross1_v",
        "cross2_k",
        "cross2_v",
        "cross1_c_k",
        "cross1_c_v",
        "cross2_c_k",
        "cross2_c_v",
    ]
    output_names = [
        "p_now",
        "p_future",
        "vad1",
        "vad2",
        "out_ar1_k",
        "out_ar1_v",
        "out_ar2_k",
        "out_ar2_v",
        "out_cross1_k",
        "out_cross1_v",
        "out_cross2_k",
        "out_cross2_v",
        "out_cross1_c_k",
        "out_cross1_c_v",
        "out_cross2_c_k",
        "out_cross2_c_v",
    ]

    # Dynamic axes: T_cached dimension (axis=3) for all cache tensors
    dynamic_axes: dict[str, dict[int, str]] = {
        "x1": {1: "T_new"},
        "x2": {1: "T_new"},
    }
    cache_inputs = input_names[2:]
    cache_outputs = output_names[4:]
    for name in cache_inputs:
        dynamic_axes[name] = {3: "T_cached"}
    for name in cache_outputs:
        dynamic_axes[name] = {3: "T_total"}

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)  # noqa: SIM115
    tmp.close()

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        tmp.name,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    return tmp.name


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

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)  # noqa: SIM115
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
