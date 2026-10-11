"""실험 녹음 — 사용자(실험이 넣은 음성)와 모델(받은 음성)을 한 시간축에 놓은 스테레오 WAV.

왼쪽이 사용자, 오른쪽이 모델이다(OpenAI 서버 녹음과 같은 배치). 시간축은 받은 모델 오디오이고, 사용자 음성은 넣기
시작한 시각의 받은 오디오 위치에 놓는다 — 레이블과 같은 위치. 입력이 서버에 닿기까지의 지연은 반영하지 않는다.
"""

from __future__ import annotations

import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def resample(pcm: bytes, src: int, dst: int) -> bytes:
    """PCM16 mono 선형 보간 리샘플."""
    if src == dst:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = int(len(x) * dst / src)
    y = np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x)
    return y.astype(np.int16).tobytes()


def read_model_channel(path: Path) -> bytes:
    """녹음에서 모델 오디오만 — 스테레오면 오른쪽 채널, 모노(이전 녹음)면 그대로."""
    with wave.open(str(path)) as w:
        channels, frames = w.getnchannels(), w.readframes(w.getnframes())
    if channels == 1:
        return frames
    return np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)[:, 1].copy().tobytes()


def write_conversation_wav(path: Path, model_pcm: bytes, user_clips: Sequence[tuple[float, bytes]], rate: int) -> None:
    """``user_clips`` 는 (시작 초, ``rate`` 의 PCM16 mono). 사용자 음성이 모델 오디오보다 늦게 끝나면 그만큼 늘린다."""
    model = np.frombuffer(model_pcm, dtype=np.int16)
    clips = [(int(round(start * rate)), np.frombuffer(pcm, dtype=np.int16)) for start, pcm in user_clips]
    length = max([len(model), *(i + len(c) for i, c in clips)])
    left = np.zeros(length, dtype=np.int16)
    right = np.zeros(length, dtype=np.int16)
    right[: len(model)] = model
    for i, c in clips:
        left[i : i + len(c)] = c
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.column_stack([left, right]).tobytes())
