"""재생 파일의 출력 레벨을 목표치에 맞춘다 — 차임(pw-play), 인사·작별 WAV(C++ play_file).

기준은 GPT-Live 의 말소리다. 대화 시간의 대부분이 그 소리이므로, 그보다 큰 고정 오디오가 끼면
그것만 시끄럽게 들린다. 라이브 출력은 벤더가 정하는 레벨이라 우리가 못 바꾸니, 맞추는 쪽은 항상 파일이다.

척도는 **유성 40ms 프레임 피크**(cpp ``MouthLevelNormalizer`` 와 동일: 40ms 프레임의 절댓값 최대,
``VOICED_PEAK_FLOOR`` 미만은 무음 취급). 통합 LUFS 를 쓰지 않는 이유는 실시간 스트림의 무음 비율이
턴마다 달라 콘텐츠가 아니라 침묵을 재기 때문이다 — 실측에서 같은 4개 세션의 편차가 LUFS 4.7 dB 대
이 척도 3.4 dB 였다.

대표값은 소리 종류에 따라 다르다:

* **말소리** — 프레임 피크의 *중앙값*. 발화 대부분이 앉아 있는 높이.
* **알림음** — *p90*. 차임은 어택이 크고 감쇠 꼬리가 길어 중앙값이 체감 크기를 과소평가한다
  (실측 ``ready.oga``: 중앙값 0.157 / p90 0.456).

정규화 결과는 ``var/audio/normalized/`` 에 16-bit WAV 사본으로 쓰고 그 경로를 재생한다. 재생 순간이
아니라 사본으로 처리하는 이유: 차임은 ``pw-play``, 인사는 C++ 로 경로가 갈리는데 사본이면 두 경로가
같은 장치를 쓰고 C++ · 브리지 프로토콜을 건드릴 필요가 없다. 원본(추적 자산)도 그대로 남는다.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_pipeline.settings import CHIME_TARGET_PEAK, SPEECH_TARGET_PEAK, VAR_DIR

logger = logging.getLogger("voice_pipeline.audio_level")

_CACHE_DIR = f"{VAR_DIR}/audio/normalized"

FRAME_MS = 40  # 분석 프레임 길이. cpp MouthLevelNormalizer 와 같은 값이라 [mouth] gain 로그와 대조된다
VOICED_PEAK_FLOOR = 0.02  # 이 아래 프레임은 무음 취급 (cpp kVoicedPeak). 말소리 최약 구간도 0.05 는 넘는다
GAIN_MAX = 4.0  # 증폭 상한. 녹음이 아주 작은 파일에서 바닥 잡음까지 끌어올리지 않게
PEAK_CEILING = 0.89  # 정규화 후 허용 절대 피크 (≈-1 dBFS). 리샘플러·DAC 클리핑 여유


def _mono(data: np.ndarray) -> np.ndarray:
    """다채널이면 평균해서 mono 로 만든다 (측정용)."""
    return data.mean(axis=1) if data.ndim > 1 else data


def frame_peaks(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """``FRAME_MS`` 프레임별 절댓값 최대를 구해 무음 프레임을 뺀 배열을 돌려준다."""
    size = int(sample_rate * FRAME_MS / 1000)
    if size <= 0 or samples.size < size:
        return np.empty(0)
    frames = samples[: samples.size // size * size].reshape(-1, size)
    peaks = np.abs(frames).max(axis=1)
    return peaks[peaks >= VOICED_PEAK_FLOOR]


def measure(samples: np.ndarray, sample_rate: int, *, speech: bool) -> float:
    """파일의 대표 레벨(0~1)을 구한다. ``speech`` 면 프레임 피크 중앙값, 아니면 p90.

    정규화된 파일을 다시 재면 목표보다 조금 높게 나올 수 있다 — ``VOICED_PEAK_FLOOR`` 가 절대값이라
    감쇠 뒤에는 작은 프레임이 더 많이 잘려 분포가 위로 밀린다. 실측 ``ready.oga`` 에서 0.6 dB 수준이라
    되먹임으로 수렴시키지 않는다.

    Args:
        samples: mono float 샘플 (-1.0~1.0).
        sample_rate: 샘플레이트 (Hz).
        speech: 말소리면 True, 알림음·효과음이면 False.

    Returns:
        대표 레벨. 유성 프레임이 하나도 없으면 0.0.
    """
    peaks = frame_peaks(samples, sample_rate)
    if peaks.size == 0:
        return 0.0
    return float(np.median(peaks) if speech else np.percentile(peaks, 90))


def gain_for(samples: np.ndarray, sample_rate: int, *, speech: bool) -> float:
    """목표 레벨에 맞추는 배율을 구한다. ``GAIN_MAX`` 와 ``PEAK_CEILING`` 양쪽으로 제한된다.

    Args:
        samples: mono float 샘플 (-1.0~1.0).
        sample_rate: 샘플레이트 (Hz).
        speech: 말소리면 True, 알림음·효과음이면 False.

    Returns:
        곱할 배율. 측정 불가(무음)면 1.0.
    """
    level = measure(samples, sample_rate, speech=speech)
    if level <= 0.0:
        return 1.0
    target = SPEECH_TARGET_PEAK if speech else CHIME_TARGET_PEAK
    gain = min(target / level, GAIN_MAX)
    absolute_peak = float(np.abs(samples).max())
    if absolute_peak > 0.0:
        gain = min(gain, PEAK_CEILING / absolute_peak)
    return gain


def _cache_path(src: Path, *, speech: bool) -> Path:
    """원본의 내용·목표가 바뀌면 달라지는 캐시 경로."""
    stat = src.stat()
    target = SPEECH_TARGET_PEAK if speech else CHIME_TARGET_PEAK
    key = f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{target}|{speech}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    return Path(_CACHE_DIR) / f"{src.stem}_{digest}.wav"


def ensure_normalized(src: Path | str, *, speech: bool) -> Path:
    """``src`` 를 목표 레벨로 맞춘 WAV 사본의 경로를 돌려준다. 이미 있으면 그대로 쓴다.

    원본은 건드리지 않는다. 채널 수·샘플레이트는 유지하고 16-bit PCM 으로 쓴다.

    Args:
        src: 원본 오디오 경로. libsndfile 이 읽는 형식(wav/ogg/flac 등)이면 된다.
        speech: 말소리면 True, 알림음·효과음이면 False.

    Returns:
        정규화된 WAV 경로.

    Raises:
        RuntimeError: 원본을 읽지 못했거나 사본을 쓰지 못했을 때.
    """
    src = Path(src)
    dst = _cache_path(src, speech=speech)
    if dst.exists():
        logger.debug("Normalized cache hit: %s", dst)
        return dst

    try:
        data, sample_rate = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio for normalization ({src}): {exc}") from exc

    gain = gain_for(_mono(data), sample_rate, speech=speech)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(dst), data * gain, sample_rate, subtype="PCM_16")
    except Exception as exc:
        raise RuntimeError(f"Failed to write normalized audio ({dst}): {exc}") from exc

    logger.info("Normalized %s → %s (gain %.2f, speech=%s)", src, dst, gain, speech)
    return dst
