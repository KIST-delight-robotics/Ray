"""Generate WAV files for eval questions using ElevenLabs TTS.

Assigns voices via round-robin from VOICES — an age-diverse set (child,
elderly M/F, young, middle-aged) so ASR is stressed across speaker types.
Multi-turn scenarios use one voice per scenario (all questions in the same
scenario share a voice). ElevenLabs library voices require a paid plan
(ELEVENLABS_API_KEY); free-tier keys 402 on the professional/library voices.

After generation, every WAV in the output directory is RMS-normalized —
neither vendor exposes loudness normalization and voices differ in level
(elderly/child library voices especially), which skews ASR/VAD results.

Usage:
    uv run python -m evaluation.prepare_audio data/eval/questions.json
    uv run python -m evaluation.prepare_audio data/eval/questions.json --output-dir data/eval/wav
    uv run python -m evaluation.prepare_audio data/eval/questions.json --force
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from voice_pipeline.adapters.tts_elevenlabs import ElevenLabsTTS
from voice_pipeline.greeting_audio import synthesize_to_wav

# (manifest/파일명용 이름, ElevenLabs voice ID). 이름은 score.py by_voice 집계와
# 파일명(`<id>_<name>.wav`)에 쓰임. 연령·성별·억양을 골고루 섞어 화자 다양성 확보.
VOICES = [
    ("omar_j", "S7IsvAvEoDfui6GSZK3A"),  # 아동 남 (indian)
    ("emma", "pPdl9cQBQq4p6mRkZy2Z"),  # 아동 여
    ("grandfather_joe", "0lp4RIz96WD1RUtvEu3Q"),  # 노인 남 (british)
    ("bill", "pqHfZKP75CvOlQylNhV4"),  # 노인 남 (american)
    ("grandma_rachel", "0rEo3eAjssGDUCXHYENf"),  # 노인 여 (us southern)
    ("beatrice", "kkPJzQOWz2Oz9cUaEaQd"),  # 노인 여 (british)
    ("sarah", "EXAVITQu4vr4xnSDxMaL"),  # 젊은 여 (american)
    ("charlie", "IKne3meq5aSn9XLyUdCD"),  # 젊은 남 (australian)
    ("george", "JBFqnCBsd6RMkjVDRZzb"),  # 중년 남 (british)
    ("alice", "Xb7hH8MSUJpSbSDYk0k2"),  # 중년 여 (british)
]


def _get_tts(voice_id: str, cache: dict[str, ElevenLabsTTS]) -> ElevenLabsTTS:
    if voice_id not in cache:
        tts = ElevenLabsTTS()
        tts._VOICE_ID = voice_id
        cache[voice_id] = tts
    return cache[voice_id]


_PEAK_CEILING = 0.95  # 정규화 후 샘플 절대값 상한 — 클리핑 방지
_GAIN_TOLERANCE = 0.02  # 이 비율 이내의 게인 변화는 재기록 생략 (재실행 멱등성)

_TRIM_KEEP_SEC = 0.2  # 발화 끝 이후 남길 여유 — 끝 자음/여운 보호
_TRIM_PEAK_RATIO = 0.02  # 발화 검출 임계 = peak 대비 비율 (TTS 꼬리는 디지털 무음에 가까움)
_TRIM_MIN_THRESHOLD = 0.005  # 임계 하한 (절대값)
_TRIM_TOLERANCE_SEC = 0.05  # 이 이하의 트림은 생략 (재실행 멱등성)


def trim_trailing_silence(path: Path, keep_sec: float = _TRIM_KEEP_SEC) -> str:
    """Cut trailing silence beyond *keep_sec* after the last audible sample.

    TTS가 붙이는 꼬리 무음(0.5~1.4초 실측)은 질문 재생 종료 시점과 실제 발화
    종료 시점을 어긋나게 해 턴 감지 지연 측정을 왜곡한다. Returns action description.
    """
    with wave.open(str(path)) as w:
        params = w.getparams()
        pcm = w.readframes(w.getnframes())
    if params.sampwidth != 2:
        return "skip (not 16-bit)"

    samples = np.frombuffer(pcm, dtype=np.int16)
    if len(samples) == 0:
        return "skip (empty)"
    amp = np.abs(samples.astype(np.float32)) / 32768.0
    peak = float(amp.max())
    if peak < _TRIM_MIN_THRESHOLD:
        return "skip (silence)"

    thresh = max(_TRIM_MIN_THRESHOLD, peak * _TRIM_PEAK_RATIO)
    last_audible = int(np.where(amp > thresh)[0][-1])
    keep = last_audible + 1 + int(keep_sec * params.framerate) * params.nchannels
    keep -= keep % params.nchannels  # 프레임 경계 정렬
    keep = min(len(samples), keep)

    cut_sec = (len(samples) - keep) / (params.framerate * params.nchannels)
    if cut_sec <= _TRIM_TOLERANCE_SEC:
        return "ok"

    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes(samples[:keep].tobytes())
    return f"tail -{cut_sec:.2f}s"


def normalize_wav(path: Path, target_rms: float) -> str:
    """Scale a 16-bit WAV to *target_rms*, peak-limited. Returns action description."""
    with wave.open(str(path)) as w:
        params = w.getparams()
        pcm = w.readframes(w.getnframes())
    if params.sampwidth != 2:
        return "skip (not 16-bit)"

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if len(samples) == 0:
        return "skip (empty)"
    rms = float(np.sqrt(np.mean(samples**2)))
    if rms < 1e-4:
        return "skip (silence)"

    gain = target_rms / rms
    peak = float(np.abs(samples).max())
    if peak * gain > _PEAK_CEILING:
        gain = _PEAK_CEILING / peak
    if abs(gain - 1.0) < _GAIN_TOLERANCE:
        return "ok"

    scaled = np.clip(samples * gain, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes((scaled * 32767.0).astype(np.int16).tobytes())
    return f"{20 * np.log10(gain):+.1f} dB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval question WAV files")
    parser.add_argument("questions", help="Path to questions JSON")
    parser.add_argument("--output-dir", default="data/eval/wav", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Regenerate existing files")
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Normalization target RMS in linear scale (0.1 ≈ -20 dBFS)",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Skip RMS normalization pass")
    parser.add_argument("--no-trim", action="store_true", help="Skip trailing-silence trim pass")
    args = parser.parse_args()

    data = json.loads(Path(args.questions).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts_cache: dict[str, ElevenLabsTTS] = {}
    manifest: dict[str, dict[str, str]] = {}
    generated = 0
    total = 0
    voice_idx = 0

    print(f"Preparing question WAVs → {output_dir}")
    print(f"  voices: {len(VOICES)} (ElevenLabs)")

    for suite in data["suites"]:
        if suite.get("multi_turn"):
            for scenario in suite.get("scenarios", []):
                name, voice_id = VOICES[voice_idx % len(VOICES)]
                voice_idx += 1
                for q in scenario["questions"]:
                    total += 1
                    wav_path = output_dir / f"{q['id']}_{name}.wav"
                    manifest[q["id"]] = {"path": str(wav_path), "voice": name}

                    if wav_path.exists() and not args.force:
                        print(f"  skip (exists): {q['id']} [{name}]")
                        continue

                    print(f"  generating: {q['id']} [{name}] — {q['text'][:60]}")
                    synthesize_to_wav(_get_tts(voice_id, tts_cache), q["text"], wav_path)
                    generated += 1
                    print(f"    saved: {wav_path} ({wav_path.stat().st_size:,} bytes)")
        else:
            for q in suite.get("questions", []):
                total += 1
                name, voice_id = VOICES[voice_idx % len(VOICES)]
                voice_idx += 1
                wav_path = output_dir / f"{q['id']}_{name}.wav"
                manifest[q["id"]] = {"path": str(wav_path), "voice": name}

                if wav_path.exists() and not args.force:
                    print(f"  skip (exists): {q['id']} [{name}]")
                    continue

                print(f"  generating: {q['id']} [{name}] — {q['text'][:60]}")
                synthesize_to_wav(_get_tts(voice_id, tts_cache), q["text"], wav_path)
                generated += 1
                print(f"    saved: {wav_path} ({wav_path.stat().st_size:,} bytes)")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nDone: {generated} generated, {total - generated} skipped")
    print(f"Manifest: {manifest_path}")

    # --- 꼬리 무음 트림 + RMS 정규화 패스 (생성 경로와 무관하게 디렉토리 전체) ---
    # 트림을 먼저 — 정규화 RMS가 무음 꼬리를 제외한 실제 발화 기준으로 계산되도록.
    wav_files = sorted(output_dir.glob("*.wav"))
    if not args.no_trim:
        print(f"\nTrimming trailing silence → keep {_TRIM_KEEP_SEC}s after last audible sample")
        trimmed = 0
        for wav_path in wav_files:
            action = trim_trailing_silence(wav_path)
            if action != "ok":
                print(f"  {wav_path.name}: {action}")
                if not action.startswith("skip"):
                    trimmed += 1
        print(f"Trimmed: {trimmed}/{len(wav_files)}")

    if not args.no_normalize:
        print(f"\nNormalizing → target RMS {args.target_rms} (peak ≤ {_PEAK_CEILING})")
        adjusted = 0
        for wav_path in wav_files:
            action = normalize_wav(wav_path, args.target_rms)
            if action not in ("ok",) and not action.startswith("skip"):
                adjusted += 1
                print(f"  {wav_path.name}: {action}")
            elif action.startswith("skip"):
                print(f"  {wav_path.name}: {action}")
        print(f"Normalized: {adjusted}/{len(wav_files)} adjusted")


if __name__ == "__main__":
    main()
