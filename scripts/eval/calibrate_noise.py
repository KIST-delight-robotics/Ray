"""Calibrate the noise-bed playback levels to target SNRs at the mic.

The acoustic bed's SNR is the *effective* ratio at the mic, not a digital
constant, so it must be measured once for the rig. Procedure:

  1. Measure the room floor (nothing playing) — context only.
  2. Play a reference question WAV through the speaker; the voiced-region RMS of
     the mic capture is the speech level ``S``.
  3. Play the bed master (gain 1.0) through the speaker; the mic-capture RMS is
     ``N_ref``.
  4. For each target SNR ``T``: the mic noise RMS must be ``S / 10^(T/20)``, and
     since mic noise scales linearly with the digital gain, the playback gain is
     ``(S / 10^(T/20)) / N_ref``. The gain is **recorded** in calibration.json
     against the single ``bed_master.wav``; NoiseBed scales the master by it at
     playback. (No per-condition WAVs — a level tweak is a one-number edit.)

If a gain would clip the bed (peak > ceiling), the target can't be reached at
the current speaker volume — the script caps the gain, reports the achievable
SNR, and tells you to raise the physical speaker/amp volume and re-run.

Reuses the eval's real mic path (AudioInput) and plays through the same dmix
device the run will use. Run on the rig with speaker + mic live.

Usage:
    uv run python scripts/eval/calibrate_noise.py                 # reference auto-picked from manifest
    uv run python scripts/eval/calibrate_noise.py --targets medium=12,loud=4
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import logging
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Suppress ALSA/JACK chatter during PyAudio init (mirrors run.py).
_handler = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)(
    lambda *_: None
)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_handler)
except Exception:
    _asound = None

from bed_audio import _read_wav_mono, _voiced_rms
from noise_bed import NoiseBed

from voice_pipeline.audio.audio_input import AudioInput

_PEAK_CEILING = 0.95
_SETTLE_SEC = 0.5
_NOISE_CAPTURE_SEC = 5.0
_FLOOR_CAPTURE_SEC = 2.0
# 측정 중 큐가 잠깐 밀려 프레임이 드롭돼도 RMS(전력 평균)는 안 흔들린다 —
# 넉넉한 큐로 드롭(=경고 플러드)을 없애 측정 중 터미널을 깨끗하게 둔다.
_CAPTURE_QUEUE_SIZE = 100000


def _drain(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _capture(q: queue.Queue, seconds: float) -> np.ndarray:
    buf = bytearray()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with contextlib.suppress(queue.Empty):
            buf.extend(q.get(timeout=0.5))
    return np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def _measure_speech(q: queue.Queue, device: str, reference: str) -> float:
    ref, sr = _read_wav_mono(reference)
    dur = len(ref) / sr
    _drain(q)
    player = threading.Thread(target=lambda: subprocess.run(["aplay", "-q", "-D", device, reference], check=False))
    player.start()
    cap = _capture(q, dur + 0.3)
    player.join()
    return _voiced_rms(cap)


def _measure_noise(q: queue.Queue, device: str, bed_master: str) -> float:
    bed = NoiseBed(device, bed_master, {"ref": 1.0})  # measure the master at gain 1.0
    bed.set_level("ref")
    time.sleep(_SETTLE_SEC)
    _drain(q)
    cap = _capture(q, _NOISE_CAPTURE_SEC)
    bed.stop()
    return _rms(cap)


def solve_gain(
    s: float, n_ref: float, floor: float, target_snr: float, max_safe_gain: float
) -> tuple[float, float, str]:
    """Bed gain so speech-vs-(bed+floor) SNR hits ``target_snr`` at the mic.

    ``n_ref`` (bed at gain 1.0) already includes the room ``floor``; the floor is
    constant and does not scale with gain, so we separate the bed-only level and
    target the *total* noise. Returns ``(applied_gain, achievable_total_snr_db,
    status)`` where status is ``""`` (ok), ``"unreachable"`` (target too noisy →
    bed would clip), or ``"floor_limited"`` (target quieter than the floor allows).
    """
    b1 = float(np.sqrt(max(n_ref**2 - floor**2, 0.0)))  # bed-only mic RMS @ gain 1.0
    desired_total = s / (10.0 ** (target_snr / 20.0))  # total noise RMS for target SNR
    bed_needed_sq = desired_total**2 - floor**2
    status = ""
    if bed_needed_sq <= 0 or b1 <= 0:
        applied, status = 0.0, "floor_limited"  # floor alone already noisier than target
    else:
        gain = float(np.sqrt(bed_needed_sq)) / b1
        if gain > max_safe_gain:
            applied, status = max_safe_gain, "unreachable"
        else:
            applied = gain
    total = float(np.sqrt((b1 * applied) ** 2 + floor**2))
    achievable = float(20.0 * np.log10(s / total)) if total > 0 else float("inf")
    return applied, achievable, status


def _resolve_reference(manifest_path: str) -> str:
    """Pick a speech-level reference WAV from the prepare_audio manifest.

    Prefers the first ASR question (any normalized question works — all share the
    same target RMS); falls back to the first entry. Resolving from the manifest
    avoids hardcoding a voice-specific filename that goes stale when voices change.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise SystemExit(f"No --reference given and manifest not found: {path} (run prepare_audio.py first)")
    manifest = json.loads(path.read_text())
    if not manifest:
        raise SystemExit(f"Manifest is empty: {path}")
    asr_ids = sorted(k for k in manifest if k.startswith("asr_"))
    key = asr_ids[0] if asr_ids else sorted(manifest)[0]
    return manifest[key]["path"]


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate noise-bed levels to target SNRs")
    # dmix를 plug로 감싸 동시 재생(믹싱)+자동 리샘플. 중첩 PCM은 작은따옴표 필수 —
    # 'plug:dmix:CARD=DAC'는 ALSA 인자 파싱이 깨짐.
    p.add_argument("--device", default="plug:'dmix:CARD=DAC,DEV=0'", help="dmix playback device")
    p.add_argument(
        "--reference",
        default=None,
        help="Clean question WAV for the speech-level probe (default: first ASR WAV in --manifest)",
    )
    p.add_argument("--manifest", default="data/eval/wav/manifest.json", help="Used to auto-pick --reference")
    p.add_argument("--bed-master", default="data/eval/noise_bed/bed_master.wav")
    p.add_argument("--targets", default="medium=15,loud=7", help="cond=snr_db,cond=snr_db")
    p.add_argument("--out-dir", default="data/eval/noise_bed")
    args = p.parse_args()

    if args.reference is None:
        args.reference = _resolve_reference(args.manifest)
        print(f"Reference (auto from manifest): {args.reference}")

    targets: dict[str, float] = {}
    for part in args.targets.split(","):
        name, val = part.split("=")
        targets[name.strip()] = float(val)

    logging.getLogger("voice_pipeline.audio").setLevel(logging.ERROR)
    audio_queue: queue.Queue = queue.Queue(maxsize=_CAPTURE_QUEUE_SIZE)
    audio_input = AudioInput(audio_queue)
    audio_input.start()
    try:
        audio_queue.get(timeout=10.0)  # wait for capture stream to open
    except queue.Empty:
        if audio_input.error is not None:
            raise audio_input.error from None
    if _asound is not None:
        _asound.snd_lib_error_set_handler(None)

    try:
        _drain(audio_queue)
        floor = _capture(audio_queue, _FLOOR_CAPTURE_SEC)
        room_floor = _rms(floor)
        print(f"Room floor RMS: {room_floor:.5f}")

        s = _measure_speech(audio_queue, args.device, args.reference)
        print(f"Speech level S (voiced, mic): {s:.5f}")

        n_ref = _measure_noise(audio_queue, args.device, args.bed_master)
        print(f"Bed N_ref (gain 1.0, mic): {n_ref:.5f}")
    finally:
        audio_input.stop()

    if s <= 0 or n_ref <= 0:
        raise SystemExit("Measurement failed (S or N_ref is zero) — check speaker/mic.")

    master, _ = _read_wav_mono(args.bed_master)
    master_peak = float(np.abs(master).max()) or 1.0
    max_safe_gain = _PEAK_CEILING / master_peak

    floor_ceiling = float(20.0 * np.log10(s / room_floor)) if room_floor > 0 else float("inf")
    print(f"Floor-limited SNR ceiling (bed off): {floor_ceiling:.1f} dB")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib: dict = {
        "device": args.device,
        "reference": args.reference,
        "bed_master": args.bed_master,
        "room_floor_rms": round(room_floor, 6),
        "speech_rms": round(s, 6),
        "bed_n_ref_rms": round(n_ref, 6),
        "floor_snr_ceiling_db": round(floor_ceiling, 2),
        "conditions": {},
    }

    flags = {
        "unreachable": "  ⚠ UNREACHABLE — regenerate bed master with higher --master-rms",
        "floor_limited": "  ⚠ FLOOR-LIMITED — room floor too high; lower mic gain or relax target",
    }
    # No per-condition WAVs are written: the gain is recorded against the single
    # master, and NoiseBed scales it at playback. Changing a level is then a
    # one-number edit in calibration.json — no re-render, no overwritten files.
    print("\nCondition          target  gain   achievable")
    for name, target_snr in targets.items():
        applied, achievable, status = solve_gain(s, n_ref, room_floor, target_snr, max_safe_gain)
        row = f"  {name:<12} {target_snr:>5.1f}dB {applied:>6.3f} {achievable:>9.1f}dB"
        print(row + flags.get(status, ""))
        calib["conditions"][name] = {
            "target_snr_db": target_snr,
            "gain": round(applied, 6),
            "achievable_snr_db": round(achievable, 2),
            "status": status or "ok",
        }

    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2, ensure_ascii=False))
    print(f"\nCalibration → {out_dir / 'calibration.json'}")
    statuses = {c["status"] for c in calib["conditions"].values()}
    if "unreachable" in statuses:
        # Speech and bed share one speaker, so physical volume scales both — it
        # can't lower the SNR floor. The bed must be made denser instead.
        print("Some targets too noisy to reach: regenerate a denser bed master (higher --master-rms).")
    if "floor_limited" in statuses:
        print("Some targets quieter than the room floor allows: lower mic gain or relax the target.")


if __name__ == "__main__":
    main()
