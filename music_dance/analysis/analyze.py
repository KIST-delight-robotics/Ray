#!/usr/bin/env python3
"""음악 → 댄스 타임라인 분석기 (HPSS 기반).

WAV 한 곡을 받아 (1) LED 밝기 엔벨로프와 (2) 모터 모션 신호를 계산해
고정 프레임레이트(fps) CSV 타임라인으로 저장한다.
C++ 모션 제어부(`motion/main.cpp`)가 이 CSV를 읽어 재생에 맞춰
LED(라즈베리파이 PWM)와 모터(Dynamixel ID6)를 구동한다.

설계 근거: docs/decisions-wip.md "음악 댄스 모드" 참조.

LED 밝기:
    하모닉 글로우(느린 attack/release) + 퍼커시브 펀치(fast attack / slow release)
    를 블렌드 → dB 스케일 → 퍼센타일 정규화 → 비대칭 스무딩 → 감마 → 바닥값.
    사람은 음량을 로그(dB)로, 밝기를 비선형(감마)으로 지각하므로 둘을 보정해
    "들리는 크기 ≈ 보이는 밝기"를 맞춘다.

모터 모션:
    퍼커시브 엔벨로프를 서보가 따라갈 수 있을 만큼 평활화 → 비트(드럼)에 맞춰
    한 축이 끄덕이듯 움직인다.

오프라인(곡 전체) 분석이므로 lookahead가 필요한 HPSS를 자유롭게 쓸 수 있다.
"""

from __future__ import annotations

import argparse
import sys

import librosa
import numpy as np


def envelope_follower(x: np.ndarray, dt: float, attack_s: float, release_s: float) -> np.ndarray:
    """1차(one-pole) 비대칭 엔벨로프 추종기.

    값이 상승할 때는 attack 시정수, 하강할 때는 release 시정수를 사용한다.
    attack을 짧게 / release를 길게 두면 타격 순간 빠르게 차오르고 천천히 잦아드는,
    실제 타악기 엔벨로프에 가까운 "반응하는" 느낌이 난다.
    """
    a_coef = np.exp(-dt / max(attack_s, 1e-6))
    r_coef = np.exp(-dt / max(release_s, 1e-6))
    y = np.empty_like(x)
    prev = float(x[0]) if len(x) else 0.0
    for i, v in enumerate(x):
        coef = a_coef if v > prev else r_coef
        prev = coef * prev + (1.0 - coef) * float(v)
        y[i] = prev
    return y


def normalize_percentile(x: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    """퍼센타일 기준 [0,1] 정규화. 곡별 음량 편차/이상치에 강건."""
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def band_energy_db(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """프레임별 RMS 에너지를 dB로 변환."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return 20.0 * np.log10(rms + 1e-7)


def analyze(
    wav_path: str,
    fps: float,
    sr: int,
    floor: float,
    gamma: float,
    w_harmonic: float,
    w_percussive: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """WAV → (실제 fps, led[0..1], motor[0..1])."""
    y, sr = librosa.load(wav_path, sr=sr, mono=True)

    hop = max(1, round(sr / fps))
    frame_length = hop * 2
    actual_fps = sr / hop
    dt = 1.0 / actual_fps

    # HPSS: 하모닉(지속음) / 퍼커시브(트랜지언트) 분리
    y_harm, y_perc = librosa.effects.hpss(y)

    h = normalize_percentile(band_energy_db(y_harm, frame_length, hop))
    p = normalize_percentile(band_energy_db(y_perc, frame_length, hop))

    n = min(len(h), len(p))
    h, p = h[:n], p[:n]

    # 비대칭 스무딩: 하모닉은 느린 글로우, 퍼커시브는 빠른 펀치
    h_env = envelope_follower(h, dt, attack_s=0.15, release_s=0.40)
    p_env = envelope_follower(p, dt, attack_s=0.010, release_s=0.18)

    # LED: 블렌드 → 최종 퍼센타일 정규화(대비 확보) → 감마 → 바닥값.
    # 최종 정규화를 두는 이유: MR/마스터링된 트랙은 음압이 높고 고르기 때문에
    # 성분별 정규화만으로는 곡 전체가 최대치에 붙어(saturation) 디밍이 안 보인다.
    # 블렌드 곡선 자체를 곡 전체 기준으로 다시 펴 줘야 번쩍임이 또렷해진다.
    v = w_harmonic * h_env + w_percussive * p_env
    v = normalize_percentile(v, 5.0, 98.0)
    led = floor + (1.0 - floor) * np.power(v, gamma)

    # 모터: 퍼커시브를 서보가 따라올 만큼 평활화. release 를 짧게 둬서 비트 사이엔
    # 홈 쪽으로 돌아오고 강한 타격에 크게 끄덕이도록 대비를 준다.
    motor = envelope_follower(p, dt, attack_s=0.04, release_s=0.22)
    motor = normalize_percentile(motor, 5.0, 98.0)

    return actual_fps, led.astype(np.float32), motor.astype(np.float32)


def write_timeline(path: str, fps: float, wav_path: str, led: np.ndarray, motor: np.ndarray) -> None:
    n = len(led)
    with open(path, "w") as f:
        f.write("# music_dance timeline v1\n")
        f.write(f"# fps={fps:.6f}\n")
        f.write(f"# n={n}\n")
        f.write(f"# duration_sec={n / fps:.3f}\n")
        f.write(f"# wav={wav_path}\n")
        f.write("led,motor\n")
        for i in range(n):
            f.write(f"{led[i]:.6f},{motor[i]:.6f}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="음악 → 댄스 타임라인(LED 밝기 + 모터 모션) 분석기")
    ap.add_argument("wav", help="입력 WAV 경로")
    ap.add_argument("-o", "--out", default="timeline.csv", help="출력 CSV 경로 (기본 timeline.csv)")
    ap.add_argument("--fps", type=float, default=100.0, help="타임라인 프레임레이트 (기본 100)")
    ap.add_argument("--sr", type=int, default=22050, help="분석 샘플레이트 (기본 22050)")
    ap.add_argument("--floor", type=float, default=0.12, help="LED 밝기 바닥값 [0..1] (기본 0.12)")
    ap.add_argument("--gamma", type=float, default=2.2, help="감마 보정 (기본 2.2)")
    ap.add_argument("--w-harmonic", type=float, default=0.30, help="하모닉 글로우 가중 (기본 0.30)")
    ap.add_argument("--w-percussive", type=float, default=0.95, help="퍼커시브 펀치 가중 (기본 0.95)")
    args = ap.parse_args()

    print(f"[analyze] 로딩/분석: {args.wav} (sr={args.sr}, fps={args.fps})", file=sys.stderr)
    actual_fps, led, motor = analyze(
        args.wav, args.fps, args.sr, args.floor, args.gamma, args.w_harmonic, args.w_percussive
    )
    write_timeline(args.out, actual_fps, args.wav, led, motor)
    print(
        f"[analyze] 완료: {args.out}  frames={len(led)}  fps={actual_fps:.2f}  "
        f"dur={len(led) / actual_fps:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
