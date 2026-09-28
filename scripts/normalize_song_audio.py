"""노래 WAV 의 재생 레벨을 목표치(``settings.MUSIC_TARGET_PEAK``)로 맞춘다.

    uv run python scripts/normalize_song_audio.py --dry-run   # 계획만 출력
    uv run python scripts/normalize_song_audio.py             # 적용 (원본은 백업)
    uv run python scripts/normalize_song_audio.py --restore   # 백업에서 되돌리기

**반드시 모션 CSV(headMotion/mouthMotion/ledMotion) 생성을 끝낸 뒤에 돌릴 것.** 궤적은 오디오
레벨에서 뽑으므로, 줄인 음원으로 뽑으면 입이 작아진다. 순서는 항상 "궤적 생성 → 정규화" 다.

노래는 차임·인사와 달리 재생 시점에 맞추지 않는다. Python 은 C++ 에 경로가 아니라 ``audio_name`` 만
보내고(``play_audio_csv``), C++ 은 ``sf::Music`` 으로 ``assets/audio/music/<name>.wav`` 를 직접
스트리밍하므로 정규화 사본을 끼워 넣을 자리가 없다. 곡 추가는 어차피 CSV 준비가 필요한 수동 절차라
음원 자체를 한 번 맞춰 두는 편이 런타임 게인(프로토콜 + C++ + 캐시)보다 싸다.

척도·클램프는 :mod:`voice_pipeline.audio_level` 과 공유한다 — 통계는 말소리와 같은 40ms 프레임 피크
중앙값이고 목표값만 다르다. 이미 목표 근처인 곡은 건너뛰므로 여러 번 돌려도 안전하다.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_pipeline.audio_level import gain_for, measure  # noqa: E402
from voice_pipeline.settings import ASSETS_DIR, MUSIC_TARGET_PEAK, VAR_DIR  # noqa: E402

MUSIC_DIR = f"{ASSETS_DIR}/audio/music"  # cpp/main.cpp MUSIC_DIR 과 같은 경로
BACKUP_DIR = f"{VAR_DIR}/audio/music_original"  # 원본 보관. var/ 는 gitignore 라 저장소에 안 실린다
# 목표와 이만큼 안이면 손대지 않는다. 감쇠 뒤 재측정은 목표보다 최대 1 dB 쯤 높게 나오므로
# (audio_level.measure 참고 — 절대 무음 기준이라 작은 프레임이 더 잘린다) 그보다 넓게 잡아야
# 재실행 때마다 같은 곡을 다시 양자화하지 않는다. 1 dB 차이는 들리지 않는다.
SKIP_WITHIN_DB = 1.0


def _db(ratio: float) -> float:
    return 20 * math.log10(ratio) if ratio > 0 else float("-inf")


def _restore(music_dir: Path, backup_dir: Path) -> int:
    if not backup_dir.is_dir():
        print(f"백업 폴더가 없습니다: {backup_dir}")
        return 1
    restored = 0
    for src in sorted(backup_dir.glob("*.wav")):
        shutil.copy2(src, music_dir / src.name)
        print(f"  되돌림  {src.name}")
        restored += 1
    print(f"\n{restored}곡을 원본으로 되돌렸습니다.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=float, default=MUSIC_TARGET_PEAK, help="목표 중앙값 (0~1)")
    parser.add_argument("--dir", default=MUSIC_DIR, help="노래 WAV 폴더")
    parser.add_argument("--backup-dir", default=BACKUP_DIR, help="원본 백업 폴더")
    parser.add_argument("--dry-run", action="store_true", help="적용하지 않고 계획만 출력")
    parser.add_argument("--restore", action="store_true", help="백업에서 원본으로 되돌리기")
    args = parser.parse_args()

    music_dir = Path(args.dir)
    backup_dir = Path(args.backup_dir)
    if args.restore:
        return _restore(music_dir, backup_dir)

    songs = sorted(music_dir.glob("*.wav"))
    if not songs:
        print(f"WAV 가 없습니다: {music_dir}")
        return 1

    print(f"목표 중앙값 {args.target:.3f} — {len(songs)}곡{' (dry-run)' if args.dry_run else ''}\n")
    print(f"{'곡':34s} {'전':>7s} {'배율':>7s} {'dB':>7s} {'후(예상)':>9s}")

    changed = 0
    for path in songs:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        level = measure(mono, sample_rate, speech=True)
        gain = gain_for(mono, sample_rate, speech=True, target=args.target)

        if abs(_db(gain)) < SKIP_WITHIN_DB:
            print(f"{path.name[:34]:34s} {level:7.3f} {'—':>7s} {'—':>7s} {'(유지)':>9s}")
            continue

        print(f"{path.name[:34]:34s} {level:7.3f} {gain:7.3f} {_db(gain):+7.1f} {level * gain:9.3f}")
        changed += 1
        if args.dry_run:
            continue

        backup = backup_dir / path.name
        if not backup.exists():  # 첫 정규화 때의 원본만 지킨다 — 두 번째 실행이 덮어쓰지 않게
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)

        tmp = path.with_suffix(".wav.tmp")  # 중간에 죽어도 원본이 반쯤 덮이지 않게
        # 확장자가 .tmp 라 형식을 추론하지 못한다 — 명시해야 한다
        sf.write(str(tmp), np.clip(data * gain, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
        tmp.replace(path)

    if args.dry_run:
        print(f"\n{changed}곡이 대상입니다. 적용하려면 --dry-run 없이 다시 실행하세요.")
    else:
        print(f"\n{changed}곡 적용 완료. 원본은 {backup_dir} — 되돌리려면 --restore.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
