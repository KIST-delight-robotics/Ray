#!/usr/bin/env bash
# 온라인 입 궤적 점검용 로그 캡처.
# C++(./build/Ray)를 한 번만 띄우고 각 WAV를 재생, 재생마다 새로 생성된
# pos4_audio CSV를 예문 이름으로 라벨링해 logs/mouth_inspection/ 에 모은다.
#
# 사용법: bash scripts/hardware/capture_mouth_logs.sh [wav ...]
#   인자 없으면 output/tts_*.wav 전부.

set -uo pipefail

if [ "$#" -gt 0 ]; then FILES=("$@"); else FILES=(output/tts_*.wav); fi

OUTDIR=logs/mouth_inspection
mkdir -p "$OUTDIR"

echo "Starting C++ (./build/Ray)..."
./build/Ray > "$OUTDIR/cpp_stdout.log" 2>&1 &
CPP_PID=$!
cleanup() { echo "Stopping C++ (PID $CPP_PID)..."; kill -INT "$CPP_PID" 2>/dev/null; wait "$CPP_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM
sleep 4

for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "skip (not found): $f"; continue; }
  name=$(basename "$f" .wav)
  echo ">>> $name"
  before=$(ls -1 logs/pos4_audio/*.csv 2>/dev/null | wc -l)
  uv run python scripts/hardware/play_wav_motion.py "$f" > "$OUTDIR/${name}_play.log" 2>&1
  # 재생으로 새로 생긴 가장 최신 pos4_audio CSV를 라벨 복사
  newest=$(ls -t logs/pos4_audio/*.csv 2>/dev/null | head -1)
  after=$(ls -1 logs/pos4_audio/*.csv 2>/dev/null | wc -l)
  if [ "$after" -gt "$before" ] && [ -n "$newest" ]; then
    cp "$newest" "$OUTDIR/${name}.csv"
    rows=$(wc -l < "$OUTDIR/${name}.csv")
    echo "    captured $rows rows -> $OUTDIR/${name}.csv"
  else
    echo "    WARN: no new pos4_audio CSV detected for $name"
  fi
done

echo "캡처 완료: $OUTDIR/"
ls -1 "$OUTDIR"/*.csv 2>/dev/null