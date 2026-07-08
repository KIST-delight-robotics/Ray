#!/usr/bin/env bash
# 온라인 입 궤적 평가용 로그 캡처 (기존 파이프라인 무수정, 재생 유틸만 재사용).
# ./build/Ray 를 한 번 띄우고 eval/mouth/wavs/*.wav 를 순차 재생,
# 재생마다 새로 생긴 pos4_audio CSV를 eval/mouth/logs/<name>.csv 로 라벨링.
#
# 사용법: bash eval/mouth/capture.sh [wav ...]

set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

EVAL=eval/mouth
LOGDIR="$EVAL/logs"
mkdir -p "$LOGDIR"

if [ "$#" -gt 0 ]; then FILES=("$@"); else FILES=("$EVAL"/wavs/*.wav); fi

echo "Starting C++ (./build/Ray)..."
./build/Ray > "$LOGDIR/_cpp_stdout.log" 2>&1 &
CPP_PID=$!
cleanup() { echo "Stopping C++ (PID $CPP_PID)..."; kill -INT "$CPP_PID" 2>/dev/null; wait "$CPP_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM
sleep 4

for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "skip: $f"; continue; }
  name=$(basename "$f" .wav)
  echo ">>> $name"
  before=$(ls -1 logs/pos4_audio/*.csv 2>/dev/null | wc -l)
  uv run python scripts/hardware/play_wav_motion.py "$f" > "$LOGDIR/${name}_play.log" 2>&1
  newest=$(ls -t logs/pos4_audio/*.csv 2>/dev/null | head -1)
  after=$(ls -1 logs/pos4_audio/*.csv 2>/dev/null | wc -l)
  if [ "$after" -gt "$before" ] && [ -n "$newest" ]; then
    cp "$newest" "$LOGDIR/${name}.csv"
    echo "    captured $(wc -l < "$LOGDIR/${name}.csv") rows"
  else
    echo "    WARN: no new pos4_audio CSV for $name"
  fi
done
echo "캡처 완료: $LOGDIR/"
ls -1 "$LOGDIR"/*.csv 2>/dev/null