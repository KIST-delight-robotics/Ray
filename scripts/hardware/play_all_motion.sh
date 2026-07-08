#!/usr/bin/env bash
# C++(./build/Ray)를 한 번만 띄우고 여러 WAV를 순서대로 로봇이 말하게 한다.
# 사용법:
#   bash scripts/hardware/play_all_motion.sh                      # output/norm_*.wav + output/pron_*.wav 전부
#   bash scripts/hardware/play_all_motion.sh output/foo.wav ...   # 지정한 파일들만

set -uo pipefail

# 재생할 파일 목록 (인자 없으면 기본 글롭)
if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  FILES=(output/norm_*.wav output/pron_*.wav)
fi

# C++ 한 번 시작
echo "Starting C++ process (./build/Ray)..."
./build/Ray &
CPP_PID=$!

# 종료 시 C++ 정리
cleanup() {
  echo "Stopping C++ process (PID $CPP_PID)..."
  kill -INT "$CPP_PID" 2>/dev/null
  wait "$CPP_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# WebSocket 서버가 뜰 때까지 대기
sleep 4

# 파일별로 순서대로 재생 (--start-cpp 없이, 이미 떠 있는 C++에 붙음)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "skip (not found): $f"; continue; }
  echo ">>> $f"
  uv run python scripts/hardware/play_wav_motion.py "$f"
done

echo "전체 재생 완료."
