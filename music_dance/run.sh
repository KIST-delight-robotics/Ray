#!/usr/bin/env bash
# music_dance 전체 실행: 분석(Python) → 빌드(C++) → 재생+모터+LED 동기 구동
#
# 사용법:
#   ./run.sh [WAV경로]
# 인자 없으면 저장소 루트의 V_ZionT_MR.wav 를 사용.
#
# 모터(/dev/ttyUSB0)와 LED(하드웨어 PWM)를 함께 쓴다.
# PWM sysfs export 는 root 권한이 필요하므로 한 번 sudo 로 채널을 열고 권한을
# 사용자에게 넘긴 뒤, 본체(dance)는 일반 사용자로 실행한다(오디오·시리얼 세션 유지).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

WAV="${1:-$ROOT/V_ZionT_MR.wav}"
TIMELINE="$HERE/timeline.csv"

# 하드웨어 파라미터 (필요 시 여기서 조정)
PORT="/dev/ttyUSB0"
BAUD=2000000
MOTOR_ID=6
MOTOR_HOME=100
MOTOR_AMP=300
PWMCHIP=0
PWMCHAN=1            # "PWM1"

if [[ ! -f "$WAV" ]]; then
  echo "WAV 없음: $WAV" >&2
  exit 1
fi

echo "==> 1/3 음향 분석 (HPSS → 타임라인)"
cd "$ROOT"
uv run --with librosa --with soundfile python "$HERE/analysis/analyze.py" "$WAV" \
  -o "$TIMELINE" --fps 100

echo "==> 2/3 C++ 모션 제어부 빌드"
cmake -S "$HERE/motion" -B "$HERE/motion/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$HERE/motion/build" -j

echo "==> PWM 채널 준비 (sudo 1회: export + 권한 이양)"
sudo sh -c "
  base=/sys/class/pwm/pwmchip$PWMCHIP
  [ -e \$base/pwm$PWMCHAN ] || echo $PWMCHAN > \$base/export
  sleep 0.2
  chown -R $(id -un) \$base/pwm$PWMCHAN
  chmod -R u+rw \$base/pwm$PWMCHAN
" || echo "  (PWM 준비 실패 — LED 없이 진행될 수 있음)"

echo "==> 3/3 재생 + 모터 + LED 동기 구동 (Ctrl-C 중단)"
"$HERE/motion/build/dance" \
  --timeline "$TIMELINE" --wav "$WAV" \
  --port "$PORT" --baud "$BAUD" --id "$MOTOR_ID" \
  --motor-home "$MOTOR_HOME" --motor-amp "$MOTOR_AMP" \
  --pwmchip "$PWMCHIP" --pwmchan "$PWMCHAN"
