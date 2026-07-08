#!/usr/bin/env bash
# ElevenLabs TTS 정규화/발음 검증용 배치 합성.
# 사용법: bash scripts/tts_normalization_check.sh
# 결과: output/*.wav

set -euo pipefail

VENDOR=elevenlabs

run() {
  echo "=== $1 ==="
  uv run python scripts/tts_to_file.py "$2" --vendor "$VENDOR" -n "$1"
}

# --- 정규화 검증 ---
run norm_01_cardinal  "I counted 1,234 birds in the park."
run norm_02_date      "The contract starts on March 3, 2026."
run norm_03_time      "Let's meet at 3:45 PM, not 9:05 AM."
run norm_04_currency  "It costs \$1,250.99 plus €50 and £20."
run norm_05_ordinal   "She finished 1st in the 21st century."
run norm_06_percent   "Our accuracy reached 99.5% this quarter."
run norm_07_phone     "Please call 555-123-4567 after noon."
run norm_08_abbr      "Dr. Smith lives on 5th Ave. near Mt. Hood."
run norm_09_acronym   "NASA and the FBI use a new API in the USA."
run norm_10_units     "It's 25°C outside, about 100 km away, 5 kg total."
run norm_11_fraction  "Add 1/2 cup of flour and 3/4 teaspoon of salt."
run norm_12_symbol    "Email john@test.com or visit example.com & reply ASAP."
run norm_13_negative  "The range was -5 to 10 degrees on Feb 28."
run norm_14_mixed     "Order #42 shipped 12/25/2025 for \$99 to Apt. 3B."

# --- 발음 검증 ---
run pron_15_read      "I read a book yesterday, and I read every day."
run pron_16_wind      "The wind was strong, so I had to wind the rope."
run pron_17_th        "The three thin thinkers thought thirty thoughts."
run pron_18_plosive   "Peter Piper picked a peck of pickled peppers."
run pron_19_vowel     "She sees the sea; the bee is in the beach."
run pron_20_prosody   "Good morning! How are you today? I hope you slept well and feel ready for our conversation."

echo
echo "완료. output/ 디렉터리에서 *.wav 확인하세요."
