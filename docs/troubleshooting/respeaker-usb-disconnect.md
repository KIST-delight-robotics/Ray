# ReSpeaker XVF3800 USB 연결 끊김 (재열거 폭주)

## Summary

ReSpeaker Flex XVF3800(USB 마이크)이 운용 중 USB 버스에서 반복적으로
disconnect/재열거(re-enumerate)되며, 마이크 캡처가 멈춰 ASR이 동작하지 않는 현상.
약 6/08부터 누적·악화돼 왔고, 6/12 eval 장시간 실행 중 11분간 지속 발생해 ASR이 완전히 끊겼다.
**`docs/troubleshooting/dac-i2s-issue.md`의 PCM5122 I2S DAC 이슈와는 별개**이며(아래 구분 참조),
**ReSpeaker 경로(허브 경유)의 물리 USB 연결 문제**로 판단된다.

- **최초 확인 장애**: 2026-06-12 (run `data/eval/results/20260612_164141`)
- **조치**: 2026-06-15 13:45 — 허브 제거, Pi USB 포트에 **직결**로 전환 (아래 Remediation)
- **상태**: **종결 (2026-07-08 점검)** — 직결 전환 후 3주간 재발 없음. 허브가 원인으로 확정 (아래 점검 결과)

## Environment

- **Board**: Raspberry Pi 5, kernel 6.8.0-1057-raspi
- **Mic**: Seeed reSpeaker Flex XVF3800 C16K6Ch (USB `2886:001e`, 6ch, high-speed 480M)
- **장애 시 연결**: Huasheng USB2.0 HUB (`214b:7250`) 경유
  - 허브 = USB 경로 `2-1`, 같은 허브에 Logitech Unifying Receiver(`046d:c52b`) = `2-1.1`
  - ReSpeaker = 허브 포트 2(`2-1.2`) 또는 포트 4(`2-1.4`)
- **캡처 경로**: `voice_pipeline/audio/audio_input.py` (PyAudio, `_DEVICE_INDEX=1`, 6ch)

## Symptom

- eval/파이프라인 실행 중 ASR이 도중에 **아무 것도 인식하지 못함**.
- SessionLoop 로그상 `Audio starvation (5.0s without frames) — terminating session` 반복.
- 마이크 캡처 스레드는 **예외 없이(에러 로그 없이) 그대로 멈춤** — `stream.read()`가
  사라진 장치 핸들에서 블로킹.
- **질문 재생 스피커(DAC, `plughw:0,0`)는 정상 동작** — 입력 장치만 죽음.
- **프로그램만 재시작하면(재부팅 없이) 즉시 복구** → 재열거된 장치를 새로 open하기 때문.

## Root Cause

시스템 저널(`journalctl -k`)에서 ReSpeaker(`2-1.2`)가 반복적으로
`USB disconnect` → `new high-speed USB device` 재열거되는 것이 확인됨.
열거 단계 오류 코드가 동반됨:

- `usb_set_interface failed (-19)` (ENODEV)
- `device descriptor read/64, error -71` / `device not accepting address, error -71` (EPROTO, 신호/프로토콜)
- `(-32)` (EPIPE)

**수동 탈착과 구분**: 커널은 사람이 뽑은 것과 전기적 고장을 동일하게
`USB disconnect`로 기록한다. 그러나 다음은 수동으로 불가능 → 전기적 고장으로 확정:
- 수 초 내 연속 재열거(burst, 예: 8초에 4회)
- `-71`/`set_interface failed` 등 열거 단계 에러 동반
- 사람 개입 없이 자동 재연결

### 6/12 ASR 장애 (run 20260612_164141)

- 마지막 정상 질문 `tt_051`(17:06:11) → **첫 실패 `tt_052`**(17:06:13)부터 ASR 0건.
- 17:06:10 `2-1.2` USB disconnect 시작, device number 5→6→…→15로 17:06~17:17 지속 플래핑
  (8초에 4회 재열거 + `set_interface failed -19/-71`).
- eval에는 복구 로직 없음: `scripts/eval/run.py`는 `audio_input.start()`를 1회만 호출하고
  `audio_input.error`도 시작 후 10초 내 1회만 점검 → 한 번 hang되면 이후 전 질문 연쇄 실패.

## 같은 문제의 과거 이력 (burst/에러 동반 = 고장)

| 날짜·시각 | 패턴 | 에러 |
|---|---|---|
| ~5/20–6/05 | 단발 disconnect (수 시간 간격) | 없음 — 수동/산발, 고장 아님 |
| 6/08 10:49:52–10:50:11 | 19초에 4회 | -19 |
| 6/09 09:31:05–:16 / 13:34:14–:37 | 11초·23초에 각 4회 | — |
| 6/11 17:15:09–:19 | **10초에 6회** | -19 ×3 |
| 6/12 13:36 / 13:50 / 17:06 / 17:19 | 5초3회 / 24초8회 / eval 11분 지속 / burst | -19, **-71**, -32 |
| 6/15 10:07 / 10:22 | ~1분 ~6회 / 18초 4회 | -19 ×2 |

→ **6/08부터 반복·악화되는 만성·진행성 고장.** 에러도 초기 -19에서 6/12엔 -71/-32까지 확장.

## 배제된 가설

- **PCM5122 I2S DAC 이슈(`docs/troubleshooting/dac-i2s-issue.md`) 아님**:
  - 그 이슈는 DAC **출력**이 죽고 **재부팅만이 복구 수단**. 본 건은 **입력(마이크)**만 죽고
    **프로그램 재시작으로 복구**됨.
  - 6/12 하루 PipeWire `resync` 에러 **0건**(I2S 이슈의 시그니처 부재).
- **허브 전체 전원 붕괴(brownout) 아님**:
  - 6/12 17:06~17:17 ReSpeaker만 끊기는 동안 같은 허브 Logitech(`2-1.1`)는 **무사**.
  - 커널에 글로벌 undervoltage/over-current/throttle 이벤트 **0건**.
- **단일 허브 포트 불량 아님**:
  - 6/15 허브 포트 2(`2-1.2`)·포트 4(`2-1.4`) **양쪽 모두** `-71` 실패.
  - → 특정 포트가 아니라 **"허브를 경유하는 경로" 자체**가 문제.

## Remediation (2026-06-15)

- 13:44 허브 포트들(`2-1.2`/`2-1.4`)에서 `device not accepting address -71`로 **열거 자체 실패**.
- 13:45:10 **Huasheng 허브를 빼고 Pi USB 포트에 직결** → 경로 `4-1`(bus 4 root hub),
  **에러 0건으로 1회 깔끔 열거**, devnum=3 유지.
- 직결 직후 8분 이상 disconnect/에러 0건 (허브에선 수 초~수 분 내 재발하던 것과 대조).

### 관찰 메모

- ESP32-S3 LED가 "원래(허브) 포트에서 점멸, 다른 포트에선 비점멸"이라는 보고가 있었으나,
  로그상 점멸하던 포트가 곧 실패 포트였음 → 점멸은 정상 신호가 아니라 **반복 리셋/재열거 신호**일
  가능성. (LED 정확한 의미는 미확인. lsusb에 Espressif `303a` 미존재.)
- **미확인**: 직결 전환 시 USB 케이블도 교체했는지 여부. 동일 케이블이었다면 허브가 범인으로 거의 확정,
  케이블도 바꿨다면 허브/케이블 둘 다 후보.

## 점검 결과 (2026-07-08) — 재발 없음, 종결

직결 전환 후 ReSpeaker는 6/17 10:33부터 `2-2`(bus 2 root port)로 이동해 운용 중.
6/17 이후 약 3주간 `2-2` 경로 점검 결과:

- **열거 단계 에러 0건** — `-71`/`-19`/`-32`, `not accepting address`, `descriptor read` 모두 없음
- **burst 재열거 0건** — 고장 시그니처(수 초 내 연속 재열거) 부재
- 단발 disconnect 4건은 모두 수동 탈착 패턴 (재연결까지 5시간/21분 공백 또는 작업 중 단발):
  6/25 10:45→15:57, 6/26 11:31, 7/7 15:44→16:05, 7/7 16:08
- **허브는 여전히 문제**: 허브(`2-1`)는 Logitech 리시버(`2-1.1`)용으로 계속 사용 중인데,
  6/24 13:12·6/29 13:33에도 `not accepting address -22`, `can't read configurations -71` 등
  에러 지속 → ReSpeaker는 허브를 떠난 뒤 무증상, 허브는 다른 장치와도 에러 → **허브 불량 확정**

## 향후 확인 방법 (재발 시)

같은 고장이 재발하는지 점검 (`-k`는 `-b`를 암시해 현재 부팅만 보므로,
여러 부팅에 걸친 기간은 `_TRANSPORT=kernel`로 조회):

```bash
# 1) 특정 시점 이후 USB disconnect / 열거 에러 (정상이면 출력 없음)
journalctl _TRANSPORT=kernel --since "2026-06-15 13:45" --no-pager \
  -g "USB disconnect|usb_set_interface failed|error -71|not accepting address|descriptor read"

# 2) ReSpeaker 재열거 횟수 (부팅/재플러그 1회당 1건이면 정상; 많으면 churn = 재발)
journalctl _TRANSPORT=kernel --since "2026-06-15" --no-pager -g "reSpeaker Flex XVF3800" | grep -c "Product:"

# 3) 현재 churn 상태 — devnum이 낮게 유지되면 양호 (경로는 바뀔 수 있어 product로 탐색)
for d in /sys/bus/usb/devices/*/; do \
  grep -qi reSpeaker "$d/product" 2>/dev/null && \
  echo "path=$(basename "$d") busnum=$(cat "$d/busnum") devnum=$(cat "$d/devnum")"; done
```

- `--since` 날짜를 바꿔 가며(예: 다음 점검일) 그 이후 구간만 보면 됨.
- 포트를 또 옮기면 경로(`4-1` 등)가 바뀌므로, 위 1·2번은 포트 경로 대신 제품명/에러 패턴으로 매칭함.
- 재발 시: 케이블 교체 → 다른 Pi 직결 포트 → 그래도 재발 시 ReSpeaker 장치 불량 의심 순으로 좁힌다.
