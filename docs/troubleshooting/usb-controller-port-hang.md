# USB 컨트롤러 포트 멈춤 (전원은 나오는데 장치 감지 안 됨)

## Summary

운용 중 ReSpeaker가 USB에서 끊긴 뒤 다시 인식되지 않은 현상.
포트는 전원을 계속 공급했지만(ReSpeaker LED 점등, 핸드폰 충전 표시 정상)
커널은 장치를 꽂아도 아무 이벤트도 받지 못했다.
**Pi 5의 xHCI 컨트롤러(`xhci-hcd.0`)를 소프트웨어로 리셋(unbind/bind)하자
케이블·장치를 그대로 둔 채 즉시 복구** → 케이블이나 ReSpeaker가 아니라
**컨트롤러 포트 상태가 멈춘 것**으로 판단.

- **발생**: 2026-10-21 16:57:53
- **복구**: 2026-10-21 17:20:55 (컨트롤러 리셋)
- **상태**: 복구됨. 촉발 원인은 미상 (아래 미확인 사항)

## Environment

- **Board**: Raspberry Pi 5, kernel 6.8.0-1065-raspi
- **Mic/Speaker**: Seeed reSpeaker Flex XVF3800 L16K6Ch (USB `2886:0022`), Pi USB 포트 직결
- **USB 구성** (버스 ↔ 컨트롤러):

| 버스 | 컨트롤러 | 연결 장치 |
|---|---|---|
| usb2 (USB2) / usb3 (USB3) | `xhci-hcd.0` | `2-1` Logitech Unifying Receiver, `2-2` ReSpeaker |
| usb4 (USB2) / usb5 (USB3) | `xhci-hcd.1` | `4-2` FT232H |
| usb1 | `1000480000.usb` (USB-C 포트) | 없음 |

## Symptom

- `lsusb`에 ReSpeaker 없음, `arecord -l`/`aplay -l`에 ReSpeaker 카드 없음 → 마이크·스피커 동시 소실.
- 포트 상태(`/sys/bus/usb/devices/usb2/2-0:1.0/usb2-port2/state`) = `not attached`.
- 장치를 다시 꽂거나 다른 장치(핸드폰)를 꽂아도 **커널 로그에 이벤트 0건**.
- 전원은 정상 공급 (LED 점등, 핸드폰 충전) → 데이터 쪽 감지만 죽음.

## Timeline (`journalctl -k`)

| 시각 | 내용 |
|---|---|
| 16:15:06 | `2-2` ReSpeaker가 끊김과 동시에 같은 초에 재연결 (사람 손으로 불가능한 속도 → 전기적 순간 끊김) |
| 16:57:36 | 같은 컨트롤러의 `2-1` Logitech 수신기가 끊겼다가 즉시 재연결 |
| 16:57:53 | `2-2` ReSpeaker `USB disconnect` |
| 16:58:01–17:02:33 | `usb2-port2: attempt power cycle` → `unable to enumerate USB device` **17회 반복** 후 멈춤 |
| 17:02:33 이후 | USB 이벤트 없음 (재삽입·핸드폰 연결 모두 미감지) |
| 17:20:53–:55 | `xhci-hcd.0` unbind → bind, Logitech·ReSpeaker 모두 정상 재열거 |

`-71`/`-19`/`-32` 같은 열거 단계 에러는 **한 건도 없었다** — 장치가 오류를 내며
반복 재열거된 것이 아니라, 포트가 감지 자체를 멈춘 패턴이다.

## Root Cause 판단

- **케이블·ReSpeaker 아님**: 아무것도 바꾸지 않고 컨트롤러 리셋만으로 복구.
  다른 장치(핸드폰)도 같은 포트에서 감지되지 않았다.
- **컨트롤러 포트 상태 멈춤**: 커널의 재시도(17회)가 실패한 뒤 포트가 감지 불능 상태로 남음.
  커널의 "power cycle" 시도에도 LED/충전이 유지된 것으로 보아 VBUS는 실제로 끊기지 않았을
  가능성이 있다 (Pi 5 포트별 전원 차단 지원 여부는 미확인).

## 미확인 사항

- **촉발 원인**: 16:57:36 Logitech → 16:57:53 ReSpeaker 순으로 같은 컨트롤러의 두 포트가
  17초 사이에 흔들림 → 컨트롤러 단위의 순간 교란으로 추정. 후보는 모터 구동 시 전류 스파이크/EMI,
  정전기, 본체 접촉. 당시 모터 동작 여부는 미확인.
- **전원 부족**: 커널 로그에 undervoltage/over-current 경고는 없음. `vcgencmd get_throttled`는
  권한 문제(`/dev/vcio`)로 확인하지 못함.

## 복구 방법

케이블을 뽑지 않고 컨트롤러를 리셋한다. 같은 컨트롤러의 키보드·마우스가 2–3초 끊기므로
**반드시 한 줄로** 실행 (중간에 입력이 끊겨도 bind까지 진행되도록).

```bash
echo xhci-hcd.0 | sudo tee /sys/bus/platform/drivers/xhci-hcd/unbind; sleep 2; echo xhci-hcd.0 | sudo tee /sys/bus/platform/drivers/xhci-hcd/bind
```

- 장치가 다른 컨트롤러에 붙어 있으면 대상 이름이 다르다. 확인:
  `readlink -f /sys/bus/usb/devices/usb2` (경로에 `xhci-hcd.N`이 나옴).
- bind가 실패하면 해당 컨트롤러의 USB 장치는 재부팅 전까지 죽는다 — SSH 접속이 가능한 상태에서 하는 편이 안전.
- 복구 후 Python 파이프라인과 `build/Ray`는 끊긴 장치 핸들을 잡고 있을 수 있으므로 **재시작**한다.

## 재발 시 확인

```bash
# 포트 멈춤 시그니처 (재시도 반복 후 침묵)
journalctl -k --no-pager -g "unable to enumerate|attempt power cycle|USB disconnect"

# 포트별 상태 — 장치를 꽂았는데 not attached면 이 문제
for p in /sys/bus/usb/devices/usb*/*-0:1.0/usb*-port*; do \
  echo "$(basename $p) state=$(cat $p/state)"; done
```

- `usb1-port1`의 `disable=1`은 USB-C 포트 컨트롤러라서 정상이다. 이 문제와 무관.
- 같은 컨트롤러의 다른 장치(Logitech)가 함께 순간 끊겼는지 보면 컨트롤러 단위 교란인지 구분할 수 있다.
