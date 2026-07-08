# RPi DAC+ (PCM5122) I2S DMA Halt on Raspberry Pi 5 — WS2812(SPI0) DMA 충돌

## Summary

Raspberry Pi 5에서 RPi DAC+ (PCM5122, I2S)와 WS2812 LED (SPI0)를 함께 운용할 때, RP1의 I2S DMA 전송이 멈추면서 DAC 오디오 출력이 불가능해지는 현상.

초기에는 PipeWire echo-cancel 모듈과의 상관관계로 추적했으나, 이후 레지스터/ALSA/커널 로그 분석으로 **SPI0(WS2812) DMA와 I2S DMA의 충돌**이 원인으로 좁혀짐. echo-cancel은 직접 원인이 아니라 DAC ALSA 스트림을 상시 열어두어 충돌 노출 시간을 늘리는 조건으로 재해석 (아래 Root Cause Analysis 참고).

## Environment

- **Board**: Raspberry Pi 5 Model B Rev 1.0
- **OS**: Ubuntu 24.04, kernel 6.8.0-1053-raspi
- **DAC**: RPi DAC+ (TI PCM5122, I2C addr 0x4c, 외부 오실레이터 없음)
- **LED**: WS2812 스트립, SPI0 (`/dev/spidev0.0`) — `rpi5_ws2812` 라이브러리 (SPI DMA 사용)
  - `scripts/hardware/led.py`, `voice_pipeline/led/led_controller.py`(DirectLedController) 모두 `spi_bus=0, spi_device=0`
- **Audio stack**: PipeWire 1.0.5, WirePlumber 0.4.17
- **Overlay**: `dtoverlay=rpi-dacplus`, `dtparam=spi=on`
- **I2S driver**: `designware_i2s`, `snd_soc_iqaudio_dac`, `snd_soc_pcm512x_i2c`
- **DMA**: SPI0과 I2S 모두 RP1 페리페럴 — RP1 공유 DMA 컨트롤러 사용 (`dw_axi_dmac`, DesignWare AXI DMA Controller, 8 channels)
- **Clock**: 외부 SCK 없이 BCLK로 내부 PLL 생성 (3-wire I2S, PCM5122 정상 동작 모드)
  - 부팅 시 `No SCLK, using BCLK: -2` 출력 (expected)

## Symptom

프로그램(마이크 캡처 + TTS 재생 + LED 구동) 실행 후 수 분 ~ 수십 분 경과 시, PipeWire 로그에 2초 간격 resync 에러가 반복 출력되며 DAC 오디오 출력이 불가능해짐.

```
spa.alsa: hw:0p: follower avail:768 delay:768 target:256 thr:256, resync (187 suppressed)
```

- 발생 시간은 불규칙 (5분, 10분, 17분 등)
- 오디오 재생 도중에 끊기는 경우도 확인됨
- 프로그램 미실행 시에도 장시간(수 시간~수 일) 후 발생한 이력 있음

### 복구

- **재부팅**: 복구됨, 재발함
- **PipeWire 재시작**: 복구 안 됨
- **커널 모듈 재로드** (`snd_soc_iqaudio_dac`, `snd_soc_pcm512x`): 복구 안 됨
- **I2S 버스 드라이버 재로드** (`designware_i2s`): `vc4`(GPU) 의존성으로 언로드 불가
- 소프트웨어 레벨에서 I2S 버스를 리셋할 수 없으며, 재부팅만이 유일한 복구 수단

## Root Cause Analysis (후속 분석)

문서 최초 작성 이후 resync 에러 발생 시점에 수집한 증거 3건.

### 1. PCM5122 PLL은 정상

resync 에러 발생 중 I2C로 DAC 레지스터 확인:

- PLL locked: `0x5E = 0x40`
- clock error 없음: `0x5F = 0x10`

PCM5122 입장에서는 클럭 문제가 아님. DAC 칩은 결백하며, 실패 지점은 DAC가 아니라 호스트 쪽. (초기 Open Question이었던 "PLL 락 상태와 resync의 상관관계" 확인 완료 — 상관없음)

### 2. 호스트 측 I2S DMA 정지 (hw_ptr freeze)

ALSA PCM 상태는 `RUNNING`인데 `hw_ptr`이 1초 후에도 동일. I2S DMA 전송 자체가 멈춘 상태.

PipeWire resync 반복(avail:768 고정)의 직접 원인 — 하드웨어 포인터가 움직이지 않으니 resync가 영원히 실패.

### 3. SPI0 DMA timeout과 동시 발생

```
spidev spi0.0: DMA transaction timed out
```

이 에러가 DAC resync 시작과 **정확히 같은 시각**(13:41:27)에 기록됨. SPI0에는 WS2812 LED가 연결되어 있음.

### 결론 (현재 가설)

- Pi 5에서 SPI0과 I2S는 둘 다 RP1 페리페럴이고, RP1의 공유 DMA 컨트롤러(DesignWare AXI DMAC, 8채널)를 사용한다.
- SPI(WS2812)와 I2S(DAC)의 DMA 동시 사용 중 DMA 계층에서 충돌/락업이 발생하여 양쪽 전송이 동시에 정지한다.
- **echo-cancel 모듈은 직접 원인이 아님.** DAC ALSA 스트림을 상시 열어두어 I2S DMA가 항상 돌게 만들고, 그만큼 SPI DMA와의 충돌 노출 시간을 늘리는 조건이었던 것으로 재해석.
  - 초기 재현 조건 C(echo-cancel 비활성, 4시간 무발생)도 이것으로 설명 가능: echo-cancel이 없으면 DAC 스트림이 재생 중에만 열리므로 노출 시간이 크게 줄어든다.

## Current Mitigation

- WS2812(SPI0)와 DAC(I2S)를 **동시에 사용하지 않는** 방식으로 회피 운용 중.
- 발생 시 재부팅 외 소프트웨어 복구 수단 없음은 동일.

## Initial Observation: Echo-Cancel 상관관계 (기록 보존)

초기에는 echo-cancel 모듈 활성 여부와의 상관관계로 추적했다. 아래는 당시 관찰 기록.

### PipeWire Echo-Cancel Configuration

```
# ~/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf

context.modules = [
    # 1) Loopback: ReSpeaker XVF3800 6ch → raw mic 4ch (CH2~5)
    {
        name = libpipewire-module-loopback
        ...
    }

    # 2) Echo cancellation with beamforming
    {
        name = libpipewire-module-echo-cancel
        args = {
            capture.props = {
                target.object = "xvf3800_raw_mics"   # 마이크 입력 (4ch)
            }
            playback.props = {
                target.object = "alsa_output.platform-soc_sound.stereo-fallback"  # DAC 참조
            }
            ...
            aec.args = {
                webrtc.beamforming = true
                ...
            }
        }
    }
]
```

### Reproduction

| 조건 | echo-cancel 모듈 | 프로그램 실행 | 결과 |
|---|---|---|---|
| A | 활성화 | 실행 | 수 분~수십 분 내 resync 에러 발생, DAC 출력 불가 |
| B | 활성화 | 미실행 | 수 시간~수 일 후 resync 에러 발생한 이력 있음 |
| C | **비활성화** | 실행 | 4시간 동안 resync 에러 없음 (OOM으로 테스트 중단) |

## Confirmed Facts

- resync 발생 중에도 PCM5122는 PLL locked (`0x5E=0x40`), clock error 없음 (`0x5F=0x10`)
- ALSA `RUNNING` 상태에서 `hw_ptr` 정지 — 호스트 측 I2S DMA 전송 정지
- `spidev spi0.0: DMA transaction timed out`이 DAC resync 시작과 같은 시각에 발생
- echo-cancel 모듈 비활성화 시 프로그램을 4시간 실행해도 DAC resync 에러가 발생하지 않았음
- 발생 후 재부팅 외에 소프트웨어 복구 방법 없음
- 오디오 재생 중/미재생 중 모두 발생 가능

## Not Confirmed

- RP1 DMA 충돌의 정확한 메커니즘 (채널 고갈, 컨트롤러 락업, 드라이버 버그 등)
- SPI 사용을 완전히 배제한 상태에서의 장기 무발생 검증 (현재는 동시 사용 회피로 운용 중이며 재발 없음)
- Pi 4에서 같은 현상이 발생하는지 (미테스트 — Pi 4는 RP1이 없어 DMA 구조가 다름)

## Open Questions

1. RP1의 DesignWare AXI DMAC에서 SPI0과 I2S 채널 할당/중재 방식은? 커널(raspi) 또는 RP1 펌웨어에 알려진 이슈가 있는가?
2. WS2812를 SPI DMA가 아닌 다른 방식(PWM, PIO 등)으로 구동하면 충돌을 회피할 수 있는가?
3. DAC를 열어두는 시간을 줄이면(echo-cancel의 playback 노드 `node.passive = true` 등) 노출을 줄일 수 있는가 — 근본 해결은 아니지만 완화책으로 유효한가?
