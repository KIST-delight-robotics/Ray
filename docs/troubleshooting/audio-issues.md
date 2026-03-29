# Audio Troubleshooting Log

## Environment
- Raspberry Pi 5 (Linux 6.8.0-1048-raspi)
- DAC: RPi DAC+ (PCM512x, I2S, card 0)
- Microphone: C-Media USB PnP Sound Device (card 번호 재부팅 시 변동)
- Audio stack: PipeWire 1.0.5 + pipewire-pulse + echo-cancel module
- OpenAL (SFML) → PipeWire → DAC (C++ playback)
- PyAudio → PipeWire → echo-cancel-source (Python capture)
- 접속 방식: SSH (GDM 자동 로그인 활성화)


## 해결된 이슈

### Issue 1: SSH에서 오디오 장치 접근 불가
**증상**: `arecord`/`aplay`에서 사운드 카드를 찾지 못함. `/dev/snd/*` 장치에 ACL 미부여.

**원인**: systemd-logind는 로컬 seat 사용자에게만 `/dev/snd/*` ACL 부여. GDM 자동 로그인이 정상 작동하면 SSH에서도 간접적으로 접근 가능했으나, 어떤 시점부터 자동 로그인이 seat0을 limdaemin에게 할당하지 못함. SSH 세션은 `Remote=yes`, `Seat=없음`이라 ACL 미부여.

**해결**: `sudo usermod -aG audio limdaemin` — SSH에서도 audio 그룹 권한으로 접근.

### Issue 2: SSH에서 PipeWire 접근 불가
**증상**: SSH에서 프로그램 실행 시 PipeWire를 통한 오디오가 동작하지 않음.

**원인**: PipeWire는 user session 데몬. SSH 세션에서 `XDG_RUNTIME_DIR`과 `DBUS_SESSION_BUS_ADDRESS`가 미설정이면 PipeWire 소켓에 접근 불가.

**해결**: `~/.bashrc`에 추가:
```bash
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
```

### Issue 3: PulseAudio connection timeout
**증상**: `ALSA lib pulse.c:242:(pulse_connect) PulseAudio: Unable to connect: Timeout`

**원인**: Issue 1, 2의 결과. 또는 PipeWire가 크래시 후 재시작되었으나 pipewire-pulse와 동기화되지 않은 경우.

**해결**: Issue 1, 2 해결로 근본 해결. 증상 발생 시 `systemctl --user restart pipewire pipewire-pulse`.

### Issue 4: Echo cancellation — 정상 확인
**증상**: 로봇 음성 재생 중 에코에 의한 인터럽션 발생 의심.

**확인 결과**:
- PipeWire echo-cancel 모듈 라우팅 정상 확인 (C++ → echo-cancel-sink → DAC, USB mic → echo-cancel-capture → echo-cancel-source → Python)
- 녹음 비교: echo-cancel-source peak=23, 원본 USB 마이크 peak=18,358. 에코 캔슬 정상 동작.
- `source_master` 미지정 상태에서도 PipeWire가 USB 마이크 자동 감지.
- 에코에 의한 인터럽션이 아닌 다른 원인(장치 접근 문제 등)으로 판단.


## 모니터링 중인 이슈

### Issue 5: DAC 클럭 복구 실패 (No SCLK)
**증상**: 일정 시간 idle 후 재생 시 소리가 나지 않음. 커널 로그에 `pcm512x 1-004c: No SCLK, using BCLK: -2`.

**원인 (추정)**: DAC 마스터 모드에서 idle 시 I2S 스트림이 닫히면서 클럭이 멈추고, 재개 시 클럭 복구 실패. 트러블슈팅 과정에서 PipeWire 반복 재시작으로 유발되었을 가능성도 있음.

**시도한 해결**: `dtoverlay=rpi-dacplus,slave` — RPi가 클럭 마스터. 효과 검증 중.

**상태**: 모니터링 중. `~/.local/log/hw_alerts.log`의 `sclk_err` 카운트 변화로 재발 여부 확인.

### Issue 6: PipeWire SEGV 크래시
**증상**: PipeWire가 SIGSEGV(signal 11)로 크래시. systemd 자동 재시작되나 pipewire-pulse와 동기화 안 됨 → 오디오 먹통.

**발생 패턴**: 장시간(16시간+) 실행 후 발생. crash 파일: `/var/crash/_usr_bin_pipewire.1000.crash`

**원인**: 미확인. 디버그 심볼 없어 스택 트레이스 추출 불가. HDMI 끊김과 관련 가능성 있으나 미확인.

**상태**: 모니터링 중. `~/.local/log/hw_alerts.log`의 `pw_crash` 카운트 변화로 재발 여부 확인.

### Issue 7: HDMI 화면 끊김 후 복구 불가
**증상**: idle 상태에서 HDMI 신호가 끊기고, 마우스/키보드로 복구 불가. 모니터에 "연결 없음" 표시. SSH는 정상.

**원인**: 미확인. GNOME idle-delay(5분)에 의한 DPMS 화면 꺼짐 후 vc4 드라이버가 HDMI를 복구하지 못하는 것으로 추정.

**상태**: 모니터링 중. `~/.local/log/hw_alerts.log`의 `hdmi` 상태 변화로 발생 시점 확인.


## Config Changes Made

### `/home/limdaemin/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf`
- `source_master` 추가 후 제거 (PipeWire crash 유발, 자동 감지로 충분)
- 최종 상태: `sink_master`만 지정

### `/boot/firmware/config.txt`
- `dtoverlay=rpi-dacplus` → `dtoverlay=rpi-dacplus,slave`

### `~/.bashrc`
- `XDG_RUNTIME_DIR`, `DBUS_SESSION_BUS_ADDRESS` 환경변수 추가

### System
- `sudo usermod -aG audio limdaemin`


## Monitoring

### RPi-Monitor
- 웹 UI: `http://<IP>:8888` — Audio & Display 섹션
- RRD 데이터: `/var/lib/rpimonitor/stat/` (core_volt.rrd, throttled_num.rrd, pipewire_ok.rrd 등)
- 설정: `/etc/rpimonitor/template/audio_health.conf`

### 이상 감지 텍스트 로그
- `~/.local/log/hw_alerts.log` — 상태 변화 시에만 기록 (10초 주기 체크)
- 항목: throttled, voltage, HDMI, DAC, PipeWire, SCLK에러수, USB에러수, PW크래시수
- 이전 상태를 `/tmp/hw_monitor_prev`에 저장, 변화 없으면 기록 안 함
- 헬퍼 스크립트: `/usr/local/bin/pw_status_helper.sh`
- 타이머: `systemctl --user status pw-status-helper.timer`


## ALSA Noise Messages (무시 가능)
PyAudio/PortAudio 초기화 시 모든 가능한 PCM 장치를 열거하면서 발생하는 경고. RPi DAC+가 단순 I2S DAC이라 front, surround, hdmi, spdif 등의 PCM 타입 정의가 없어서 나오는 메시지. JACK 미실행 경고, OSS `/dev/dsp` 미존재 경고 포함. 동작에 영향 없음.


## 비고
스피커 테스트: `speaker-test -D pulse -t sine -f 440 -c 2 -l 1`
