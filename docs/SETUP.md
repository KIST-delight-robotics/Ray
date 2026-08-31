# Setup Guide — Raspberry Pi 5

Raspberry Pi 5 (Ubuntu 24.04, aarch64) 초기 설정 가이드.


## 1. 시스템 패키지 설치

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake pkg-config \
  libsndfile1-dev \
  libasound2-dev portaudio19-dev \
  libopenal-dev libvorbis-dev libflac-dev libogg-dev \
  zlib1g-dev
```


## 2. uv 설치 (Python 패키지 매니저)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 후 셸을 재시작하거나 PATH를 추가한다:

```bash
export PATH="$HOME/.local/bin:$PATH"
```


## 3. 외부 모델 저장소 클론

`pyproject.toml`의 `[tool.uv.sources]`가 이 경로를 참조하므로, `uv sync` 전에 반드시 클론해야 한다.

```bash
mkdir -p external
git clone https://github.com/ErikEkstedt/TurnGPT.git external/TurnGPT
git clone https://github.com/ErikEkstedt/VoiceActivityProjection.git external/VoiceActivityProjection
git clone https://github.com/MaAI-Kyoto/MaAI.git external/MaAI
```


## 4. Python 의존성 설치

```bash
uv sync
```

> LED 드라이버(`rpi5-ws2812`)는 기본 의존성이며 `sys_platform == 'linux'` 마커로 Linux에서만 설치된다. macOS/Windows에서는 자동 제외되고 LED 계층은 noop으로 폴백한다. 하드웨어가 연결돼 있어도 LED를 끄려면 `LED_ENABLED=0`으로 실행한다.

테스트로 설치 확인:

```bash
uv run pytest
```


## 5. Third-party 라이브러리 (모터 사용 시)

모터 없이 테스트만 하려면 이 단계를 건너뛰고 cmake에 `-DMOTOR_ENABLED=OFF`를 전달한다.

### WiringPi (GPIO 제어)

Ubuntu apt 버전(2.x)은 RPi 5를 지원하지 않으므로 수동 빌드가 필요하다.

```bash
git clone https://github.com/WiringPi/WiringPi.git
cd WiringPi && ./build && cd ..

# 설치 확인
gpio -v
```

### DynamixelSDK

cmake 빌드 시 `ExternalProject`로 자동 클론 및 빌드된다. 별도 설치 불필요.

> RPi 5 (aarch64)에서는 `linux_sbc` 디렉토리가 사용된다. CMakeLists.txt에서 아키텍처별로 자동 분기된다.


## 6. C++ 빌드

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

빌드 결과물: `build/Ray`

모터/센서 없이 빌드:

```bash
cmake -DMOTOR_ENABLED=OFF ..
```


## 7. 모델 체크포인트

체크포인트 파일은 git에 포함되지 않으므로 별도로 다운로드하여 `models/`에 배치한다.

```bash
wget -O models.zip "https://kist.gov-dooray.com/share/drive-files/m1olidaukdr7.5LnEpNTcR1OgoDBYEyMAQA"
unzip models.zip
rm models.zip
```


### 모델 캐시 부트스트랩 (첫 실행 1회, 온라인 필요)

production wiring은 부팅 시 네트워크 의존을 없애기 위해 **로컬 캐시만** 쓴다
(`create_embedder(local_files_only=True)`, `TIKTOKEN_CACHE_DIR`). 새 기기에서는 캐시가
없어 첫 기동이 실패하므로, 네트워크가 있는 상태에서 아래를 1회 실행해 캐시를 만든다:

```bash
# 임베딩 모델 (HF 허브 → ~/.cache/huggingface). local_files_only=False로 1회 로드
uv run python -c "from voice_pipeline.adapters.embedder import create_embedder; create_embedder(expected_dimension=384)"
# tiktoken 인코딩 사전 (→ $TIKTOKEN_CACHE_DIR). ray.env의 경로와 같아야 한다
mkdir -p ~/.cache/tiktoken
TIKTOKEN_CACHE_DIR=~/.cache/tiktoken uv run python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

`HF_HUB_OFFLINE=1`은 쓰지 말 것 — sentence-transformers가 허브 트리 조회를 시도해 예외로 죽는다.

## 8. API 인증 설정

### Google Cloud (ASR)

GCP 콘솔에서 서비스 계정 키(JSON)를 발급받고, 파일 경로를 환경변수에 등록한다.

```bash
# 파일 경로에 맞게 수정
echo 'export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json' >> ~/.bashrc
source ~/.bashrc
```

### OpenAI (LLM, TTS)

```bash
echo 'export OPENAI_API_KEY=sk-...' >> ~/.bashrc
source ~/.bashrc
```


## 9. USB 전류 한도 해제 (GPIO/비PD 전원 공급 시)

공식 27W USB-C PD 어댑터가 아닌 경로(GPIO 등)로 전원을 공급하면, RPi 5 펌웨어는 PD 협상이 없으므로 USB 전체 전류 한도를 600mA로 강제 제한한다. USB 마이크 + 다른 주변기기를 함께 쓰면 쉽게 over-current가 발생한다.

`/boot/firmware/config.txt` 끝에 추가:

```
usb_max_current_enable=1
```

재부팅 후 적용. 공급 전원 용량이 충분하다는 전제하에만 사용한다.

확인:

```bash
# under-voltage 알람 비트 (0=정상)
cat /sys/class/hwmon/hwmon3/in0_lcrit_alarm

# 과거 over-current 이벤트 조회
journalctl -k --no-pager | grep -i over-current
```


## 10. FTDI USB-Serial latency 단축 (모터 통신 응답성)

Dynamixel U2D2는 FTDI 기반 USB-Serial 어댑터를 사용하는데, Linux의 FTDI 드라이버는 기본적으로 수신 바이트를 **16ms 동안 버퍼링한 뒤** 유저스페이스로 전달한다. 이 때문에 status 응답을 기다리는 모든 SDK 호출(`readAllState`, `write*TxRx` 등)이 호출당 최대 16ms씩 늘어난다.

증상:
- 짧은 주기 로깅(예: `HighFreqLogger`의 5ms 목표)이 실제로는 ~16ms로 돔
- `dxl_mutex_`가 사실상 포화되어 다른 모터 제어 스레드가 starve → 동작이 "끊기고 휙휙" 튀는 현상

udev 규칙으로 latency timer를 1ms로 단축:

```bash
sudo tee /etc/udev/rules.d/99-ftdi-latency.rules << 'EOF'
ACTION=="add", SUBSYSTEM=="usb-serial", ATTR{latency_timer}="1"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

재부팅 후 적용 확인:

```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# 출력: 1
```


## 11. WiFi 절전 모드 해제 (권장)

RPi 5의 WiFi는 기본적으로 Power Management가 켜져 있어, 유휴 시 속도가 크게 떨어진다.

확인:

```bash
iwconfig wlan0 | grep "Power Management"
```

해제:

```bash
sudo iw wlan0 set power_save off
```

영구 적용 (NetworkManager 재시작 또는 재부팅 후에도 유지):

```bash
sudo tee /etc/NetworkManager/conf.d/wifi-powersave-off.conf << 'EOF'
[connection]
wifi.powersave = 2
EOF
sudo systemctl restart NetworkManager
```

> `iw` 명령이 없으면 `sudo apt install iw`로 설치하거나, 설정 파일 생성 후 NetworkManager 재시작으로 대체 가능.
