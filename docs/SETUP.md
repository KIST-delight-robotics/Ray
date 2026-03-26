# Setup Guide — Raspberry Pi 5

Raspberry Pi 5 (Ubuntu 24.04, aarch64) 초기 설정 가이드.


## 1. 시스템 패키지 설치

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake pkg-config \
  libsndfile1-dev \
  libasound2-dev portaudio19-dev \
  libopenal-dev libvorbis-dev libflac-dev libogg-dev
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
uv sync --extra hardware
```

> `--extra hardware`는 LED 드라이버(`rpi5-ws2812`) 등 하드웨어 의존성을 함께 설치한다. RPi가 아닌 환경에서는 `uv sync`만으로 충분하다.

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
unzip models.zip -d models/
rm models.zip
```


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


## 9. WiFi 절전 모드 해제 (권장)

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
