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

### WiringPi (GPIO 제어)

Ubuntu 기본 패키지에 포함되어 있지 않으므로 수동 설치가 필요하다.

```bash
git clone https://github.com/WiringPi/WiringPi.git
cd WiringPi
./build
```

설치 확인:

```bash
gpio -v
```


## 2. uv 설치 (Python 패키지 매니저)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 후 셸을 재시작하거나 PATH를 추가한다:

```bash
export PATH="$HOME/.local/bin:$PATH"
```


## 3. 프로젝트 클론

```bash
git clone <repo-url> Ray
cd Ray
```


## 4. 외부 모델 저장소 클론

`pyproject.toml`의 `[tool.uv.sources]`가 이 경로를 참조하므로, `uv sync` 전에 반드시 클론해야 한다.

```bash
mkdir -p external
git clone https://github.com/ErikEkstedt/TurnGPT.git external/TurnGPT
git clone https://github.com/ErikEkstedt/VoiceActivityProjection.git external/VoiceActivityProjection
git clone https://github.com/MaAI-Kyoto/MaAI.git external/MaAI
```

> 네트워크가 느린 경우 `--depth 1`로 shallow clone 할 수 있다.


## 5. Python 의존성 설치

```bash
uv sync
```

테스트로 설치 확인:

```bash
uv run pytest
```


## 6. DynamixelSDK 빌드

```bash
mkdir -p third_party
git clone --branch 3.7.31 --depth 1 \
  https://github.com/ROBOTIS-GIT/DynamixelSDK.git \
  third_party/DynamixelSDK-3.7.31

cd third_party/DynamixelSDK-3.7.31/c++/build/linux_sbc
make
cd ../../../../..
```

> RPi 5 (aarch64)에서는 `linux_sbc` 디렉토리를 사용한다. `linux64`는 `-m64` 플래그 때문에 빌드되지 않는다. CMakeLists.txt에서 아키텍처별로 자동 분기된다.

모터 없이 테스트만 하려면 cmake에 `-DMOTOR_ENABLED=OFF`를 전달하면 DynamixelSDK와 WiringPi가 불필요하다.


## 7. C++ 빌드

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


## 8. WiFi 절전 모드 해제 (권장)

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
