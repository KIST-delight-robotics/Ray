# virtual_light — 하드웨어 없이 "진짜 Matter" 검증

전구를 사지 않고, **소프트웨어로 진짜 Matter On/Off 조명 기기**를 이 라즈베리파이에 띄운다
([matter.js](https://github.com/project-chip/matter.js) 사용). 이건 mock이 아니라 실제 Matter
프로토콜을 mDNS로 공지하고 On/Off 클러스터를 노출하는 **실기기**다. 그래서 우리
`ChipToolBackend`(컨트롤러)가 이걸 **실제로 커미셔닝하고 켜고 끈다.**

```
[ 컨트롤러 ]                              [ 기기 = 이 폴더 ]
matter_platform_led (chip_tool 백엔드)   ──실제 Matter over IP──▶  light.mjs (matter.js 가상 On/Off 조명)
        chip-tool 커맨드                                            켜지면 콘솔에 "LIGHT is now ON 💡"
        ▲
        │ 같은 fabric 공유(멀티어드민)
   폰 Home 앱 ← `share` 로 창 열고 추가 (주의: 아래 attestation 제약)
```

한 대의 파이에서 컨트롤러+기기를 모두 돌리므로 **블루투스·두 번째 기기·방화벽 설정 불필요**
(커미셔닝은 온-네트워크/로컬 IP).

---

## 이미 검증된 사실 (이 환경에서 실제로 통과함)

- 가상조명이 `is online` + mDNS `_matterc._udp` 로 공지 → chip-tool이 **discriminator 3840** 으로 발견.
- `chip-tool pairing code 0x60 34970112332 --bypass-attestation-verifier true` → **PASE→CASE→`Commissioned`** 완료.
- `python -m matter_platform_led on` → 조명 콘솔 `LIGHT is now ON 💡`.
- `python -m matter_platform_led status` → `status: ON` (우리 `_ONOFF_RE` 파싱이 실제 출력에서 동작).
- `python -m matter_platform_led off` → `LIGHT is now OFF ⚫`.
- `python -m matter_platform_led share` → 기기에 커미셔닝 창 열림 + 폰용 QR(`MT:...`) 발급.

---

## 처음부터 재현하기

### 0. 사전 (이 환경엔 이미 설치됨)
- **Node.js 20+** (여기선 nvm으로 22 설치, sudo 불필요):
  ```bash
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
  export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm install 22
  ```
  이후 셸마다: `export PATH="$HOME/.nvm/versions/node/v22.23.1/bin:$PATH"`
- **chip-tool** (컨트롤러, root 필요):
  ```bash
  sudo snap install chip-tool
  ```

### 1. 의존성 설치 (이 폴더에서, 1회)
```bash
cd matter_platform_led/virtual_light
export PATH="$HOME/.nvm/versions/node/v22.23.1/bin:$PATH"
npm install          # @matter/main, @matter/examples
```

### 2. 가상조명 실행 (상주)
```bash
node light.mjs --storage-path=./storage
# 종료 후 완전 초기화(재커미셔닝 원할 때): node light.mjs --storage-path=./storage --storage-clear
```
`uncommissioned ... manual pairing code: 34970112332` 와 QR이 출력되면 준비 완료. **이 창은 켜둔다.**

### 3. 컨트롤러로 커미셔닝 + 제어 (다른 터미널, 리포 루트에서)
`config.toml` 이 이미 `backend = "chip_tool"`, `pairing_code = "34970112332"` 로 맞춰져 있다.
```bash
uv run python -m matter_platform_led commission   # 온-네트워크 커미셔닝(1회)
uv run python -m matter_platform_led on           # 조명 콘솔에 ON 표시
uv run python -m matter_platform_led off
uv run python -m matter_platform_led status
```
> `commission` 은 최초 1회만. 이미 커미셔닝된 기기에 다시 하면 chip-tool이 실패한다. 재시도하려면
> 2번에서 `--storage-clear` 로 조명을 공장초기화하고, chip-tool 저장소도 비운다:
> `chip-tool storage clear-all`.

웹 UI로도 확인 가능 (버튼 클릭 = 실제 chip-tool 호출):
```bash
uv run python -m matter_platform_led.webui        # http://127.0.0.1:8765
```

---

## 폰 멀티어드민 (Apple / Google / SmartThings)

Matter는 한 기기에 여러 컨트롤러가 붙을 수 있다(multi-admin). 파이가 먼저 커미셔닝한 뒤 창을 열면 폰이 같은 기기를 추가할 수 있다.

```bash
uv run python -m matter_platform_led share
# → "MT:...." QR/코드 출력. 폰 Home 앱에서 "기기 추가 → Matter → 코드 입력/스캔"
```

> ⚠️ **중요한 현실 제약 — 이 가상조명은 "개발용 인증서(test attestation, VID 0xFFF1)"** 를 쓴다.
> - **Apple Home**: 미인증 기기를 거부한다 → 이 가상조명은 **추가 실패**할 가능성이 매우 높다.
> - **Google Home**: 기본적으로 인증을 요구. 개발자 콘솔에 test VID/PID 등록 등 별도 설정이 필요.
> - **SmartThings**: 가장 관대함 → "미인증 기기" 경고를 수락하면 추가될 가능성이 있다.
>
> 즉 **멀티어드민 "메커니즘" 자체는 검증됐지만(창 열림+코드 발급)**, 폰 소비자 앱이 *개발용* 기기를
> 받아줄지는 앱 정책에 달렸다. **폰 연동을 확실히 하려면 실제 Matter 인증 전구**가 필요하다.
> 인증 전구가 오면 이 모듈은 그대로 동작한다 — `config.toml` 의 pairing code/노드만 바꾸면 된다.
>
> 순수하게 멀티어드민을 검증만 하려면, 두 번째 컨트롤러(다른 chip-tool fabric / matter.js 컨트롤러)로
> `share` 가 발급한 코드를 커미셔닝하면 된다.

---

## 파일
- `light.mjs` — matter.js 가상 On/Off 조명 (고정 test 파라미터 → pairing code 34970112332).
- `package.json` — `@matter/main`, `@matter/examples` 의존성, `type: module`.
- `.gitignore` — `node_modules/`, `storage/` 제외.

## 참고
- matter.js: https://github.com/project-chip/matter.js
- chip-tool commission & control: https://canonical-matter.readthedocs-hosted.com/en/latest/how-to/chip-tool-commission-and-control/
