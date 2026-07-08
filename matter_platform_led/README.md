# matter_platform_led — Matter LED 제어 "중간 박스"

사용자가 "불 꺼"라고 말하면 → STT → LLM 판단 → **이 모듈** → Matter WiFi LED가 꺼진다.
이 모듈은 그 흐름에서 **가운데 박스** 하나만 담당한다:

> **명령(켜/꺼)이 이미 정해진 상태에서, 그것을 실제 Matter On/Off 신호로 바꿔 전구에 보낸다.**

일부러 포함하지 않은 것:
- LLM이 *언제* 명령을 내릴지 판단하는 부분 → 나중에 배선 (아래 "LLM 통합 지점" 참고)
- 실제 전구 하드웨어에 대한 강한 의존 → 준비되면 백엔드만 교체

```
사용자 "불 꺼"
   │  (STT + LLM — 아직 미구현)
   ▼
MatterLedController.off()          ← 이 모듈의 진입점 (중간 박스)
   │
   ▼
MatterLightBackend.turn_off()      ← 교체 가능한 드라이버 인터페이스
   │
   ├─ MockBackend        (하드웨어 없이 검증 — 지금 기본값)
   ├─ ChipToolBackend    (실제 WiFi 전구 — chip-tool CLI 래핑)
   └─ PythonMatterServer (추후 저지연 프로덕션 — 아직 없음)
   ▼
 실제 전구가 WiFi로 꺼짐
```

핵심 설계 원칙: **호출부는 오직 `MatterLedController` 만 본다.** 백엔드를 바꿔도 CLI·컨트롤러·(추후) LLM tool 코드는 한 줄도 안 바뀐다.

---

## 파일 구조

| 파일 | 역할 |
|---|---|
| `interface.py` | `MatterLightBackend` 추상 인터페이스 + `LightStatus`. 위/아래를 가르는 경계선. |
| `mock_backend.py` | `MockBackend` — 하드웨어 없이 상태를 파일에 저장하며 신호 경로를 실제로 검증. |
| `chip_tool_backend.py` | `ChipToolBackend` — `chip-tool` CLI 를 subprocess 로 래핑 (실기기용). |
| `controller.py` | `MatterLedController` (중간 박스) + `build_backend()` 팩토리. |
| `config.py` / `config.toml` | 설정 로딩 (stdlib `tomllib`, 새 의존성 없음). |
| `cli.py` / `__main__.py` | `python -m matter_platform_led ...` 진입점 (`on`/`off`/`toggle`/`status`/`commission`/`share`). |
| `webui.py` | 로컬 브라우저 UI (`python -m matter_platform_led.webui`). 버튼이 실제 컨트롤러 호출. |
| `virtual_light/` | **하드웨어 없이 진짜 Matter 검증용** matter.js 가상 조명. → `virtual_light/README.md` |
| `exceptions.py` | `MatterError` 및 하위 예외. |

> **기존 파이프라인과 충돌 없음**: 이 폴더는 리포 루트의 독립 패키지다. `pyproject.toml` 의 `packages.find` 는 `voice_pipeline*` 만 포함하므로 이 모듈은 빌드 산출물/기존 import 에 전혀 끼어들지 않는다. `voice_pipeline` 의 어떤 파일도 수정하지 않았다. (참고: `voice_pipeline/led/` 의 LED 는 파이 본체에 붙은 WS2812 상태표시 스트립으로, 이 Matter WiFi 전구와는 완전히 별개다.)

---

## 지금 검증하는 법 (하드웨어 없이, mock)

`config.toml` 의 `backend = "mock"` (기본값) 상태에서 바로 실행한다.
mock 은 상태를 OS 임시폴더의 JSON 파일에 저장하므로, **별도 명령들 사이에서도 상태가 이어진다.**

```bash
# 리포 루트에서
uv run python -m matter_platform_led --backend mock commission   # 페어링(가짜)
uv run python -m matter_platform_led --backend mock status       # -> OFF
uv run python -m matter_platform_led --backend mock on           # -> on ✔
uv run python -m matter_platform_led --backend mock status       # -> ON
uv run python -m matter_platform_led --backend mock toggle       # -> OFF
uv run python -m matter_platform_led --backend mock off          # -> off ✔
```

**무엇이 검증되는가**
- 명령 → 컨트롤러 → 백엔드 → 상태변화의 **전체 경로**가 실제 프로덕션 코드로 흐른다 (mock 은 인터페이스만 가짜).
- 커미셔닝 전 `on` 호출 시 `MatterNotCommissionedError` 로 막히는 안전장치.
- CLI 인자/종료코드(성공 0, 실패 1)/에러 메시지.

mock 상태 초기화: 임시파일 삭제 (`rm /tmp/matter_led_mock_0x60.json`) 또는 `MockBackend(...).reset()`.

Python 에서 직접 (추후 LLM 이 하게 될 방식과 동일):
```python
from matter_platform_led import MatterLedController

with MatterLedController.from_config(backend_override="mock") as ctrl:
    ctrl.commission()
    ctrl.off()                 # ← "불 꺼" 가 최종적으로 이걸 호출한다
    print(ctrl.status())       # LightStatus(on=False, reachable=True)
```

---

## 진짜 Matter 검증 (가상조명 — 이 환경에서 동작 확인됨) ✅

전구가 없어도 **소프트웨어로 진짜 Matter 조명 기기**를 파이에 띄워 end-to-end로 검증했다.
`chip_tool` 백엔드로 커미셔닝→on/off→status→share(멀티어드민)까지 실제 Matter 프로토콜로 통과.
자세한 재현/폰 연동 절차는 **[`virtual_light/README.md`](virtual_light/README.md)** 참고. 요약:

```bash
# 터미널 A: 가상 조명 상주
cd matter_platform_led/virtual_light && npm install
export PATH="$HOME/.nvm/versions/node/v22.23.1/bin:$PATH"
node light.mjs --storage-path=./storage        # manual pairing code 34970112332 출력

# 터미널 B: 컨트롤러(우리 모듈, backend=chip_tool)
uv run python -m matter_platform_led commission   # 온-네트워크 커미셔닝 (1회)
uv run python -m matter_platform_led on           # → 조명 콘솔 "LIGHT is now ON 💡"
uv run python -m matter_platform_led status       # → status: ON
uv run python -m matter_platform_led off
uv run python -m matter_platform_led share        # 폰 멀티어드민용 창 열기 + 코드 발급
```

> 폰(Apple/Google) Home 앱은 **개발용 인증서 기기를 거부**할 수 있다 → `virtual_light/README.md`
> 의 "폰 멀티어드민" 절의 attestation 제약 참고. 확실한 폰 연동은 실제 인증 전구가 필요하다.

---

## 실제 WiFi 전구 연동 (chip_tool)

### 1. chip-tool 설치 (라즈베리파이 5, 64-bit OS 필수)
```bash
sudo apt install snapd
sudo reboot
sudo snap install chip-tool
```

### 2. 전구 준비
- **Matter 인증 WiFi 전구** (Nanoleaf / Tapo / Wiz Matter 등) 를 공장 초기화하여 **페어링 모드**로 둔다.
- 라벨/QR 의 **11자리 수동 페어링 코드** 또는 QR payload 를 확보.
- 이미 다른 허브(Apple/Google/SmartThings)에 물려 있다면, 해당 앱에서 "새 기기 추가/페어링 모드"로 **추가 setup code** 를 발급받아야 한다 (Matter multi-admin).

> **BLE 커미셔닝**: WiFi 전구는 최초에 파이의 **블루투스**로 붙어 WiFi 자격증명을 넘겨받는다(파이5 BLE 내장). 이후엔 WiFi 로 제어된다. 그래서 커미셔닝 시 `[wifi] ssid/password` 가 필요하다.

### 3. config.toml 채우기
```toml
backend = "chip_tool"

[device]
node_id = "0x60"          # 이 전구에 부여할 노드 ID (임의, 이후 계속 사용)
endpoint_id = 1           # 조명 On/Off 는 보통 엔드포인트 1
pairing_code = "34970112332"   # 라벨의 코드

[wifi]
ssid = "우리집WiFi"
password = "비밀번호"

[chip_tool]
bin = "chip-tool"
# 시판 인증기기 attestation 오류 시: connectedhomeip PAA 인증서 경로 지정
# paa_trust_store_path = "/home/delight/connectedhomeip/credentials/production/paa-root-certs/"
bypass_attestation = true  # 개발 중엔 true 로 우회, 배포 시 인증서로 대체 권장
```

### 4. 커미셔닝(1회) + 제어
```bash
uv run python -m matter_platform_led commission   # BLE로 WiFi 자격증명 전달 → 페어링
uv run python -m matter_platform_led on           # 전구 켜짐
uv run python -m matter_platform_led off          # 전구 꺼짐
uv run python -m matter_platform_led status       # 현재 상태 읽기
```

내부적으로 실행되는 chip-tool 명령:
- 커미셔닝(WiFi): `chip-tool pairing code-wifi 0x60 <ssid> <pw> <code> [--bypass-attestation-verifier true]`
- 켜기: `chip-tool onoff on 0x60 1` / 끄기: `chip-tool onoff off 0x60 1` / 토글: `chip-tool onoff toggle 0x60 1`
- 상태: `chip-tool onoff read on-off 0x60 1`

> ✅ `chip_tool_backend.py` 는 **가상조명(진짜 Matter 기기)으로 커미셔닝·on/off·status·share 전부 검증됨**
> (`virtual_light/` 참고). 실제 WiFi 인증 전구는 위 `code-wifi`(BLE) 경로만 추가로 타며, 시판기기
> attestation 실패 시 `bypass_attestation` 또는 `paa_trust_store_path` 로 해결한다. chip-tool 출력
> 포맷이 달라 `status` 파싱이 안 되면 `read_status()` 가 `reachable=False` 를 돌려주니 정규식
> (`_ONOFF_RE`)만 실제 출력에 맞게 조정하면 된다.

---

## 백엔드 갈아끼우기 — 어떻게 / 왜

### 어떻게
- **한 줄이면 끝**: `config.toml` 의 `backend` 값을 `"mock"` ↔ `"chip_tool"` 로 바꾸거나, CLI 에서 `--backend chip_tool` 로 그때그때 덮어쓴다.
- 호출부(`ctrl.off()`)는 **절대 안 바뀐다.** 팩토리 `build_backend()` 가 config 를 보고 알맞은 구현을 생성할 뿐이다.

### 왜 (백엔드가 여럿인 이유)
| 백엔드 | 언제 / 왜 |
|---|---|
| **mock** | 하드웨어가 없을 때. 신호 경로·안전장치·CLI·컨트롤러 로직을 실제 코드로 검증하기 위해. CI/개발용. |
| **chip_tool** | 실제 전구가 왔을 때 가장 빠르게 "정말 켜지나?"를 확인. 설치가 쉬움(`snap install`). 단, 명령마다 새 프로세스+세션 재수립 → **명령당 1~3초 지연**, 상태읽기는 텍스트 파싱이라 다소 취약. 검증/프로토타입에 적합. |
| **python-matter-server** (추후) | 실시간 대화 로봇 프로덕션. 컨트롤러가 상주하여 세션 유지 → **저지연**, 상태 이벤트 구독 가능. 대신 CHIP 네이티브 wheel 빌드/서버 상주가 필요해 무겁다. "불 꺼"가 즉각 반응해야 할 때 이걸로 교체. |

즉, **mock 으로 로직 완성 → chip_tool 로 실기기 검증 → (지연이 문제되면) python-matter-server 로 프로덕션 전환**의 3단계다. 세 번째 백엔드를 추가하려면 `MatterLightBackend` 를 구현한 `python_matter_server_backend.py` 를 만들고 `build_backend()` 에 분기 한 줄만 더하면 된다. 나머지 코드는 그대로다.

---

## 추후 LLM 통합 지점 (아직 미구현)

LLM 이 "불 꺼" 의도를 판단하면, tool/함수 호출이 결국 이걸 실행하게 배선하면 된다:

```python
# 앱 시작 시 1회 생성 (예: __main__.py DI 지점)
matter_led = MatterLedController.from_config()   # config.toml 의 backend 사용

# LLM tool 정의의 실행부에서:
def tool_set_room_light(on: bool) -> str:
    matter_led.on() if on else matter_led.off()
    return "완료"
```

- 이 프로젝트 규약상 실제 배선은 `voice_pipeline/llm/tools.py` (tool 정의/실행) 와 `voice_pipeline/__main__.py` (의존성 주입)에서 이뤄질 것이다. **지금은 건드리지 않았다** — 요청대로 중간 박스만 완성.
- 컨트롤러는 스레드에서 호출해도 안전하도록 상태를 백엔드(파일/기기)에 위임한다.

---

## 트러블슈팅 (실기기)

| 증상 | 해결 |
|---|---|
| `'chip-tool' not found` | `sudo snap install chip-tool`, 또는 `config.toml` 의 `[chip_tool].bin` 에 절대경로 지정. |
| 커미셔닝 중 attestation 실패 | `bypass_attestation = true` (개발) 또는 `paa_trust_store_path` 에 connectedhomeip PAA 인증서 경로 지정. |
| BLE 로 기기를 못 찾음 | 전구를 공장초기화해 페어링 모드로. 파이 블루투스 활성 확인(`bluetoothctl`). 다른 허브에 물려있으면 multi-admin setup code 발급. |
| `status` 가 unreachable | chip-tool 출력 포맷 상이 → `chip_tool_backend.py` 의 `_ONOFF_RE` 를 실제 출력에 맞게 조정. 또는 python-matter-server 백엔드로 전환. |
| 명령이 느리다 | chip-tool 은 명령마다 세션 재수립. 프로덕션은 python-matter-server 백엔드로 교체. |

## 참고 문서
- chip-tool 로 실제 스마트전구 제어: https://tomasmcguinness.com/2025/03/15/controlling-a-real-matter-smart-bulb-with-chip-tool/
- Matter on Ubuntu — chip-tool commission & control: https://canonical-matter.readthedocs-hosted.com/en/latest/how-to/chip-tool-commission-and-control/
- python-matter-server (추후 백엔드): https://github.com/home-assistant-libs/python-matter-server
