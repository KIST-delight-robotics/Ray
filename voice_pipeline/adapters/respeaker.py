"""ReSpeaker Flex (XVF3800) USB 제어 어댑터.

- ``reset()``: 벤더 제어 전송으로 XVF3800 칩을 리부트한다 — 보드의 RST 버튼과 같은 효과.
  장치는 USB 버스에서 잠시 사라졌다가 재열거되며(실측: 사라짐 0.17 s, ALSA/PyAudio 재인식 ~1.1 s),
  모든 XVF3800 파라미터가 기본값으로 돌아간다.
- ``wait_present()``: 장치가 USB 버스에 보일 때까지 대기. 리셋 뒤 마이크를 열기 전 가드용.

프로토콜은 Seeed 의 ``xvf_host.py``(reSpeaker_Flex 저장소) 에서 REBOOT 명령 하나만 옮겨 왔다:
vendor OUT 요청, bRequest=0, wValue=cmdid, wIndex=resid, payload 1바이트.

모든 실패(pyusb 없음, 장치 없음, 권한 거부)는 경고 로그 후 False — 리셋은 예방 조치라
파이프라인 시작을 막지 않는다. 권한은 udev 규칙으로 준다: docs/SETUP.md
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("voice_pipeline.respeaker")

_VENDOR_ID = 0x2886  # Seeed Technology — PID 는 펌웨어 변형(6ch/2ch 등)마다 달라 VID 로만 찾는다
_REBOOT_RESID = 48  # wIndex
_REBOOT_CMDID = 7  # wValue
_CTRL_TIMEOUT_MS = 1000
_DISAPPEAR_TIMEOUT_SEC = 1.0  # REBOOT 후 버스에서 사라질 때까지 (실측 0.17 s)
_POLL_INTERVAL_SEC = 0.05


def _find_device() -> Any | None:
    """VID 로 ReSpeaker USB 장치를 찾는다. pyusb 가 없으면 None."""
    try:
        import usb.core
    except ImportError:
        logger.warning("pyusb not installed — ReSpeaker control unavailable")
        return None
    try:
        return usb.core.find(idVendor=_VENDOR_ID)
    except Exception as exc:  # libusb backend 없음 등
        logger.warning("ReSpeaker USB lookup failed: %s", exc)
        return None


def reset() -> bool:
    """XVF3800 을 리부트하고 장치가 USB 버스에서 사라질 때까지 기다린다.

    Returns:
        REBOOT 전송에 성공했으면 True. 장치 없음·권한 거부 등은 경고 로그 후 False.
    """
    dev = _find_device()
    if dev is None:
        logger.warning("ReSpeaker not found on USB — skipping reset")
        return False

    import usb.util

    request_type = usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE
    try:
        dev.ctrl_transfer(request_type, 0, _REBOOT_CMDID, _REBOOT_RESID, [1], _CTRL_TIMEOUT_MS)
    except Exception as exc:
        logger.warning(
            "ReSpeaker reset failed: %s (USB 쓰기 권한이 없으면 udev 규칙 확인 — docs/SETUP.md)",
            exc,
        )
        return False
    finally:
        try:
            usb.util.dispose_resources(dev)
        except Exception:
            pass

    # 전송 직후엔 아직 옛 장치가 보인다. 뒤따르는 wait_present() 가 재열거된 장치를 기다리도록
    # 사라짐까지 확인한다 (상한 안에 안 사라져도 실패로 보지 않는다).
    deadline = time.monotonic() + _DISAPPEAR_TIMEOUT_SEC
    while time.monotonic() < deadline and _find_device() is not None:
        time.sleep(_POLL_INTERVAL_SEC)
    logger.info("ReSpeaker reset sent — device re-enumerating")
    return True


def wait_present(timeout_sec: float) -> bool:
    """ReSpeaker 가 USB 버스에 보일 때까지 최대 ``timeout_sec`` 대기.

    Args:
        timeout_sec: 대기 상한(초).

    Returns:
        장치가 보이면 True, 타임아웃이면 False.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        if _find_device() is not None:
            return True
        if time.monotonic() >= deadline:
            logger.warning("ReSpeaker not visible on USB after %.1fs", timeout_sec)
            return False
        time.sleep(_POLL_INTERVAL_SEC)
