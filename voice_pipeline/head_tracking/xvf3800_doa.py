"""ReSpeaker Flex XVF3800 음원방향(DOA) USB 리더.

XVF3800은 USB 컨트롤 전송으로 DOA_VALUE 파라미터(resid=20, cmdid=18)를 노출한다:
  payload[0..1] = DOA 각도(uint16, 0~359°),  payload[2] = 음성감지(VAD, 1/0).
이는 ALSA 오디오 캡처(6ch)와 다른 컨트롤 인터페이스라 같은 프로세스에서 공존 가능.
udev 룰(99-respeaker.rules)로 일반 사용자 접근 가능(root 불필요).
"""

from __future__ import annotations

import contextlib
import logging

import usb.core
import usb.util

logger = logging.getLogger("voice_pipeline.head_tracking")

_VID = 0x2886  # Seeed
_DOA_RESID = 20
_DOA_CMDID = 18
_DOA_LEN = 5  # 응답 = 상태 1B + payload 4B
_TIMEOUT_MS = 400  # 짧게 — 행 방지


class XVF3800DOA:
    """XVF3800에서 (DOA 각도, 음성감지)를 읽는다."""

    def __init__(self, vid: int = _VID) -> None:
        dev = usb.core.find(idVendor=vid)
        if dev is None:
            raise RuntimeError(f"ReSpeaker XVF3800 (VID 0x{vid:04x}) not found")
        self._dev = dev
        logger.info("XVF3800 DOA reader 연결 (PID 0x%04x)", dev.idProduct)

    def read(self) -> tuple[int, bool]:
        """(doa_deg 0~359, speech) 반환. USB 오류는 예외로 전파."""
        r = self._dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            0x80 | _DOA_CMDID,
            _DOA_RESID,
            _DOA_LEN,
            _TIMEOUT_MS,
        ).tolist()
        doa = (r[1] + r[2] * 256) % 360
        speech = bool(r[3])
        return doa, speech

    def close(self) -> None:
        with contextlib.suppress(Exception):
            usb.util.dispose_resources(self._dev)
