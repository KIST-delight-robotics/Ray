"""DOA → look_at 추적 컨트롤러 (voice_pipeline 내부 스레드).

매 폴링마다 XVF3800에서 (DOA, VAD)를 읽어, 로봇 yaw 목표를 정해 CppBridge로 look_at 전송.
실제 부드러운 회전(min-jerk/pursue) + yaw 안전 클램프는 C++가 담당.

설계(데드존/히스테리시스/정면복귀):
  - 음성 감지(speech=1) 중에만 새 방향을 잡는다.
  - 목표 변화가 deadzone_deg 미만이면 무시(떨림 방지).
  - 일정 시간(return_delay_s) 말이 없으면 정면(0°)으로 복귀.
토글은 cpp/config.toml [head_tracking].enabled 하나로 통일(Python·C++ 공용).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.head_tracking.xvf3800_doa import XVF3800DOA

logger = logging.getLogger("voice_pipeline.head_tracking")

_HT_DEBUG = os.environ.get("HEAD_TRACK_DEBUG") == "1"  # 1이면 DOA·look_at 실시간 로그

# cpp/config.toml 경로 (repo_root/cpp/config.toml)
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "cpp" / "config.toml"


@dataclass
class HeadTrackingConfig:
    enabled: bool = False
    doa_front_deg: float = 0.0  # 로봇 정면에 해당하는 DOA 각도(캘리브레이션)
    doa_sign: float = 1.0  # +1/-1 (고개가 반대로 돌면 뒤집기)
    max_yaw_deg: float = 35.0  # 출력 yaw 한계(C++ tick 클램프 3200~4000 ≈ ±35°와 정합)
    deadzone_deg: float = 8.0  # (락 방식에선 미사용)
    return_delay_s: float = 2.0  # 음성 끊긴 뒤 정면 복귀까지 대기(=락 해제, 응답종료 근사)
    min_move_interval_s: float = 0.3  # 명령 최소 간격
    lock_confirm_s: float = 0.50  # 음성이 이만큼 지속돼야 그 방향에 락(짧은 blip 무시)
    t_per_deg: float = 0.03  # 조준 회전 속도(초/도). 1/t_per_deg = 각속도(°/s). 0.03→약 33°/s
    t_min_s: float = 0.35
    t_max_s: float = 1.50
    return_t_per_deg: float = 0.05  # 정면 복귀 속도(초/도). 거리 비례 → 일정 속도(멀수록 오래)
    return_duration_s: float = 2.0  # 정면 복귀 소요시간 상한(초)
    poll_hz: float = 20.0


def load_head_tracking_config(path: Path = _CONFIG_PATH) -> HeadTrackingConfig:
    """cpp/config.toml [head_tracking] 섹션을 읽는다(없으면 기본=비활성)."""
    cfg = HeadTrackingConfig()
    try:
        with open(path, "rb") as f:
            tbl = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("config 읽기 실패(%s) → head tracking 비활성", exc)
        return cfg
    ht = tbl.get("head_tracking", {})
    for field in (
        "enabled",
        "doa_front_deg",
        "doa_sign",
        "max_yaw_deg",
        "deadzone_deg",
        "return_delay_s",
        "min_move_interval_s",
        "lock_confirm_s",
        "t_per_deg",
        "t_min_s",
        "t_max_s",
        "return_t_per_deg",
        "return_duration_s",
        "poll_hz",
    ):
        if field in ht:
            setattr(cfg, field, ht[field])
    return cfg


def _wrap180(deg: float) -> float:
    """0~359(또는 임의) → [-180, 180)."""
    return ((deg + 180.0) % 360.0) - 180.0


class HeadTrackingController:
    """XVF3800 DOA를 읽어 CppBridge로 look_at을 보내는 백그라운드 스레드."""

    def __init__(self, bridge: CppBridge, cfg: HeadTrackingConfig) -> None:
        self._bridge = bridge
        self._cfg = cfg
        self._doa: XVF3800DOA | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = threading.Event()  # clear=일시중지(SLEEP). 웨이크워드 후 resume()로 set.
        self._last_target = 0.0  # 현재 머리 yaw 목표(°)
        self._last_send_t = 0.0

    def start(self) -> bool:
        """DOA 리더 열고 스레드 기동. 장치 없거나 실패하면 False(파이프라인은 계속)."""
        try:
            self._doa = XVF3800DOA()
        except Exception as exc:  # noqa: BLE001 — 장치 문제로 전체가 죽으면 안 됨
            logger.warning("head tracking 시작 실패(%s) → 비활성으로 계속", exc)
            return False
        self._thread = threading.Thread(target=self._run, name="head-tracking", daemon=True)
        self._thread.start()
        logger.info(
            "head tracking 시작 (front=%.0f° sign=%+d max=±%.0f° deadzone=%.0f°)",
            self._cfg.doa_front_deg,
            int(self._cfg.doa_sign),
            self._cfg.max_yaw_deg,
            self._cfg.deadzone_deg,
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._doa is not None:
            self._doa.close()
            self._doa = None

    def resume(self) -> None:
        """웨이크워드 이후(대화 모드) — DOA 추적 시작."""
        if not self._active.is_set():
            self._active.set()
            logger.info("head tracking 재개 (대화 모드)")

    def pause(self) -> None:
        """SLEEP 복귀 등 — 추적 중지 + 정면 복귀(거리 비례)."""
        if self._active.is_set():
            self._active.clear()
            self._return_to_front()  # 일정 속도로 정면 복귀
            logger.info("head tracking 일시중지 (정면 복귀)")

    def _run(self) -> None:
        """락 방식: 음성이 lock_confirm_s 이상 지속되면 그 순간 DOA로 1회 조준 후 '고정'.
        고정 중엔 DOA 변화를 무시(떨림·멀티·blip 방지). 침묵이 return_delay_s 넘으면 정면 복귀."""
        c = self._cfg
        dt = 1.0 / max(c.poll_hz, 1.0)
        last_speech_t = 0.0
        onset_t: float | None = None  # 음성 시작 시각(락 확정 대기)
        locked = False
        assert self._doa is not None
        while not self._stop.is_set():
            if not self._active.is_set():  # SLEEP(웨이크 전): 추적 안 함 + 상태 리셋
                onset_t, locked = None, False
                time.sleep(0.1)
                continue
            now = time.monotonic()
            try:
                doa, speech = self._doa.read()
            except Exception as exc:  # noqa: BLE001 — 일시적 USB 오류는 무시하고 계속
                logger.debug("DOA read 오류: %s", exc)
                time.sleep(0.2)
                continue

            if speech:
                last_speech_t = now
                if not locked:
                    if onset_t is None:
                        onset_t = now  # 음성 시작 — 지속 확인 시작
                    elif now - onset_t >= c.lock_confirm_s:
                        # 지속 확정 → 현재 DOA로 1회 조준하고 고정 (각속도 = 1/t_per_deg로 제한)
                        rel = _wrap180(doa - c.doa_front_deg)
                        yaw = max(-c.max_yaw_deg, min(c.max_yaw_deg, c.doa_sign * rel))
                        delta = abs(yaw - self._last_target)
                        dur = max(c.t_min_s, min(c.t_max_s, delta * c.t_per_deg))
                        self._safe_send(yaw, dur)
                        self._last_target, self._last_send_t = yaw, now
                        locked = True
                        if _HT_DEBUG:
                            logger.info("[HT] LOCK DOA=%d → yaw=%.1f dur=%.2f", doa, yaw, dur)
                # locked: DOA 무시(고정 유지)
            else:
                onset_t = None  # 음성 끊김 → 짧은 blip은 확정 전이라 무시됨
                # 침묵이 충분히 길면(응답 종료 근사) 정면 복귀 + 락 해제
                if locked and (now - last_speech_t) > c.return_delay_s:
                    self._return_to_front(now)
                    locked = False
                    if _HT_DEBUG:
                        logger.info("[HT] RELEASE → 정면 복귀")

            time.sleep(dt)

    def _return_to_front(self, now: float | None = None) -> None:
        """정면(0°)으로 복귀. 소요시간 = |현재각| × return_t_per_deg (일정 속도, 멀수록 오래)."""
        if now is None:
            now = time.monotonic()
        delta = abs(self._last_target)
        if delta < 1e-3:
            return  # 이미 정면
        c = self._cfg
        dur = max(c.t_min_s, min(c.return_duration_s, delta * c.return_t_per_deg))
        self._safe_send(0.0, dur)
        if _HT_DEBUG:
            logger.info("[HT] >>> RETURN to front dur=%.2f (from %.0f°)", dur, self._last_target)
        self._last_target, self._last_send_t = 0.0, now

    def _safe_send(self, yaw_deg: float, duration: float) -> None:
        try:
            self._bridge.send_look_at(yaw_deg, duration)
        except Exception as exc:  # noqa: BLE001 — 전송 실패가 추적 스레드를 죽이면 안 됨
            logger.debug("look_at 전송 실패: %s", exc)
