"""ReSpeaker XVF3800 DOA 기반 고개(yaw) 추적.

`uv run ray`(대화모드) 안에서 단일 프로세스로 동작한다(별도 프로세스 없음 → 마이크/포트 충돌 없음).
XVF3800의 DOA(음원 방향) + VAD를 USB로 읽어, 기존 CppBridge로 look_at을 보낸다.
C++가 [head_tracking].enabled일 때만 yaw 모터를 구동한다.
"""

from voice_pipeline.head_tracking.controller import (
    HeadTrackingConfig,
    HeadTrackingController,
    load_head_tracking_config,
)

__all__ = ["HeadTrackingController", "HeadTrackingConfig", "load_head_tracking_config"]
