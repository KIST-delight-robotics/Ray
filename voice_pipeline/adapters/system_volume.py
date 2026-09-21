"""PipeWire 기본 출력 싱크 볼륨 래퍼 (``wpctl``).

C++ 재생부는 SFML → OpenAL → PipeWire 로 소리를 내므로, 싱크 볼륨을 바꾸면 대화 음성·인사 WAV·차임이
한 경로로 함께 조절된다. 입 모션은 볼륨 적용 전 PCM 으로 만들고 XVF3800 AEC 참조는 PipeWire 뒤 USB 신호라
둘 다 영향받지 않는다. 두 프로세스 모두 ``systemd --user`` 아래에서 돌아 ``XDG_RUNTIME_DIR`` 가 있으므로
서비스 환경에서도 wpctl 이 PipeWire 에 붙는다.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("voice_pipeline.system_volume")

_SINK = "@DEFAULT_AUDIO_SINK@"  # PipeWire 기본 싱크 (ReSpeaker). 기기마다 노드 ID 가 달라 별칭으로 지정
_TIMEOUT_SEC = 3.0  # wpctl 응답 상한. PipeWire 가 안 떠 있으면 바로 실패하므로 여유만 둔다


def set_sink_volume(percent: int) -> None:
    """기본 싱크 볼륨을 ``percent`` (0~100) % 로 맞춘다.

    Raises:
        RuntimeError: wpctl 이 없거나, 실패했거나, 시간 내 응답하지 않았을 때.
    """
    percent = max(0, min(100, int(percent)))
    cmd = ["wpctl", "set-volume", _SINK, f"{percent}%"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=_TIMEOUT_SEC)
    except FileNotFoundError as exc:
        raise RuntimeError("wpctl not found: PipeWire tools are not installed") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip() if exc.stderr else f"exit {exc.returncode}"
        raise RuntimeError(f"wpctl set-volume failed: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"wpctl set-volume timed out after {_TIMEOUT_SEC}s") from exc
    logger.debug("Sink volume set to %d%%", percent)
