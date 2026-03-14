"""Turn-taking module: VAP, TurnGPT, and TurnDetector."""

from voice_pipeline.turn_taking.exceptions import (
    TurnDetectorError,
    TurnGPTError,
    TurnTakingError,
    VAPError,
)

__all__ = [
    "AsyncTurnGPT",
    "AsyncVAP",
    "SyncTurnGPTAdapter",
    "TurnDetector",
    "TurnDetectorError",
    "TurnGPTError",
    "TurnGPTWrapper",
    "TurnTakingError",
    "VAPError",
    "VAPWrapper",
]


def __getattr__(name: str):  # noqa: N807
    if name == "VAPWrapper":
        from voice_pipeline.turn_taking.vap import VAPWrapper

        return VAPWrapper
    if name == "TurnGPTWrapper":
        from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

        return TurnGPTWrapper
    if name == "TurnDetector":
        from voice_pipeline.turn_taking.turn_detector import TurnDetector

        return TurnDetector
    if name == "AsyncVAP":
        from voice_pipeline.turn_taking.async_vap import AsyncVAP

        return AsyncVAP
    if name == "AsyncTurnGPT":
        from voice_pipeline.turn_taking.async_turngpt import AsyncTurnGPT

        return AsyncTurnGPT
    if name == "SyncTurnGPTAdapter":
        from voice_pipeline.turn_taking.async_turngpt import SyncTurnGPTAdapter

        return SyncTurnGPTAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
