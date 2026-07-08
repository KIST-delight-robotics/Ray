"""Turn-taking module: VAP, TurnGPT, and TurnDetector."""

from voice_pipeline.turn_taking.exceptions import (
    TurnDetectorError,
    TurnGPTError,
    TurnTakingError,
    VAPError,
)

__all__ = [
    "MaAIVAPModel",
    "SyncTurnGPTAdapter",
    "ThreadedTurnGPT",
    "ThreadedVAP",
    "TurnDetector",
    "TurnDetectorError",
    "TurnGPTError",
    "TurnGPTWrapper",
    "TurnTakingError",
    "VAPError",
]


def __getattr__(name: str):  # noqa: N807
    if name == "TurnGPTWrapper":
        from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

        return TurnGPTWrapper
    if name == "TurnDetector":
        from voice_pipeline.turn_taking.turn_detector import TurnDetector

        return TurnDetector
    if name == "MaAIVAPModel":
        from voice_pipeline.turn_taking.maai_vap import MaAIVAPModel

        return MaAIVAPModel
    if name == "ThreadedVAP":
        from voice_pipeline.turn_taking.threaded_vap import ThreadedVAP

        return ThreadedVAP
    if name == "ThreadedTurnGPT":
        from voice_pipeline.turn_taking.threaded_turngpt import ThreadedTurnGPT

        return ThreadedTurnGPT
    if name == "SyncTurnGPTAdapter":
        from voice_pipeline.turn_taking.threaded_turngpt import SyncTurnGPTAdapter

        return SyncTurnGPTAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
