"""Turn-taking module: VAP, TurnGPT, and combined TurnDetector."""

from voice_pipeline.turn_taking.exceptions import (
    TurnDetectorError,
    TurnGPTError,
    TurnTakingError,
    VAPError,
)

__all__ = [
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
