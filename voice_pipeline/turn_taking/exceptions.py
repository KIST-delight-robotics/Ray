"""Exceptions for the turn-taking module."""

from voice_pipeline.core.exceptions import PipelineError


class TurnTakingError(PipelineError):
    """Base exception for turn-taking errors."""


class VAPError(TurnTakingError):
    """Error in the VAP model wrapper."""


class TurnGPTError(TurnTakingError):
    """Error in the TurnGPT model wrapper."""


class TurnDetectorError(TurnTakingError):
    """Error in the combined turn detector."""
