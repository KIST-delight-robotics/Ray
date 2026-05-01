"""Exceptions for the audio module."""

from voice_pipeline.core.exceptions import PipelineError


class AudioInputError(PipelineError):
    """Error in audio input capture."""
