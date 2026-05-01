"""Exceptions for the wakeword module."""

from voice_pipeline.core.exceptions import PipelineError


class WakewordError(PipelineError):
    """Error in wakeword detection."""
