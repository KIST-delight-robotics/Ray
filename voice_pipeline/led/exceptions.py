"""Exceptions for the LED module."""

from voice_pipeline.core.exceptions import PipelineError


class LEDError(PipelineError):
    """Error in LED controller initialization or operation."""
