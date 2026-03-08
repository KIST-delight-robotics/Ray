"""Exceptions for the generation module."""

from voice_pipeline.core.exceptions import PipelineError


class GenerationError(PipelineError):
    """Base exception for generation-related errors."""


class SpeechGeneratorError(GenerationError):
    """Error raised by SpeechGenerator operations."""
