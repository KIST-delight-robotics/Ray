"""TTS module exceptions."""

from voice_pipeline.core.exceptions import PipelineError


class TTSError(PipelineError):
    """Raised when a TTS operation fails."""
