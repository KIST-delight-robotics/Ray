"""ASR module exceptions."""

from voice_pipeline.core.exceptions import PipelineError


class ASRError(PipelineError):
    """Raised when an ASR operation fails."""
