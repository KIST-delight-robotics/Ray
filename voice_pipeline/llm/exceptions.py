"""LLM module exceptions."""

from voice_pipeline.core.exceptions import PipelineError


class LLMError(PipelineError):
    """Raised when an LLM operation fails."""
