"""Orchestrator module exceptions."""

from voice_pipeline.core.exceptions import PipelineError


class OrchestratorError(PipelineError):
    """Raised when the orchestrator encounters an unrecoverable error."""
