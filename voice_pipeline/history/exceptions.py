"""History module exceptions."""

from voice_pipeline.core.exceptions import PipelineError


class HistoryError(PipelineError):
    """Raised when a history operation is attempted without an active session."""
