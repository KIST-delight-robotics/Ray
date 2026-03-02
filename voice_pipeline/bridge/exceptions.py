"""Bridge-specific exceptions."""

from voice_pipeline.core.exceptions import PipelineError


class BridgeError(PipelineError):
    """Error in the C++ bridge WebSocket connection."""
