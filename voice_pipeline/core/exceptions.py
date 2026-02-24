"""Base exception for the voice pipeline.

Module-specific exceptions inherit from PipelineError and are
defined in their respective module's exceptions.py.
"""


class PipelineError(Exception):
    """Base exception for all voice pipeline errors."""
