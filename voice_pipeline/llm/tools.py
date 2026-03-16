"""Tool definitions for the LLM module.

Usage:
    from voice_pipeline.llm.tools import resolve_tools
    tool_defs = resolve_tools(["web_search"])
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("voice_pipeline.llm")

ToolDef = dict[str, Any]

WEB_SEARCH: ToolDef = {"type": "web_search"}

_TOOLS: dict[str, ToolDef] = {
    "web_search": WEB_SEARCH,
}


def resolve_tools(names: list[str]) -> list[ToolDef]:
    """Resolve tool names to API-ready tool definitions.

    Unknown names are logged as warnings and skipped.
    """
    resolved = []
    for name in names:
        if name in _TOOLS:
            resolved.append(_TOOLS[name])
        else:
            logger.warning("Unknown tool name '%s', skipping", name)
    return resolved
