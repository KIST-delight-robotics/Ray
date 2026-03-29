"""Tool definitions for the LLM module.

All tool configuration in one place. When adding a new tool:
1. Add an entry to _TOOL_REGISTRY with definition and token_cost.
2. Measure token_cost by comparing API input_tokens with/without the tool.

Usage:
    from voice_pipeline.llm.tools import resolve_tools, get_tools_token_cost
    tool_defs = resolve_tools(["web_search"])
    cost = get_tools_token_cost(["web_search"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("voice_pipeline.llm")

ToolDef = dict[str, Any]


@dataclass(frozen=True)
class _ToolEntry:
    """Tool definition + measured token cost."""

    definition: ToolDef
    token_cost: int  # measured via API input_tokens comparison


_TOOL_REGISTRY: dict[str, _ToolEntry] = {
    "web_search": _ToolEntry(
        definition={"type": "web_search"},
        token_cost=294,
    ),
}


def resolve_tools(names: list[str]) -> list[ToolDef]:
    """Resolve tool names to API-ready tool definitions.

    Unknown names are logged as warnings and skipped.
    """
    resolved = []
    for name in names:
        entry = _TOOL_REGISTRY.get(name)
        if entry is not None:
            resolved.append(entry.definition)
        else:
            logger.warning("Unknown tool name '%s', skipping", name)
    return resolved


def get_tools_token_cost(names: list[str]) -> int:
    """Return the total token cost of the given tools.

    Uses empirically measured values. Unknown tools are skipped.
    """
    total = 0
    for name in names:
        entry = _TOOL_REGISTRY.get(name)
        if entry is not None:
            total += entry.token_cost
    return total
