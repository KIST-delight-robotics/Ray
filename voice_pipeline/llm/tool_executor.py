"""Executes LLM function-tool calls and builds follow-up message items.

Currently handles ``control_light`` by driving an :class:`ILightControl`.
The concrete light controller (e.g. the Matter controller) is adapted to
that interface at wiring time, so this module stays decoupled from any
specific smart-home stack.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from voice_pipeline.core.interfaces import ILightControl, IToolExecutor
from voice_pipeline.core.types import ToolCall

logger = logging.getLogger("voice_pipeline.llm")


class ToolExecutor(IToolExecutor):
    """Runs tool calls and returns ``function_call`` / ``function_call_output`` items."""

    def __init__(self, light: ILightControl | None = None) -> None:
        """Initialize the executor.

        Args:
            light: Light controller for ``control_light``. ``None`` disables
                light control (the tool reports it is unavailable).
        """
        self._light = light

    def resolve(self, tool_calls: tuple[ToolCall, ...]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for tc in tool_calls:
            output = self._dispatch(tc)
            # Echo the call, then its result, so the follow-up turn has full context.
            items.append({"type": "function_call", "call_id": tc.call_id, "name": tc.name, "arguments": tc.arguments})
            items.append({"type": "function_call_output", "call_id": tc.call_id, "output": output})
        return items

    def _dispatch(self, tc: ToolCall) -> str:
        """Run one tool call, returning a short English result string for the LLM."""
        if tc.name != "control_light":
            logger.warning("Unknown tool call '%s'", tc.name)
            return f"Error: unknown tool '{tc.name}'."
        return self._control_light(tc.arguments)

    def _control_light(self, arguments: str) -> str:
        if self._light is None:
            return "Error: the light is not available right now."
        try:
            action = str(json.loads(arguments or "{}").get("action", "")).lower()
        except (json.JSONDecodeError, AttributeError):
            return "Error: could not understand the light command."

        try:
            if action == "on":
                self._light.on()
                return "The light is now on."
            if action == "off":
                self._light.off()
                return "The light is now off."
            if action == "toggle":
                self._light.toggle()
                return "The light has been toggled."
        except Exception:
            logger.warning("Light control failed for action %r", action, exc_info=True)
            return "Error: failed to control the light."
        return f"Error: unsupported action '{action}'."
