"""Exceptions for the Matter LED control package."""

from __future__ import annotations


class MatterError(Exception):
    """Base error for all Matter LED control failures."""


class MatterNotCommissionedError(MatterError):
    """A control command was issued before the device was commissioned/paired."""


class MatterCommissionError(MatterError):
    """Commissioning (pairing the bulb onto the fabric) failed."""


class MatterCommandError(MatterError):
    """A control command (on/off/toggle/read) failed at the backend."""
