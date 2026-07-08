"""Backend interface — the swappable Matter driver contract.

Everything above this line (the CLI, :class:`MatterLedController`, and later the
LLM tool call) depends ONLY on :class:`MatterLightBackend`. Everything below it
(mock / chip-tool / python-matter-server) is an implementation detail you can
swap without touching a single caller. That is the whole point of the abstraction:
you develop against ``mock`` today and flip to ``chip_tool`` when the bulb arrives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LightStatus:
    """Observed state of the Matter On/Off light.

    Attributes:
        on: ``True`` if the light is currently on, ``False`` if off.
        reachable: ``True`` if the backend could reach the device to read state.
            ``False`` means the value in ``on`` is a stale/best-effort guess.
    """

    on: bool
    reachable: bool = True


class MatterLightBackend(ABC):
    """Drives a single Matter On/Off light (one node, one endpoint).

    Implementations:
        * ``MockBackend`` — no hardware; keeps state in a file so the CLI can be
          verified end-to-end today.
        * ``ChipToolBackend`` — wraps the ``chip-tool`` CLI for a real WiFi bulb.
        * (future) ``PythonMatterServerBackend`` — persistent, low-latency.

    Contract: methods raise :class:`~matter_platform_led.exceptions.MatterError`
    subclasses on failure. Control commands (:meth:`turn_on` etc.) must raise
    :class:`~matter_platform_led.exceptions.MatterNotCommissionedError` if the
    device has never been commissioned.
    """

    @abstractmethod
    def commission(self, pairing_code: str, ssid: str | None, password: str | None) -> None:
        """Pair the bulb onto this controller's fabric (one-time setup).

        Args:
            pairing_code: 11-digit manual pairing code or QR payload from the device.
            ssid: WiFi SSID to hand to the bulb over BLE. ``None`` means the device
                is already on the IP network (on-network commissioning).
            password: WiFi password matching ``ssid`` (ignored when ``ssid`` is None).

        Raises:
            MatterCommissionError: Pairing failed.
        """

    @abstractmethod
    def turn_on(self) -> None:
        """Send the On command to the light.

        Raises:
            MatterNotCommissionedError: Device was never commissioned.
            MatterCommandError: The command failed at the backend.
        """

    @abstractmethod
    def turn_off(self) -> None:
        """Send the Off command to the light.

        Raises:
            MatterNotCommissionedError: Device was never commissioned.
            MatterCommandError: The command failed at the backend.
        """

    @abstractmethod
    def toggle(self) -> None:
        """Send the Toggle command to the light.

        Raises:
            MatterNotCommissionedError: Device was never commissioned.
            MatterCommandError: The command failed at the backend.
        """

    @abstractmethod
    def read_status(self) -> LightStatus:
        """Read the current On/Off state from the device.

        Returns:
            LightStatus: Current state; ``reachable=False`` if it could not be read.

        Raises:
            MatterNotCommissionedError: Device was never commissioned.
        """

    @abstractmethod
    def open_commissioning_window(self, timeout_sec: int = 180) -> str:
        """Open a commissioning window so a second admin (e.g. a phone) can pair.

        Matter allows several controllers to share one device (multi-admin). This
        opens a temporary window on the device we already commissioned and returns
        a fresh setup payload to type/scan in the phone's Home app.

        Args:
            timeout_sec: How long the window stays open before auto-closing.

        Returns:
            A QR setup payload ("MT:...") or manual pairing code for the 2nd admin.

        Raises:
            MatterNotCommissionedError: We never commissioned the device first.
            MatterCommandError: Opening the window failed.
        """

    def close(self) -> None:  # noqa: B027 — intentional optional hook, not all backends need cleanup
        """Release any resources (sessions, sockets). Default: no-op."""
