"""chip-tool backend — drives a real Matter WiFi bulb via the ``chip-tool`` CLI.

This wraps the official Matter reference controller (``chip-tool`` from the
connectedhomeip project) with ``subprocess``. It is the fastest path to
verifying a real commercial bulb: install chip-tool, commission once, then
each on/off is a single command.

Trade-off (documented in README): every command spawns a fresh process and
re-establishes a secure (CASE) session, so latency is ~1-3 s per command and
state reads are parsed from text output. For low-latency production, add a
``PythonMatterServerBackend`` later — the interface stays identical.

NOTE: This backend has not been exercised against real hardware yet (no bulb
available). The command shapes below follow the chip-tool docs; verify against
your device per the README "hardware bring-up" checklist.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

from matter_platform_led.exceptions import (
    MatterCommandError,
    MatterCommissionError,
    MatterError,
)
from matter_platform_led.interface import LightStatus, MatterLightBackend

logger = logging.getLogger("matter_platform_led.chip_tool")

# Matches the OnOff attribute in chip-tool read output, tolerant of format drift:
#   "OnOff: TRUE", "OnOff = false", "OnOff: 1", etc.
_ONOFF_RE = re.compile(r"onoff\s*[:=]\s*(true|false|0x0*1|0x0*0|[01])\b", re.IGNORECASE)

# Setup payloads printed by `pairing open-commissioning-window` (for multi-admin).
_QR_RE = re.compile(r"SetupQRCode:\s*\[?(MT:[^\]\s]+)\]?", re.IGNORECASE)
_MANUAL_RE = re.compile(r"Manual pairing code:\s*\[?(\d{11,})\]?", re.IGNORECASE)


class ChipToolBackend(MatterLightBackend):
    """Subprocess wrapper around ``chip-tool`` for one On/Off light."""

    def __init__(
        self,
        node_id: str,
        endpoint_id: int,
        *,
        chip_tool_bin: str = "chip-tool",
        paa_trust_store_path: str = "",
        bypass_attestation: bool = True,
        command_timeout_sec: float = 30.0,
    ) -> None:
        self._node_id = node_id
        self._endpoint_id = str(endpoint_id)
        self._bin = chip_tool_bin
        self._paa = paa_trust_store_path
        self._bypass = bypass_attestation
        self._timeout = command_timeout_sec

    # -- process plumbing --------------------------------------------------

    def _run(self, args: list[str]) -> str:
        if shutil.which(self._bin) is None:
            raise MatterError(
                f"'{self._bin}' not found on PATH. Install it (e.g. `sudo snap install chip-tool`) "
                "or set [chip_tool].bin in config.toml."
            )
        cmd = [self._bin, *args]
        logger.info("chip-tool: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MatterCommandError(f"chip-tool timed out after {self._timeout}s: {' '.join(cmd)}") from exc

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise MatterCommandError(f"chip-tool exited {proc.returncode}: {tail}")
        return proc.stdout

    def _attestation_flags(self) -> list[str]:
        flags: list[str] = []
        if self._paa:
            flags += ["--paa-trust-store-path", self._paa]
        if self._bypass:
            flags += ["--bypass-attestation-verifier", "true"]
        return flags

    # -- backend contract --------------------------------------------------

    def commission(self, pairing_code: str, ssid: str | None, password: str | None) -> None:
        if not pairing_code:
            raise MatterCommissionError("pairing_code is empty — set [device].pairing_code in config.toml")

        if ssid:
            # WiFi device not yet on the network: hand it WiFi creds over BLE.
            args = ["pairing", "code-wifi", self._node_id, ssid, password or "", pairing_code]
        else:
            # Device already reachable on the IP network.
            args = ["pairing", "code", self._node_id, pairing_code]
        args += self._attestation_flags()

        try:
            self._run(args)
        except MatterError as exc:
            raise MatterCommissionError(str(exc)) from exc
        logger.info("commissioned node=%s (%s)", self._node_id, "wifi" if ssid else "on-network")

    def turn_on(self) -> None:
        self._run(["onoff", "on", self._node_id, self._endpoint_id])

    def turn_off(self) -> None:
        self._run(["onoff", "off", self._node_id, self._endpoint_id])

    def toggle(self) -> None:
        self._run(["onoff", "toggle", self._node_id, self._endpoint_id])

    def open_commissioning_window(self, timeout_sec: int = 180) -> str:
        # option 1 = Enhanced Commissioning Method (fresh dynamic passcode),
        # iteration 1000 (PBKDF), discriminator 3840 (matches the example device).
        out = self._run(
            [
                "pairing",
                "open-commissioning-window",
                self._node_id,
                "1",
                str(timeout_sec),
                "1000",
                "3840",
            ]
        )
        qr = _QR_RE.search(out)
        if qr:
            return qr.group(1)
        manual = _MANUAL_RE.search(out)
        if manual:
            return manual.group(1)
        raise MatterCommandError("could not parse setup payload from open-commissioning-window output")

    def read_status(self) -> LightStatus:
        out = self._run(["onoff", "read", "on-off", self._node_id, self._endpoint_id])
        matches = _ONOFF_RE.findall(out)
        if not matches:
            logger.warning("could not parse OnOff from chip-tool output; treating as unreachable")
            return LightStatus(on=False, reachable=False)
        value = matches[-1].lower()
        is_on = value in {"true", "1", "0x01", "0x1"}
        return LightStatus(on=is_on, reachable=True)
