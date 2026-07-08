"""Configuration loading for the Matter LED package.

Config lives in ``config.toml`` next to this file and is parsed with the stdlib
``tomllib`` (Python 3.11+). No new dependency is added to the project.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from matter_platform_led.exceptions import MatterError

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class MatterConfig:
    """Resolved configuration for one Matter On/Off light.

    Attributes:
        backend: Which backend to build — ``"mock"`` or ``"chip_tool"``.
        node_id: Node id assigned at commissioning (e.g. ``"0x60"``). Identifies
            the device to the controller after pairing.
        endpoint_id: Endpoint carrying the On/Off cluster (lights are usually 1).
        pairing_code: 11-digit manual code or QR payload from the device label.
        wifi_ssid: SSID handed to the bulb during WiFi (BLE) commissioning.
            Empty string → treat as on-network commissioning (no WiFi creds sent).
        wifi_password: Password for ``wifi_ssid``.
        chip_tool_bin: ``chip-tool`` executable name/path.
        paa_trust_store_path: PAA cert dir for attestation of commercial devices.
            Empty string → not passed.
        bypass_attestation: Dev-only escape hatch to skip attestation verification.
        command_timeout_sec: Per-command subprocess timeout for the chip-tool backend.
    """

    backend: str
    node_id: str
    endpoint_id: int
    pairing_code: str
    wifi_ssid: str
    wifi_password: str
    chip_tool_bin: str
    paa_trust_store_path: str
    bypass_attestation: bool
    command_timeout_sec: float


def load_config(path: str | Path | None = None) -> MatterConfig:
    """Load :class:`MatterConfig` from a TOML file.

    Args:
        path: Path to a config TOML. Defaults to ``config.toml`` beside this module.

    Returns:
        The resolved configuration.

    Raises:
        MatterError: The file is missing or malformed.
    """

    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise MatterError(f"config file not found: {cfg_path}")

    try:
        with cfg_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MatterError(f"failed to read config {cfg_path}: {exc}") from exc

    device = raw.get("device", {})
    wifi = raw.get("wifi", {})
    chip = raw.get("chip_tool", {})

    return MatterConfig(
        backend=str(raw.get("backend", "mock")).lower(),
        node_id=str(device.get("node_id", "0x60")),
        endpoint_id=int(device.get("endpoint_id", 1)),
        pairing_code=str(device.get("pairing_code", "")),
        wifi_ssid=str(wifi.get("ssid", "")),
        wifi_password=str(wifi.get("password", "")),
        chip_tool_bin=str(chip.get("bin", "chip-tool")),
        paa_trust_store_path=str(chip.get("paa_trust_store_path", "")),
        bypass_attestation=bool(chip.get("bypass_attestation", True)),
        command_timeout_sec=float(chip.get("command_timeout_sec", 30.0)),
    )
