"""MatterLedController — the middle box.

This is the layer the rest of the world talks to. A command has already been
decided (by a human at the CLI today, by the LLM later); this class turns it
into a real Matter signal through whichever backend the config selected.

    LLM / CLI  ──▶  MatterLedController.off()  ──▶  backend.turn_off()  ──▶  bulb
                   └──────── this file ───────┘   └──── swappable ────┘

Swapping backends never changes callers: they only ever see this class.
"""

from __future__ import annotations

import logging

from matter_platform_led.config import MatterConfig, load_config
from matter_platform_led.exceptions import MatterError
from matter_platform_led.interface import LightStatus, MatterLightBackend

logger = logging.getLogger("matter_platform_led.controller")


def build_backend(config: MatterConfig) -> MatterLightBackend:
    """Construct the backend selected by ``config.backend``.

    Args:
        config: Resolved configuration.

    Returns:
        A ready-to-use backend instance.

    Raises:
        MatterError: ``config.backend`` names an unknown backend.
    """

    if config.backend == "mock":
        # Imported lazily so the mock path has zero import cost concerns and the
        # chip-tool path never imports mock plumbing (and vice-versa).
        from matter_platform_led.mock_backend import MockBackend

        return MockBackend(node_id=config.node_id, endpoint_id=config.endpoint_id)

    if config.backend == "chip_tool":
        from matter_platform_led.chip_tool_backend import ChipToolBackend

        return ChipToolBackend(
            node_id=config.node_id,
            endpoint_id=config.endpoint_id,
            chip_tool_bin=config.chip_tool_bin,
            paa_trust_store_path=config.paa_trust_store_path,
            bypass_attestation=config.bypass_attestation,
            command_timeout_sec=config.command_timeout_sec,
        )

    raise MatterError(f"unknown backend '{config.backend}' (expected 'mock' or 'chip_tool')")


class MatterLedController:
    """High-level on/off control over a single Matter light.

    Construct via :meth:`from_config` (reads ``config.toml`` and picks the
    backend) or pass a backend directly for tests.
    """

    def __init__(self, backend: MatterLightBackend, config: MatterConfig) -> None:
        self._backend = backend
        self._config = config

    @classmethod
    def from_config(
        cls,
        config_path: str | None = None,
        backend_override: str | None = None,
    ) -> MatterLedController:
        """Build a controller from a config file.

        Args:
            config_path: Path to ``config.toml``; defaults to the packaged one.
            backend_override: If set (``"mock"``/``"chip_tool"``), overrides the
                ``backend`` field in the file. Lets the CLI ``--backend`` flag win.

        Returns:
            A ready controller.
        """

        config = load_config(config_path)
        if backend_override is not None:
            config = MatterConfig(**{**config.__dict__, "backend": backend_override.lower()})
        backend = build_backend(config)
        logger.info(
            "controller ready: backend=%s node=%s endpoint=%s",
            config.backend,
            config.node_id,
            config.endpoint_id,
        )
        return cls(backend=backend, config=config)

    # -- the command surface (what the LLM tool will call later) -----------

    def commission(self) -> None:
        """Pair the configured device using the config's pairing code / WiFi creds."""
        ssid = self._config.wifi_ssid or None
        self._backend.commission(self._config.pairing_code, ssid, self._config.wifi_password)

    def on(self) -> None:
        """Turn the light on."""
        self._backend.turn_on()

    def off(self) -> None:
        """Turn the light off."""
        self._backend.turn_off()

    def toggle(self) -> None:
        """Toggle the light."""
        self._backend.toggle()

    def status(self) -> LightStatus:
        """Read the current on/off state."""
        return self._backend.read_status()

    def share(self, timeout_sec: int = 180) -> str:
        """Open a commissioning window for a 2nd admin (phone). Returns a setup code."""
        return self._backend.open_commissioning_window(timeout_sec)

    def close(self) -> None:
        """Release backend resources."""
        self._backend.close()

    def __enter__(self) -> MatterLedController:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
