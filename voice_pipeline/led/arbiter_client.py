"""Client for the OS_LED ownership arbiter (``os_led_display.py`` daemon).

The OS rainbow daemon owns the shared 24-LED WS2812 strip on ``/dev/spidev0.0``
and shows a rainbow while the Pi is idle. Before RAY drives the same strip it
must borrow it: connecting to the arbiter socket makes the daemon fade the
rainbow out and stop writing SPI, so the two never write the bus at once.

Holding the connection open = holding the token. Releasing (or crashing, which
drops the socket) makes the daemon fade the rainbow back in. If the daemon is
not running (socket absent), every call is a no-op and RAY drives the strip
standalone.
"""

from __future__ import annotations

import contextlib
import logging
import socket

logger = logging.getLogger("voice_pipeline.led")

CONTROL_SOCK = "/run/os-led.sock"
_CONNECT_TIMEOUT_S = 1.0
_GRANT_TIMEOUT_S = 3.0


class OSLedArbiterClient:
    """Borrows the WS2812 strip from the OS_LED rainbow daemon."""

    def __init__(self, sock_path: str = CONTROL_SOCK) -> None:
        self._sock_path = sock_path
        self._conn: socket.socket | None = None

    def acquire(self) -> None:
        """Borrow the strip from the rainbow daemon.

        Blocks until the daemon has faded out and stopped driving SPI, so RAY
        can take over without interleaved frames. A missing/unreachable daemon
        is treated as "standalone" — RAY proceeds to drive the strip directly.
        """
        if self._conn is not None:
            return
        try:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(_CONNECT_TIMEOUT_S)
            conn.connect(self._sock_path)
        except (FileNotFoundError, ConnectionRefusedError):
            logger.info("OS_LED arbiter not present — driving strip standalone")
            return
        except OSError as exc:
            logger.warning("OS_LED arbiter connect failed (%s) — standalone", exc)
            return

        try:
            conn.sendall(b"ACQUIRE\n")
            conn.settimeout(_GRANT_TIMEOUT_S)
            resp = conn.recv(32)
        except OSError as exc:
            logger.warning("OS_LED arbiter handshake failed (%s) — standalone", exc)
            conn.close()
            return

        if b"GRANTED" not in resp:
            logger.warning("OS_LED arbiter did not grant — proceeding anyway")
        conn.settimeout(None)
        self._conn = conn
        logger.info("OS_LED strip acquired from rainbow daemon")

    def release(self) -> None:
        """Return the strip — the daemon fades the rainbow back in."""
        if self._conn is None:
            return
        with contextlib.suppress(OSError):
            self._conn.sendall(b"RELEASE\n")
        with contextlib.suppress(OSError):
            self._conn.close()
        self._conn = None
        logger.info("OS_LED strip released back to rainbow daemon")
