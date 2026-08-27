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
import os
import socket
import time

logger = logging.getLogger("voice_pipeline.led")

CONTROL_SOCK = "/run/os-led.sock"
_CONNECT_TIMEOUT_S = 1.0
_GRANT_TIMEOUT_S = 3.0
# The daemon is a system service (After=multi-user.target, ~+45 s at boot) while
# RAY runs under the user manager and can be ready earlier (~+40 s after the boot
# speed-ups). Falling through to standalone in that window means both processes
# drive the same WS2812 line and it flickers — so wait for the socket instead.
# How long depends on whether the daemon exists on this machine at all:
#   installed (robot Pi)  → wait long enough to cover its boot-time start
#   not installed (dev box) → fail fast, standalone is the intended mode there
# Unit file presence is the install marker; it is world-readable, so this works
# from the user session without systemd access.
_DAEMON_UNIT = "/etc/systemd/system/os-led-display.service"
_CONNECT_RETRY_S = 5.0
_CONNECT_RETRY_INSTALLED_S = 30.0
_CONNECT_RETRY_INTERVAL_S = 0.25


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
        conn = self._connect_with_retry()
        if conn is None:
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

    def _connect_with_retry(self) -> socket.socket | None:
        """Connect to the arbiter socket, retrying while it is merely absent.

        Returns None once the retry window is exhausted (daemon not installed or
        not up), which the caller treats as standalone mode.
        """
        installed = os.path.exists(_DAEMON_UNIT)
        window = _CONNECT_RETRY_INSTALLED_S if installed else _CONNECT_RETRY_S
        deadline = time.monotonic() + window
        attempt = 0
        while True:
            attempt += 1
            try:
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.settimeout(_CONNECT_TIMEOUT_S)
                conn.connect(self._sock_path)
                if attempt > 1:
                    logger.info("OS_LED arbiter reached after %d attempts", attempt)
                return conn
            except (FileNotFoundError, ConnectionRefusedError):
                if attempt == 1 and installed:
                    logger.info("OS_LED daemon installed but socket absent — waiting up to %.0fs", window)
                if time.monotonic() >= deadline:
                    if installed:
                        logger.warning(
                            "OS_LED daemon installed but never came up within %.0fs — "
                            "driving strip standalone (expect contention if it starts later)",
                            window,
                        )
                    else:
                        logger.info("OS_LED arbiter not present — driving strip standalone")
                    return None
                time.sleep(_CONNECT_RETRY_INTERVAL_S)
            except OSError as exc:
                logger.warning("OS_LED arbiter connect failed (%s) — standalone", exc)
                return None

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
