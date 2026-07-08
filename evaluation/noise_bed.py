"""Continuous ambient noise-bed playback for the e2e SNR conditions.

Plays one bed *master* WAV in a **gapless** loop through ``aplay`` over a dmix
PCM, so the bed coexists with question playback on the same speaker. Each
condition (e.g. ``quiet``/``medium``/``loud``) is a per-condition **gain**
applied to that single master at playback time; a ``None`` gain means the bed is
off (room floor only). Keeping one master + recorded gains (in calibration.json)
instead of pre-scaled per-condition WAVs keeps the artifact fixed and makes a
level change a one-number edit — no re-rendering, no overwritten files.

Gapless looping is done by keeping one long-lived ``aplay`` reading raw PCM from
stdin while a writer thread feeds the (scaled) master PCM over and over. A naive
``while :; do aplay <file>; done`` restarts ``aplay`` each loop, dropping a brief
silent gap into the noise floor exactly where turn-detection might be listening.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
import wave

import numpy as np

logger = logging.getLogger("eval.noise_bed")

# Feed the pipe in small chunks (not one ~MB blob) so the writer rechecks the
# stop flag every fraction of a second — a single giant write blocks for the
# whole clip and pins the buffer lock, hanging stop().
_FEED_CHUNK = 8192


class NoiseBed:
    """Manages the looping noise-bed ``aplay`` subprocess and its level.

    Plays a single ``master`` WAV scaled by a per-condition gain, so the level is
    a recorded number rather than a separate baked file.
    """

    def __init__(self, device: str, master_path: str, gains: dict[str, float | None]) -> None:
        """``gains`` maps condition name → master gain (``None`` = bed off)."""
        self._device = device
        self._master_path = master_path
        self._gains = gains
        self._current: str | None = None
        self._proc: subprocess.Popen | None = None
        self._writer: threading.Thread | None = None
        self._stop = threading.Event()
        self._master: tuple[np.ndarray, int] | None = None  # (int16 samples, rate)

    def _load_master(self) -> tuple[np.ndarray, int]:
        if self._master is None:
            with wave.open(self._master_path) as w:
                if w.getsampwidth() != 2 or w.getnchannels() != 1:
                    raise ValueError(f"{self._master_path}: expected 16-bit mono bed WAV")
                samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                self._master = (samples, w.getframerate())
        return self._master

    def _scaled_pcm(self, gain: float) -> bytes:
        """Master × gain as int16 bytes, clipped to the valid range (gain>1 guard)."""
        samples, _ = self._load_master()
        scaled = np.clip(np.rint(samples.astype(np.float32) * gain), -32768, 32767)
        return scaled.astype(np.int16).tobytes()

    def set_level(self, condition: str) -> None:
        """Switch the bed to ``condition`` (no-op if already active)."""
        if condition == self._current:
            return
        if condition not in self._gains:
            raise KeyError(f"unknown noise condition: {condition}")
        self._stop_playback()
        self._current = condition
        gain = self._gains[condition]
        if gain is None:
            logger.info("Noise bed → %s (off)", condition)
            return

        _, rate = self._load_master()
        pcm = self._scaled_pcm(gain)
        self._stop.clear()
        self._proc = self._spawn(rate)
        # Pass proc explicitly so the writer never touches self._proc, which
        # _stop_playback may null concurrently.
        self._writer = threading.Thread(target=self._feed, args=(self._proc, pcm), daemon=True)
        self._writer.start()
        logger.info("Noise bed → %s (gain %.3f)", condition, gain)

    def _spawn(self, rate: int) -> subprocess.Popen:
        """Launch the looping player reading raw PCM from stdin (override in tests)."""
        return subprocess.Popen(
            ["aplay", "-q", "-D", self._device, "-t", "raw", "-f", "S16_LE", "-r", str(rate), "-c", "1", "-"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _feed(self, proc: subprocess.Popen, pcm: bytes) -> None:
        # Small chunked writes keep the stream continuous (paced by pipe
        # backpressure) while letting the loop notice _stop within one chunk.
        try:
            while not self._stop.is_set():
                for i in range(0, len(pcm), _FEED_CHUNK):
                    if self._stop.is_set():
                        return
                    proc.stdin.write(pcm[i : i + _FEED_CHUNK])
                proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass  # proc terminated mid-write during a level switch / stop

    def _stop_playback(self) -> None:
        self._stop.set()
        proc, writer = self._proc, self._writer
        self._proc, self._writer = None, None
        if proc is not None:
            # Terminate aplay FIRST: killing it breaks the pipe so any blocked
            # write() in the writer raises BrokenPipe and releases the buffer
            # lock. Closing stdin before terminate would deadlock on that lock.
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=2.0)
        if writer is not None:
            writer.join(timeout=2.0)
        if proc is not None and proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()

    def stop(self) -> None:
        """Stop the bed and release the device."""
        self._stop_playback()
        self._current = None
