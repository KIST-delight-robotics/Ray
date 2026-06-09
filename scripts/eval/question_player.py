"""WAV playback via aplay for eval question delivery."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("eval.question_player")


class QuestionPlayer:
    """Plays WAV files through a specific ALSA output device."""

    def __init__(self, device: str = "default", beep_wav: str | None = None) -> None:
        self._device = device
        self._beep_wav = beep_wav

    def beep(self) -> bool:
        """Play the short session-start identification beep, if configured.

        Returns ``True`` if a beep was attempted (so the caller knows to drain
        the mic queue afterwards), ``False`` when no beep WAV is set. Failures
        are logged but never raised — a broken beep must not abort an eval run.
        """
        if not self._beep_wav:
            return False
        try:
            subprocess.run(
                ["aplay", "-D", self._device, self._beep_wav],
                check=True,
                capture_output=True,
            )
        except Exception:
            logger.warning("Beep playback failed", exc_info=True)
        return True

    def play(self, wav_path: str) -> None:
        """Play a WAV file. Blocks until playback finishes."""
        logger.info("Playing: %s (device=%s)", wav_path, self._device)
        try:
            subprocess.run(
                ["aplay", "-D", self._device, wav_path],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("aplay failed: %s", e.stderr.decode(errors="replace"))
            raise
