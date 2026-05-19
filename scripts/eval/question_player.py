"""WAV playback via aplay for eval question delivery."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("eval.question_player")


class QuestionPlayer:
    """Plays WAV files through a specific ALSA output device."""

    def __init__(self, device: str = "default") -> None:
        self._device = device

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
