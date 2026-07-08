"""WAV playback via aplay for eval question delivery."""

from __future__ import annotations

import logging
import subprocess
import time

logger = logging.getLogger("eval.question_player")


class QuestionPlayer:
    """Plays WAV files through a specific ALSA output device."""

    _BEEP_RETRY_DELAY_SEC = 0.3  # 일시적 device busy 대비 재시도 전 대기

    def __init__(self, device: str = "default", beep_wav: str | None = None) -> None:
        self._device = device
        self._beep_wav = beep_wav

    def beep(self) -> bool:
        """Play the short session-start identification beep, if configured.

        Returns ``True`` if a beep was attempted (so the caller knows to drain
        the mic queue afterwards), ``False`` when no beep WAV is set. Failures
        are retried once (transient device busy), then logged but never
        raised — a broken beep must not abort an eval run.
        """
        if not self._beep_wav:
            return False
        for attempt in (1, 2):
            try:
                subprocess.run(
                    ["aplay", "-D", self._device, self._beep_wav],
                    check=True,
                    capture_output=True,
                )
                return True
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode(errors="replace").strip()
                if attempt == 1:
                    logger.debug("Beep attempt 1 failed (%s), retrying", stderr)
                    time.sleep(self._BEEP_RETRY_DELAY_SEC)
                else:
                    logger.warning("Beep playback failed: %s", stderr)
            except Exception:
                logger.warning("Beep playback failed", exc_info=True)
                break
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
