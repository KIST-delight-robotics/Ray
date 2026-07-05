"""Main entry point for the voice pipeline.

Usage:
    uv run ray              # via project.scripts entry point
    uv run python -m voice_pipeline  # as module
"""

from __future__ import annotations

import ctypes
import logging
import os

# Suppress ALSA/JACK noise during PyAudio initialization.
# Restored after AudioInput construction so runtime errors are still visible.
_alsa_error_handler = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)(lambda *_: None)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_alsa_error_handler)
except Exception:
    _asound = None
import queue
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.core.types import AudioFrame, CppEventType, LEDState, SystemMode
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.memory.writer import MemoryWriter
from voice_pipeline.tts.greeting_audio import ensure_greeting_audio
from voice_pipeline.wakeword.wakeword import WakewordDetector
from voice_pipeline.wiring import build_components

logger = logging.getLogger("voice_pipeline")

_GREETING_TIMEOUT_SEC = 10.0
_FAREWELL_TIMEOUT_SEC = 10.0
_FRAME_TIMEOUT_SEC = 0.1
_POLL_INTERVAL_SEC = 0.05
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flush_bridge_events(bridge: CppBridge) -> None:
    try:
        while bridge.poll_event() is not None:
            pass
    except Exception:
        logger.debug("Error flushing bridge events", exc_info=True)


def _drain_audio_queue(audio_queue: queue.Queue[AudioFrame]) -> None:
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def _wait_playback(
    bridge: CppBridge,
    shutdown_event: threading.Event,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while not shutdown_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            event = bridge.poll_event()
        except Exception:
            logger.warning("Bridge poll_event error during playback wait", exc_info=True)
            break
        if event is not None and event.event_type == CppEventType.PLAYBACK_COMPLETE:
            break
        time.sleep(min(_POLL_INTERVAL_SEC, remaining))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the voice pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-40s %(levelname)-7s %(message)s",
    )
    for entry in os.environ.get("LOG_LEVEL", "").split(","):
        entry = entry.strip()
        if "=" in entry:
            name, level = entry.split("=", 1)
            logging.getLogger(name.strip()).setLevel(level.strip().upper())

    # --- Shared component graph (production defaults: data/ray.db, LED via env) ---
    components = build_components()

    if _asound is not None:
        _asound.snd_lib_error_set_handler(None)

    audio_queue = components.audio_queue
    audio_input = components.audio_input
    bridge = components.bridge
    led = components.led
    shutdown_event = components.shutdown_event

    # --- Production-only pieces: wakeword, memory writer, greeting audio ---
    wakeword = WakewordDetector(language_code=components.language_code, vad_model=components.silero_vad_model)
    write_llm = OpenAILLM(model="gpt-4o-mini", temperature=0.0, reasoning_effort=None, max_tokens=4096, tools=[])
    memory_writer = MemoryWriter(
        components.memory_storage,
        components.vector_index,
        components.embedder,
        write_llm,
        components.token_counter,
    )
    write_executor = ThreadPoolExecutor(max_workers=1)
    greeting_paths = ensure_greeting_audio(components.tts)

    # --- Signal handling ---
    def _handle_signal(*_: object) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_signal)

    # --- Mode loop ---
    mode = SystemMode.SLEEP
    current_history = None
    current_session_id: str | None = None
    session_started_at: str | None = None
    session_started = False

    logger.info("Pipeline starting")
    bridge.connect()
    audio_input.start()
    try:
        while not shutdown_event.is_set():
            # ---- SLEEP ----
            if mode == SystemMode.SLEEP:
                led.set_state(LEDState.SLEEPING)
                while not shutdown_event.is_set():
                    try:
                        frame = audio_queue.get(timeout=_FRAME_TIMEOUT_SEC)
                    except queue.Empty:
                        if audio_input.error is not None:
                            raise audio_input.error from None
                        continue
                    if wakeword.feed_audio(frame):
                        logger.info("Wakeword detected — transitioning to GREETING")
                        mode = SystemMode.GREETING
                        break

            # ---- GREETING ----
            elif mode == SystemMode.GREETING:
                try:
                    bridge.connect()
                except Exception:
                    logger.error("Bridge connect failed — returning to SLEEP", exc_info=True)
                    wakeword.reset()
                    mode = SystemMode.SLEEP
                    continue

                _flush_bridge_events(bridge)
                led.set_state(LEDState.IDLE)

                try:
                    bridge.send_play_file(greeting_paths.greeting)
                except Exception:
                    logger.warning("Failed to send greeting", exc_info=True)

                _wait_playback(bridge, shutdown_event, _GREETING_TIMEOUT_SEC)
                mode = SystemMode.ACTIVE

            # ---- ACTIVE ----
            elif mode == SystemMode.ACTIVE:
                _drain_audio_queue(audio_queue)
                try:
                    session = components.create_session()
                except Exception:
                    logger.error("Session factory failed", exc_info=True)
                    wakeword.reset()
                    mode = SystemMode.SLEEP
                    continue

                current_history = session.history
                current_session_id = session.session_id
                session_started_at = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
                session_started = True

                current_history.new_session(session.session_id)
                logger.info("Session started: %s", session.session_id)

                try:
                    session.session_loop.run()
                except Exception:
                    logger.error("SessionLoop run failed", exc_info=True)

                mode = SystemMode.FAREWELL

            # ---- FAREWELL ----
            elif mode == SystemMode.FAREWELL:
                _flush_bridge_events(bridge)

                try:
                    bridge.send_play_file(greeting_paths.farewell)
                except Exception:
                    logger.warning("Failed to send farewell", exc_info=True)

                _wait_playback(bridge, shutdown_event, _FAREWELL_TIMEOUT_SEC)

                if session_started and current_history is not None:
                    try:
                        current_history.save()
                    except Exception:
                        logger.warning("History save error in farewell", exc_info=True)
                    if current_session_id and session_started_at:
                        try:
                            write_executor.submit(memory_writer.process_session, current_session_id, session_started_at)
                        except Exception:
                            logger.warning("on_session_end callback failed", exc_info=True)

                session_started = False
                _drain_audio_queue(audio_queue)
                led.set_state(LEDState.SLEEPING)
                current_history = None
                current_session_id = None
                session_started_at = None
                wakeword.reset()
                mode = SystemMode.SLEEP
                logger.info("Session ended — returning to SLEEP")

    finally:
        if session_started and current_history is not None:
            try:
                current_history.save()
            except Exception:
                logger.warning("History save error on shutdown", exc_info=True)
            if current_session_id and session_started_at:
                try:
                    write_executor.submit(memory_writer.process_session, current_session_id, session_started_at)
                except Exception:
                    logger.warning("on_session_end callback failed on shutdown", exc_info=True)

        audio_input.stop()
        bridge.disconnect()
        components.stop_async()
        write_executor.shutdown(wait=True)
        components.executor.shutdown(wait=True)
        components.asr.stop()
        wakeword.close()
        led.close()
        components.memory_storage.close()
        components.trace_store.close()
        logging.getLogger("openai._base_client").removeHandler(components.retry_handler)
        components.call_store.close()
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
