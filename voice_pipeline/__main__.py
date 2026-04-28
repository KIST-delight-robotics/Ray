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
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.wakeword import WakewordDetector
from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.context.context_builder import ContextBuilder
from voice_pipeline.context.formatters import (
    format_raw_transcript_block,
    format_session_summary_block,
)
from voice_pipeline.core.types import AudioFrame, CppEventType, LEDState, SystemMode
from voice_pipeline.embedding.embedder import create_embedder
from voice_pipeline.generation.speech_generator import SpeechGenerator
from voice_pipeline.history.conversation_history import ConversationHistory
from voice_pipeline.history.storage_backend import create_storage_backend
from voice_pipeline.led.led_controller import LEDController
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.llm.token_counter import create_token_counter
from voice_pipeline.llm.tools import get_tools_token_cost
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import _DEFAULT_DB_PATH, _DEFAULT_DIMENSION, SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.memory.writer import MemoryWriter
from voice_pipeline.session_loop import SessionComponents, SessionLoop
from voice_pipeline.trace.trace_store import SQLiteTraceStore
from voice_pipeline.tts.greeting_audio import ensure_greeting_audio
from voice_pipeline.tts.tts import OpenAITTS
from voice_pipeline.tts.utterance_truncator import TimestampTruncator
from voice_pipeline.turn_taking.async_turngpt import AsyncTurnGPT
from voice_pipeline.turn_taking.async_vap import AsyncVAP
from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper
from voice_pipeline.turn_taking.turn_detector import TurnDetector
from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

logger = logging.getLogger("voice_pipeline")

_AUDIO_QUEUE_SIZE = 300
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

    language_code = "en-US"

    # --- Process-level singletons ---
    asr = GoogleCloudASR(language_code=language_code)
    llm = OpenAILLM(model="gpt-5.4", temperature=0.7, max_tokens=256, tools=["web_search"])
    tts = OpenAITTS()
    vap = MaAIVAPWrapper(tts.output_sample_rate)
    turngpt = TurnGPTWrapper()
    bridge = CppBridge()
    wakeword = WakewordDetector(language_code=language_code)
    led = LEDController()
    storage = create_storage_backend()
    executor = ThreadPoolExecutor(max_workers=SpeechGenerator.MAX_WORKERS)
    token_counter = create_token_counter(llm.model)
    tools_token_cost = get_tools_token_cost(llm.tools)

    # --- Memory system ---
    embedder = create_embedder(expected_dimension=_DEFAULT_DIMENSION)
    memory_storage = SQLiteMemoryStorage(_DEFAULT_DB_PATH)
    trace_store = SQLiteTraceStore(_DEFAULT_DB_PATH)
    vector_index = NumpyVectorIndex()
    ids, vectors = memory_storage.load_all_embeddings()
    if ids:
        vector_index.load(ids, vectors)
    write_llm = OpenAILLM(model="gpt-4o-mini", temperature=0.0, max_tokens=4096, tools=[])
    memory_writer = MemoryWriter(memory_storage, vector_index, embedder, write_llm, token_counter)
    write_executor = ThreadPoolExecutor(max_workers=1)

    # --- Pre-generate greeting/farewell audio ---
    greeting_paths = ensure_greeting_audio(tts)

    # --- Audio queue + input ---
    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=_AUDIO_QUEUE_SIZE)
    audio_input = AudioInput(audio_queue)

    if _asound is not None:
        _asound.snd_lib_error_set_handler(None)

    # --- Session factory ---
    prev_async: list[AsyncVAP | AsyncTurnGPT] = []
    shutdown_event = threading.Event()

    def session_factory() -> SessionComponents:
        for wrapper in prev_async:
            wrapper.stop()
        prev_async.clear()

        vap.reset()
        turngpt.reset()

        session_id = str(uuid.uuid4())

        profiles = memory_storage.get_all_profiles()
        recent = storage.get_recent_sessions(ConversationHistory.PREVIOUS_SESSION_COUNT, exclude_session_id=session_id)
        recent_session_ids = [s[0] for s in recent]
        session_episodes = memory_storage.get_episodes_by_session_ids(recent_session_ids)
        processed_ids = memory_storage.get_processed_session_ids(recent_session_ids)
        session_summaries: list[str] = []
        for sid, started_at, _ended_at in recent:
            episodes = session_episodes.get(sid, [])
            if episodes:
                session_summaries.append(format_session_summary_block(started_at, episodes))
            elif sid in processed_ids:
                continue
            else:
                utterances = memory_storage.get_utterances(sid)
                if utterances:
                    session_summaries.append(format_raw_transcript_block(started_at, utterances))
        exclude_session_ids = {session_id} | set(recent_session_ids)

        async_vap = AsyncVAP(vap)
        async_turngpt = AsyncTurnGPT(turngpt)
        prev_async.extend([async_vap, async_turngpt])

        history = ConversationHistory(storage, token_counter)
        retriever = MemoryRetriever(memory_storage, vector_index, embedder)
        context_builder = ContextBuilder(
            history,
            DEFAULT_SYSTEM_PROMPT,
            token_counter,
            tools_token_cost=tools_token_cost,
            profiles=profiles,
            session_summaries=session_summaries,
        )
        turn_detector = TurnDetector(async_vap, async_turngpt, embedder)
        generator = SpeechGenerator(
            context_builder,
            llm,
            tts,
            executor,
            retriever=retriever,
            history=history,
            exclude_session_ids=exclude_session_ids,
        )
        truncator = TimestampTruncator()
        session_loop = SessionLoop(
            asr=asr,
            turn_detector=turn_detector,
            speech_generator=generator,
            cpp_bridge=bridge,
            history=history,
            truncator=truncator,
            led=led,
            audio_queue=audio_queue,
            tts_sample_rate=tts.output_sample_rate,
            memory_storage=memory_storage,
            session_id=session_id,
            token_counter=token_counter,
            trace_store=trace_store,
            shutdown_event=shutdown_event,
        )
        return SessionComponents(session_loop=session_loop, history=history, session_id=session_id)

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
                    components = session_factory()
                except Exception:
                    logger.error("Session factory failed", exc_info=True)
                    mode = SystemMode.SLEEP
                    continue

                current_history = components.history
                current_session_id = components.session_id
                session_started_at = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
                session_started = True

                current_history.new_session(components.session_id)
                logger.info("Session started: %s", components.session_id)

                try:
                    components.session_loop.run()
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
        for wrapper in prev_async:
            wrapper.stop()
        prev_async.clear()
        write_executor.shutdown(wait=True)
        executor.shutdown(wait=True)
        asr.stop()
        wakeword.close()
        led.close()
        memory_storage.close()
        trace_store.close()
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
