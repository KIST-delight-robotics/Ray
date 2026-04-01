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
import uuid
from concurrent.futures import ThreadPoolExecutor

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.wakeword import WakewordDetector
from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.context.context_builder import ContextBuilder
from voice_pipeline.core.config import PipelineConfig
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
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.memory.writer import MemoryWriter
from voice_pipeline.orchestrator.orchestrator import Orchestrator
from voice_pipeline.session.session_manager import SessionComponents, SessionManager
from voice_pipeline.similarity.similarity import create_similarity
from voice_pipeline.tts.greeting_audio import ensure_greeting_audio
from voice_pipeline.tts.tts import OpenAITTS
from voice_pipeline.tts.utterance_truncator import TimestampTruncator
from voice_pipeline.turn_taking.async_turngpt import AsyncTurnGPT
from voice_pipeline.turn_taking.async_vap import AsyncVAP
from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper
from voice_pipeline.turn_taking.turn_detector import TurnDetector
from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper


def main() -> None:
    """Launch the voice pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-40s %(levelname)-7s %(message)s",
    )
    # Per-module log level: LOG_LEVEL="voice_pipeline.turn_taking=DEBUG"
    for entry in os.environ.get("LOG_LEVEL", "").split(","):
        entry = entry.strip()
        if "=" in entry:
            name, level = entry.split("=", 1)
            logging.getLogger(name.strip()).setLevel(level.strip().upper())
    config = PipelineConfig()

    # --- Process-level singletons (expensive init, reused across sessions) ---
    asr = GoogleCloudASR(config.asr, config.audio)
    llm = OpenAILLM(config.llm)
    tts = OpenAITTS(config.tts)
    vap = MaAIVAPWrapper(config.maai_vap, config.audio, config.tts)
    turngpt = TurnGPTWrapper(config.turngpt)
    bridge = CppBridge(config.cpp_bridge)
    wakeword = WakewordDetector(config.wakeword, config.audio)
    led = LEDController(config.led)
    storage = create_storage_backend(config.history)
    similarity = create_similarity(config.similarity)
    executor = ThreadPoolExecutor(max_workers=config.speech_generator.max_workers)
    token_counter = create_token_counter(config.llm.model)
    tools_token_cost = get_tools_token_cost(config.llm.tools)

    # --- Memory system (process-level) ---
    embedder = create_embedder(
        config.memory.embedding_model,
        config.memory.embedding_backend,
        use_onnx=config.memory.use_onnx,
        expected_dimension=config.memory.embedding_dimension,
    )
    memory_storage = SQLiteMemoryStorage(config.memory)
    vector_index = NumpyVectorIndex()
    ids, vectors = memory_storage.load_all_embeddings()
    if ids:
        vector_index.load(ids, vectors)
    write_llm = OpenAILLM(config.memory.write_llm)
    memory_writer = MemoryWriter(
        memory_storage, vector_index, embedder, write_llm, config.memory, token_counter
    )
    write_executor = ThreadPoolExecutor(max_workers=1)

    # --- Pre-generate greeting/farewell audio ---
    greeting_paths = ensure_greeting_audio(tts, config.tts, config.greeting_audio)

    # --- Audio queue + input ---
    audio_queue = queue.Queue(maxsize=config.session.audio_queue_size)
    audio_input = AudioInput(audio_queue, config.audio, config.audio_input)

    # Restore default ALSA error handler
    if _asound is not None:
        _asound.snd_lib_error_set_handler(None)

    # --- Session factory: creates fresh per-session components ---
    prev_async: list[AsyncVAP | AsyncTurnGPT] = []

    def session_factory() -> SessionComponents:
        # Stop async wrappers from the previous session
        for wrapper in prev_async:
            wrapper.stop()
        prev_async.clear()

        vap.reset()
        turngpt.reset()

        session_id = str(uuid.uuid4())

        # Memory: load profiles + previous session summaries
        profiles = memory_storage.get_all_profiles()
        recent = storage.get_recent_sessions(
            config.history.previous_session_count, exclude_session_id=session_id
        )
        recent_session_ids = [s[0] for s in recent]
        session_episodes = memory_storage.get_episodes_by_session_ids(recent_session_ids)
        session_summaries = [(s[1], session_episodes.get(s[0], [])) for s in recent]
        exclude_session_ids = {session_id} | set(recent_session_ids)

        async_vap = AsyncVAP(vap)
        async_turngpt = AsyncTurnGPT(turngpt)
        prev_async.extend([async_vap, async_turngpt])

        history = ConversationHistory(storage, token_counter)
        retriever = MemoryRetriever(memory_storage, vector_index, embedder, config.memory)
        context_builder = ContextBuilder(
            history,
            config.history,
            DEFAULT_SYSTEM_PROMPT,
            token_counter,
            tools_token_cost=tools_token_cost,
            profiles=profiles,
            session_summaries=session_summaries,
        )
        turn_detector = TurnDetector(
            async_vap,
            async_turngpt,
            similarity,
            config.turn_detector,
            config.similarity,
            config.audio,
        )
        generator = SpeechGenerator(
            context_builder,
            llm,
            tts,
            config.speech_generator,
            executor,
            retriever=retriever,
            history=history,
            exclude_session_ids=exclude_session_ids,
        )
        truncator = TimestampTruncator()
        orchestrator = Orchestrator(
            asr=asr,
            turn_detector=turn_detector,
            speech_generator=generator,
            cpp_bridge=bridge,
            history=history,
            truncator=truncator,
            led=led,
            config=config.orchestrator,
            tts_config=config.tts,
            audio_config=config.audio,
            memory_storage=memory_storage,
            session_id=session_id,
            token_counter=token_counter,
        )
        return SessionComponents(orchestrator=orchestrator, history=history, session_id=session_id)

    # --- Memory write callback ---
    def on_session_end(session_id: str, started_at: str) -> None:
        write_executor.submit(memory_writer.process_session, session_id, started_at)

    # --- SessionManager ---
    sm = SessionManager(
        audio_input=audio_input,
        wakeword=wakeword,
        session_factory=session_factory,
        cpp_bridge=bridge,
        led=led,
        config=config.session,
        greeting_audio_path=greeting_paths.greeting,
        farewell_audio_path=greeting_paths.farewell,
        audio_queue=audio_queue,
        on_session_end=on_session_end,
    )

    # --- Signal handling ---
    def _handle_signal(*_: object) -> None:
        sm.shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_signal)

    # --- Run ---
    try:
        sm.run()
    finally:
        for wrapper in prev_async:
            wrapper.stop()
        prev_async.clear()
        write_executor.shutdown(wait=True)
        executor.shutdown(wait=True)
        asr.stop()
        bridge.disconnect()
        wakeword.close()
        led.close()
        memory_storage.close()


if __name__ == "__main__":
    main()
