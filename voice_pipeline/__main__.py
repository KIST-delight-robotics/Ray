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
from concurrent.futures import ThreadPoolExecutor

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.wakeword import WakewordDetector
from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.context.context_builder import ContextBuilder
from voice_pipeline.core.config import PipelineConfig
from voice_pipeline.generation.speech_generator import SpeechGenerator
from voice_pipeline.history.conversation_history import ConversationHistory
from voice_pipeline.history.storage_backend import create_storage_backend
from voice_pipeline.led.led_controller import LEDController
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.llm.token_counter import create_token_counter
from voice_pipeline.llm.tools import get_tools_token_cost
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

        async_vap = AsyncVAP(vap)
        async_turngpt = AsyncTurnGPT(turngpt)
        prev_async.extend([async_vap, async_turngpt])

        history = ConversationHistory(storage, token_counter)
        context_builder = ContextBuilder(
            history,
            config.history,
            DEFAULT_SYSTEM_PROMPT,
            token_counter,
            tools_token_cost=tools_token_cost,
        )
        turn_detector = TurnDetector(
            async_vap,
            async_turngpt,
            similarity,
            config.turn_detector,
            config.similarity,
            config.audio,
        )
        generator = SpeechGenerator(context_builder, llm, tts, config.speech_generator, executor)
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
        )
        return SessionComponents(orchestrator=orchestrator, history=history)

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
        executor.shutdown(wait=True)
        asr.stop()
        bridge.disconnect()
        wakeword.close()
        led.close()


if __name__ == "__main__":
    main()
