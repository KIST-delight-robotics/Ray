"""Main entry point for the voice pipeline.

Usage:
    uv run ray              # via project.scripts entry point
    uv run python -m voice_pipeline  # as module
"""

from __future__ import annotations

import logging
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
from voice_pipeline.history.storage_backend import MemoryStorageBackend
from voice_pipeline.led.led_controller import LEDController
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.llm.token_counter import create_token_counter
from voice_pipeline.orchestrator.orchestrator import Orchestrator
from voice_pipeline.session.session_manager import SessionComponents, SessionManager
from voice_pipeline.tts.tts import OpenAITTS
from voice_pipeline.tts.utterance_truncator import TimestampTruncator
from voice_pipeline.turn_taking.turn_detector import TurnDetector
from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper
from voice_pipeline.turn_taking.vap import VAPWrapper


def main() -> None:
    """Launch the voice pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = PipelineConfig()

    # --- Process-level singletons (expensive init, reused across sessions) ---
    asr = GoogleCloudASR(config.asr, config.audio)
    llm = OpenAILLM(config.llm)
    tts = OpenAITTS(config.tts)
    vap = VAPWrapper(config.vap, config.audio, config.tts)
    turngpt = TurnGPTWrapper(config.turngpt)
    bridge = CppBridge(config.cpp_bridge)
    wakeword = WakewordDetector(config.wakeword, config.audio)
    led = LEDController(config.led)
    storage = MemoryStorageBackend()
    executor = ThreadPoolExecutor(max_workers=config.speech_generator.max_workers)
    token_counter = create_token_counter(config.llm.model)

    # --- Audio queue + input ---
    audio_queue = queue.Queue(maxsize=config.session.audio_queue_size)
    audio_input = AudioInput(audio_queue, config.audio, config.audio_input)

    # --- Session factory: creates fresh per-session components ---
    def session_factory() -> SessionComponents:
        vap.reset()
        turngpt.reset()
        history = ConversationHistory(storage)
        context_builder = ContextBuilder(
            history, config.history, DEFAULT_SYSTEM_PROMPT, token_counter
        )
        turn_detector = TurnDetector(vap, turngpt, config.turn_detector, config.audio)
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
        executor.shutdown(wait=True)
        asr.stop()
        bridge.disconnect()
        wakeword.close()
        led.close()


if __name__ == "__main__":
    main()
