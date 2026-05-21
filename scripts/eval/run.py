"""E2E evaluation runner for the voice pipeline.

Plays pre-generated question WAV files through a physical speaker,
lets the full pipeline process them, and records session mappings.
Actual turn data (ASR text, response, latency) is captured by the
existing storage systems (trace_store, history, memory_storage).

Usage:
    uv run python scripts/eval/run.py \\
        --questions data/eval/questions.json \\
        --device plughw:1,0 \\
        --output-dir data/eval/results
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import queue
import random
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Suppress ALSA/JACK noise during PyAudio initialization.
_alsa_error_handler = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)(lambda *_: None)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_alsa_error_handler)
except Exception:
    _asound = None

import torch
from question_player import QuestionPlayer
from silero_vad import load_silero_vad

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.constants import SAMPLE_RATE
from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.core.types import AudioFrame
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
from voice_pipeline.memory.storage import _DEFAULT_DIMENSION, SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.session_loop import SessionComponents, SessionLoop
from voice_pipeline.trace.trace_store import SQLiteTraceStore
from voice_pipeline.tts.tts import OpenAITTS
from voice_pipeline.turn_taking.async_turngpt import AsyncTurnGPT
from voice_pipeline.turn_taking.async_vap import AsyncVAP
from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper
from voice_pipeline.turn_taking.turn_detector import TurnDetector
from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

logger = logging.getLogger("eval")

_AUDIO_QUEUE_SIZE = 300
_STARTUP_DELAY_SEC = 1.5
_TURN_TIMEOUT_SEC = 60.0
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain_audio_queue(audio_queue: queue.Queue[AudioFrame]) -> None:
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def _load_manifest(questions_path: str, wav_dir: str) -> dict[str, str]:
    """Load WAV manifest. Falls back to <wav_dir>/<id>.wav convention."""
    manifest_path = Path(wav_dir) / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    # Fallback: derive from question IDs
    data = json.loads(Path(questions_path).read_text())
    manifest = {}
    for suite in data["suites"]:
        for q in suite["questions"]:
            wav_path = Path(wav_dir) / f"{q['id']}.wav"
            if wav_path.exists():
                manifest[q["id"]] = str(wav_path)
    return manifest


# ---------------------------------------------------------------------------
# Single-turn execution
# ---------------------------------------------------------------------------


def _run_single_turn(
    suite: dict,
    question: dict,
    wav_path: str,
    player: QuestionPlayer,
    session_map: list[dict],
    create_session: Callable[..., SessionComponents],
    audio_queue: queue.Queue[AudioFrame],
) -> None:
    """Execute one single-turn eval: play question, capture response."""
    _drain_audio_queue(audio_queue)

    turn_event = threading.Event()
    play_end_time = [0.0]
    turn_shift_time = [0.0]
    final_asr_text = [""]

    def on_turn_done(ts_time: float, asr_text: str) -> None:
        turn_shift_time[0] = ts_time
        final_asr_text[0] = asr_text
        turn_event.set()
        components.session_loop.request_stop()

    skip = suite.get("category") == "asr"
    components = create_session(
        on_turn_complete=on_turn_done,
        memory_enabled=suite.get("memory", False),
        skip_generation=skip,
    )
    components.history.new_session(components.session_id)

    def play_with_delay() -> None:
        time.sleep(_STARTUP_DELAY_SEC)
        try:
            player.play(wav_path)
            play_end_time[0] = time.monotonic()
        except Exception:
            logger.error("Failed to play %s", wav_path, exc_info=True)

    player_thread = threading.Thread(target=play_with_delay, daemon=True)
    player_thread.start()

    try:
        components.session_loop.run()
    except Exception:
        logger.error("SessionLoop error", exc_info=True)
    finally:
        components.history.save()

    player_thread.join(timeout=5.0)

    success = turn_event.is_set()
    early_turn_shift = success and play_end_time[0] == 0.0
    vap_delay = None
    if not early_turn_shift and play_end_time[0] > 0 and turn_shift_time[0] >= play_end_time[0]:
        vap_delay = round((turn_shift_time[0] - play_end_time[0]) * 1000, 1)

    error = None
    if not success:
        error = "no_response"
    elif early_turn_shift:
        error = "early_turn_shift"

    session_map.append(
        {
            "question_id": question["id"],
            "session_id": components.session_id,
            "suite_name": suite["name"],
            "input_text": question["text"],
            "asr_text": final_asr_text[0],
            "success": success and not early_turn_shift,
            "error": error,
            "turn_detection_delay_ms": vap_delay,
        }
    )

    status = "OK" if success else "FAIL"
    logger.info("[%s] %s: %s", status, question["id"], question["text"][:60])


# ---------------------------------------------------------------------------
# Interruption execution
# ---------------------------------------------------------------------------


def _run_interruption(
    suite: dict,
    question: dict,
    wav_path: str,
    interrupt_wav_path: str,
    interrupt_delay_sec: float,
    player: QuestionPlayer,
    session_map: list[dict],
    create_session: Callable[..., SessionComponents],
    audio_queue: queue.Queue[AudioFrame],
    interrupt_audio: str = "",
) -> None:
    """Execute one interruption test: play question, wait for response, interrupt."""
    _drain_audio_queue(audio_queue)

    turn_event = threading.Event()
    playback_started_event = threading.Event()
    play_end_time = [0.0]
    turn_shift_time = [0.0]
    interrupted = [False]

    def on_turn_done(ts_time: float, asr_text: str) -> None:
        turn_shift_time[0] = ts_time
        turn_event.set()
        components.session_loop.request_stop()

    def on_playback_started() -> None:
        playback_started_event.set()

    components = create_session(
        on_turn_complete=on_turn_done,
        on_playback_started=on_playback_started,
        memory_enabled=suite.get("memory", False),
    )
    components.history.new_session(components.session_id)

    def play_and_interrupt() -> None:
        time.sleep(_STARTUP_DELAY_SEC)
        try:
            player.play(wav_path)
            play_end_time[0] = time.monotonic()
        except Exception:
            logger.error("Failed to play %s", wav_path, exc_info=True)
            return

        if not playback_started_event.wait(timeout=_TURN_TIMEOUT_SEC):
            logger.error("Playback never started for %s", question["id"])
            return
        time.sleep(interrupt_delay_sec)

        try:
            player.play(interrupt_wav_path)
            interrupted[0] = True
        except Exception:
            logger.error("Failed to play interrupt %s", interrupt_wav_path, exc_info=True)

    player_thread = threading.Thread(target=play_and_interrupt, daemon=True)
    player_thread.start()

    try:
        components.session_loop.run()
    except Exception:
        logger.error("SessionLoop error", exc_info=True)
    finally:
        components.history.save()

    player_thread.join(timeout=10.0)

    success = turn_event.is_set()
    session_map.append(
        {
            "question_id": question["id"],
            "session_id": components.session_id,
            "suite_name": suite["name"],
            "input_text": question["text"],
            "interrupt_audio": interrupt_audio,
            "interrupt_delay_sec": interrupt_delay_sec,
            "interrupted": interrupted[0],
            "success": success,
            "error": None if success else "no_response",
            "turn_detection_delay_ms": None,
        }
    )

    status = "OK" if success else "FAIL"
    logger.info(
        "[%s] %s: delay=%.1fs int=%s interrupted=%s",
        status,
        question["id"],
        interrupt_delay_sec,
        interrupt_audio,
        interrupted[0],
    )


# ---------------------------------------------------------------------------
# Multi-turn execution
# ---------------------------------------------------------------------------


def _run_multi_turn_suite(
    suite: dict,
    wav_map: dict[str, str],
    player: QuestionPlayer,
    session_map: list[dict],
    create_session: Callable[..., SessionComponents],
    audio_queue: queue.Queue[AudioFrame],
) -> None:
    """Execute a multi-turn suite: all questions in one session."""
    _drain_audio_queue(audio_queue)

    questions = suite["questions"]
    turn_event = threading.Event()
    turn_index = [0]
    turn_shift_times: dict[int, float] = {}
    turn_asr_texts: dict[int, str] = {}

    def on_turn_done(ts_time: float, asr_text: str) -> None:
        turn_shift_times[turn_index[0]] = ts_time
        turn_asr_texts[turn_index[0]] = asr_text
        turn_index[0] += 1
        if turn_index[0] >= len(questions):
            components.session_loop.request_stop()
        turn_event.set()

    skip = suite.get("category") == "asr"
    components = create_session(
        on_turn_complete=on_turn_done,
        memory_enabled=suite.get("memory", False),
        skip_generation=skip,
    )
    components.history.new_session(components.session_id)

    play_end_times: dict[str, float] = {}

    def play_sequence() -> None:
        time.sleep(_STARTUP_DELAY_SEC)
        for i, q in enumerate(questions):
            wav_path = wav_map.get(q["id"])
            if wav_path is None:
                logger.error("No WAV for question %s", q["id"])
                break
            if i > 0:
                turn_event.clear()
            try:
                player.play(wav_path)
                play_end_times[q["id"]] = time.monotonic()
            except Exception:
                logger.error("Failed to play %s", wav_path, exc_info=True)
                break
            if i < len(questions) - 1 and not turn_event.wait(timeout=_TURN_TIMEOUT_SEC):
                logger.error("Turn timeout at question %s", q["id"])
                components.session_loop.request_stop()
                break

    player_thread = threading.Thread(target=play_sequence, daemon=True)
    player_thread.start()

    try:
        components.session_loop.run()
    except Exception:
        logger.error("SessionLoop error", exc_info=True)
    finally:
        components.history.save()

    player_thread.join(timeout=5.0)

    completed_turns = turn_index[0]
    for i, q in enumerate(questions):
        vap_delay = None
        pe = play_end_times.get(q["id"], 0.0)
        ts = turn_shift_times.get(i, 0.0)
        if pe > 0 and ts >= pe:
            vap_delay = round((ts - pe) * 1000, 1)
        session_map.append(
            {
                "question_id": q["id"],
                "session_id": components.session_id,
                "suite_name": suite["name"],
                "input_text": q["text"],
                "asr_text": turn_asr_texts.get(i, ""),
                "success": i < completed_turns,
                "error": None if i < completed_turns else "incomplete",
                "turn_detection_delay_ms": vap_delay,
            }
        )

    logger.info(
        "Suite %s: %d/%d turns completed",
        suite["name"],
        completed_turns,
        len(questions),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E evaluation runner")
    parser.add_argument("--questions", required=True, help="Path to questions JSON")
    parser.add_argument("--device", default="default", help="ALSA device for question playback")
    parser.add_argument("--output-dir", default="data/eval/results", help="Output directory")
    parser.add_argument("--wav-dir", default="data/eval/wav", help="Directory with question WAVs")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 question per suite")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-40s %(levelname)-7s %(message)s",
    )
    for entry in os.environ.get("LOG_LEVEL", "").split(","):
        entry = entry.strip()
        if "=" in entry:
            name, level = entry.split("=", 1)
            logging.getLogger(name.strip()).setLevel(level.strip().upper())

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load questions & WAV manifest ---
    questions_data = json.loads(Path(args.questions).read_text())
    wav_map = _load_manifest(args.questions, args.wav_dir)

    missing = []
    for suite in questions_data["suites"]:
        for q in suite["questions"]:
            if q["id"] not in wav_map:
                missing.append(q["id"])
    if missing:
        logger.error("Missing WAV files for: %s", ", ".join(missing))
        logger.error("Run prepare_audio.py first")
        sys.exit(1)

    # --- Module initialization ---
    language_code = "en-US"
    eval_db = str(output_dir / "eval.db")

    asr = GoogleCloudASR(language_code=language_code)
    llm = OpenAILLM(
        model="gpt-5.4",
        temperature=0.7,
        reasoning_effort="none",
        max_tokens=256,
        tools=["web_search"],
    )
    tts = OpenAITTS()
    vap = MaAIVAPWrapper(tts.output_sample_rate)
    turngpt = TurnGPTWrapper()
    silero_vad_model = load_silero_vad(onnx=True)
    _vad_buf = bytearray()
    _vad_last_score = [0.0]
    _vad_call_count = [0]
    _VAD_INFER_INTERVAL = 3  # 3프레임(90ms)마다 추론, 사이는 캐시 반환
    _SILERO_CHUNK_BYTES = 512 * 2  # 512 samples × 16-bit

    def vad_fn(frame: AudioFrame) -> float:
        _vad_call_count[0] += 1
        if _vad_call_count[0] % _VAD_INFER_INTERVAL != 0:
            return _vad_last_score[0]
        _vad_buf.extend(frame)
        while len(_vad_buf) >= _SILERO_CHUNK_BYTES:
            chunk = bytes(_vad_buf[:_SILERO_CHUNK_BYTES])
            del _vad_buf[:_SILERO_CHUNK_BYTES]
            samples = torch.frombuffer(bytearray(chunk), dtype=torch.int16).float() / 32768.0
            _vad_last_score[0] = silero_vad_model(samples, SAMPLE_RATE).item()
        return _vad_last_score[0]

    bridge = CppBridge()
    led = LEDController()
    storage = create_storage_backend("sqlite", db_path=eval_db)
    executor = ThreadPoolExecutor(max_workers=SpeechGenerator.MAX_WORKERS)
    token_counter = create_token_counter(llm.model)
    tools_token_cost = get_tools_token_cost(llm.tools)

    embedder = create_embedder(expected_dimension=_DEFAULT_DIMENSION)
    memory_storage = SQLiteMemoryStorage(eval_db)
    trace_store = SQLiteTraceStore(eval_db)
    vector_index = NumpyVectorIndex()

    AudioInput._DEVICE_INDEX = 1
    AudioInput._CAPTURE_CHANNELS = 6
    AudioInput._EXTRACT_CHANNEL = 0

    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=_AUDIO_QUEUE_SIZE)
    audio_input = AudioInput(audio_queue)

    if _asound is not None:
        _asound.snd_lib_error_set_handler(None)

    # --- Session factory ---
    prev_async: list[AsyncVAP | AsyncTurnGPT] = []
    shutdown_event = threading.Event()

    def create_session(
        *,
        on_turn_complete: Callable[[float], None] | None = None,
        on_playback_started: Callable[[], None] | None = None,
        memory_enabled: bool = True,
        skip_generation: bool = False,
    ) -> SessionComponents:
        for wrapper in prev_async:
            wrapper.stop()
        prev_async.clear()

        vap.reset()
        turngpt.reset()

        session_id = str(uuid.uuid4())
        async_vap = AsyncVAP(vap)
        async_turngpt = AsyncTurnGPT(turngpt)
        prev_async.extend([async_vap, async_turngpt])

        ms = memory_storage if memory_enabled else None
        history = ConversationHistory(storage, token_counter)
        retriever = MemoryRetriever(memory_storage, vector_index, embedder) if memory_enabled else None
        turn_detector = TurnDetector(async_vap, async_turngpt, embedder, vad_fn=vad_fn)
        generator = SpeechGenerator(
            llm,
            tts,
            history,
            token_counter,
            DEFAULT_SYSTEM_PROMPT,
            executor,
            tools_token_cost=tools_token_cost,
            memory_storage=ms,
            retriever=retriever,
            session_id=session_id,
        )
        session_loop = SessionLoop(
            asr=asr,
            turn_detector=turn_detector,
            speech_generator=generator,
            cpp_bridge=bridge,
            history=history,
            led=led,
            audio_queue=audio_queue,
            tts_sample_rate=tts.output_sample_rate,
            memory_storage=ms,
            session_id=session_id,
            token_counter=token_counter,
            trace_store=trace_store,
            shutdown_event=shutdown_event,
            on_turn_complete=on_turn_complete,
            on_playback_started=on_playback_started,
            disable_exit_keywords=True,
            skip_generation=skip_generation,
        )
        return SessionComponents(
            session_loop=session_loop,
            history=history,
            session_id=session_id,
        )

    # --- Signal handling ---
    def _handle_signal(*_: object) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_signal)

    # --- Run eval ---
    player = QuestionPlayer(args.device)
    session_map: list[dict] = []
    started_at = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)

    logger.info("Eval starting — %d suites", len(questions_data["suites"]))
    bridge.connect()
    audio_input.start()

    try:
        for suite in questions_data["suites"]:
            if shutdown_event.is_set():
                break

            category = suite.get("category", "")
            if category.startswith("_"):
                continue

            questions = suite["questions"]
            if args.quick and len(questions) > 1:
                questions = random.sample(questions, 1)

            logger.info("Suite: %s (%d/%d questions)", suite["name"], len(questions), len(suite["questions"]))

            if suite.get("category") == "interruption":
                delays = suite.get("interrupt_delays_sec", [2.0])
                interrupt_audios = suite.get("interrupt_audios", [])
                if not interrupt_audios:
                    logger.error("Suite %s has no interrupt_audios", suite["name"])
                    continue
                for delay in delays:
                    for int_id in interrupt_audios:
                        interrupt_wav = wav_map.get(int_id, "")
                        if not interrupt_wav:
                            logger.error("No WAV for interrupt %s", int_id)
                            continue
                        for question in questions:
                            if shutdown_event.is_set():
                                break
                            wav_path = wav_map[question["id"]]
                            _run_interruption(
                                suite,
                                question,
                                wav_path,
                                interrupt_wav,
                                delay,
                                player,
                                session_map,
                                create_session,
                                audio_queue,
                                interrupt_audio=int_id,
                            )
            elif suite.get("multi_turn"):
                sampled_suite = {**suite, "questions": questions}
                _run_multi_turn_suite(
                    sampled_suite,
                    wav_map,
                    player,
                    session_map,
                    create_session,
                    audio_queue,
                )
            else:
                for question in questions:
                    if shutdown_event.is_set():
                        break
                    wav_path = wav_map[question["id"]]
                    _run_single_turn(
                        suite,
                        question,
                        wav_path,
                        player,
                        session_map,
                        create_session,
                        audio_queue,
                    )
    finally:
        audio_input.stop()
        bridge.disconnect()
        for wrapper in prev_async:
            wrapper.stop()
        prev_async.clear()
        executor.shutdown(wait=False)
        asr.stop()
        led.close()
        memory_storage.close()
        trace_store.close()

    # --- Save session mapping ---
    finished_at = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
    successful = sum(1 for s in session_map if s["success"])
    result = {
        "started_at": started_at,
        "finished_at": finished_at,
        "total": len(session_map),
        "successful": successful,
        "failed": len(session_map) - successful,
        "eval_db": eval_db,
        "sessions": session_map,
    }
    sessions_path = output_dir / "sessions.json"
    sessions_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    logger.info(
        "Eval complete: %d/%d successful — %s",
        successful,
        len(session_map),
        sessions_path,
    )

    # --- Report & Score & Dashboard ---
    from dashboard import build_html
    from report import build_report, print_summary
    from score import print_scores, score_report

    report = build_report(output_dir)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    scored = score_report(report)
    scored_path = output_dir / "scored.json"
    scored_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False))

    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(build_html(scored))

    print_summary(report)
    print_scores(scored)
    logger.info("Dashboard: %s", dashboard_path)


if __name__ == "__main__":
    main()
